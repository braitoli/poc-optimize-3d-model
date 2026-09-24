"""
face_reduce.py

Face repair & reduction (pipeline Step 3), with two interchangeable engines:
- "meshlab": pymeshlab, the MeshLab filter set (https://github.com/cnr-isti-vclab/meshlab)
- "cgal":    the optimizer/cgal/build/mesh_repair helper, CGAL Polygon Mesh Processing
             (https://doc.cgal.org/latest/PMP_Mesh_repair/index.html)

Operations, always applied in this order and each one optional:
- repair             degenerate, duplicate and non-manifold faces
- self_intersection  faces of a self-intersecting pair that no camera sees (a visible one is kept:
                     cutting it would leave a hole where you can see it)
- isolated           connected components smaller than `isolated_min_faces`
- hidden             faces never the closest surface from any of `hidden_views` directions
- merge              quadric edge collapse (MeshLab) / Garland-Heckbert edge collapse (CGAL)

Quality budget: the surface deviation of the result from the input mesh, measured as the maximum
distance from points sampled on the input to the result's surface and expressed as a percentage of
the input's bounding box diagonal, must stay within `quality_budget_percent`; and the angle between
the original surface and the one that replaced it must stay within `normal_budget_degrees` at the
99th percentile, because a collapse can round a crease away while barely moving the surface. That
second budget defaults to "auto", which reads the angle off the mesh itself. The four removal
operations are measured together and abort the pipeline when they exceed it; `merge` instead
searches for the smallest face count that stays within it.

Only `merge` changes topology, and it invalidates the mesh's UVs (the caller must re-chart):
the removal operations keep every surviving face's vertices and UVs untouched.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import trimesh

from optimizer.core.errors import PipelineAbort
from optimizer.core.shell_orient import _sphere_dirs, face_visibility

# CGAL first: it is the default because at a tight quality budget its Garland-Heckbert collapse
# reduces far more than MeshLab's quadric one, which leaves a handful of spots off by a few tenths
# of a percent and so gets vetoed by the budget everywhere (measured: koidrax at 0.1%, -66% vs -6%)
ENGINES = ("cgal", "meshlab")
OPS = ("repair", "self_intersection", "isolated", "hidden", "merge")
# Self-intersection removal is opt-in: on an AI shell it only cuts what `hidden` would cut anyway,
# because the visible face of an intersecting pair is never removed.
DEFAULT_OPS = ("repair", "isolated", "hidden", "merge")
DEFAULT_QUALITY_BUDGET_PERCENT = 0.1
# Shading, not position, is what shows a crease: a collapse can round a sharp edge off while moving
# the surface by almost nothing, so the budget bounds the angle between the original surface and
# the result as well.
#
# 20 degrees, measured. It is the widest setting at which the rate of faces lit unlike their own
# neighbours - the defect an eye actually catches - stays at the floor on every model tried:
# per 10,000 faces, dinoki 0.7, vosiruto 1.2, zelvaron 3.4. At 30 the two hard-surface ones jump
# to 5.4 and 8.5. It is also most of what the setting can do: past roughly 30 degrees the 0.1%
# surface budget takes over as the binding one, and zelvaron at 45 comes out identical to 30.
DEFAULT_NORMAL_BUDGET_DEGREES = 10.0
# Which part of the surface the shading budget answers for. At the 95th percentile a glossy plate
# covering a few percent of the model can be flattened without the number moving at all, and a
# specular highlight breaking into facets is exactly what gets noticed, so it is the 99th.
NORMAL_BUDGET_PERCENTILE = 99.0
# The surface budget is read at the same percentile, and the worst point anywhere is held to this
# multiple of it. A single pathological spot otherwise vetoes every candidate - measured on more
# than one model, where the distance plateaus at ~1.5x the budget however mild the collapse is,
# and the run then returns the mesh untouched while 99% of it was well inside the budget.
MAX_DEVIATION_FACTOR = 10.0
DEFAULT_ISOLATED_MIN_FACES = 25
DEFAULT_HIDDEN_VIEWS = 64
DEFAULT_HIDDEN_RESOLUTION = 448
# Sample points on the input mesh for the deviation measurement. The draw is seeded: the same
# model and settings have to reduce to the same mesh every run, and the merge search compares
# candidates against each other, which only means anything on one fixed cloud.
DEVIATION_SAMPLES = 60000
DEVIATION_SEED = 20260923
# Bisections of the merge target face count (each one runs the engine and measures the deviation)
MERGE_SEARCH_ITERATIONS = 6
# The merge search never goes below this fraction of the input face count
MERGE_MIN_RATIO = 0.01

_cgal_default = Path(__file__).resolve().parents[1] / "cgal" / "build" / "mesh_repair"
_cgal_which = shutil.which("mesh_repair")
CGAL_HELPER = _cgal_default if _cgal_default.exists() else (Path(_cgal_which) if _cgal_which else _cgal_default)
CGAL_UNAVAILABLE = (
    f"The CGAL engine needs the mesh_repair helper at {CGAL_HELPER}: build it with "
    f"optimizer/cgal/build.sh (or run ./setup.sh), or switch the engine to 'meshlab' "
    f"(which reduces far less at a tight quality budget)"
)
MESHLAB_UNAVAILABLE = "The MeshLab engine needs the 'pymeshlab' package (pip install pymeshlab)"


def engine_available(engine: str) -> Tuple[bool, str]:
    """(True, "") when `engine` can run here, (False, reason) otherwise."""
    if engine == "meshlab":
        try:
            import pymeshlab  # noqa: F401
        except ImportError as err:
            return False, f"{MESHLAB_UNAVAILABLE}: {err}"
        return True, ""
    if engine == "cgal":
        if not CGAL_HELPER.exists():
            return False, CGAL_UNAVAILABLE
        if not os.access(CGAL_HELPER, os.X_OK):
            return False, f"The CGAL mesh_repair helper at {CGAL_HELPER} is not executable"
        return True, ""
    raise ValueError(f"Unsupported engine '{engine}' (expected one of {', '.join(ENGINES)})")


def validate_normal_budget(budget: float) -> float:
    """`budget` unchanged when it is a usable shading budget, ValueError otherwise.

    An angle above 0 and at most 90; 90 switches the budget off, since no collapse can turn a
    surface further than that."""
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 90:
        raise ValueError(
            f"normal_budget_degrees must be an angle above 0 and at most 90, got {budget!r}"
        )
    return float(budget)


def validate_ops(ops: Sequence[str]) -> Tuple[str, ...]:
    """The requested operations in canonical order; ValueError on an unknown name."""
    unknown = [op for op in ops if op not in OPS]
    if unknown:
        raise ValueError(f"Unsupported face reduction op(s) {', '.join(unknown)} (expected one of {', '.join(OPS)})")
    return tuple(op for op in OPS if op in set(ops))


class _Deviation:
    """Distance from a fixed point cloud sampled on the reference mesh to a candidate's surface,
    as a percentage of the reference's bounding box diagonal. open3d builds the BVH per candidate;
    the sample points are drawn once so every candidate is measured against the same cloud.

    The cloud covers only the faces a camera can see (`visible`): quality here is what the rendered
    model looks like, and deleting geometry buried inside the model changes nothing on screen,
    however far it is from the surface that remains."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        visible: np.ndarray,
        samples: int = DEVIATION_SAMPLES
    ):
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        self.diagonal = float(np.linalg.norm(extent))
        if not np.isfinite(self.diagonal) or self.diagonal <= 0:
            raise PipelineAbort(
                f"Cannot measure face reduction quality: the mesh bounding box has zero size (diagonal {self.diagonal})"
            )
        reference = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        weight = reference.area_faces * visible
        total = float(weight.sum())
        if not total > 0:
            raise PipelineAbort(
                "Cannot measure face reduction quality: no visible face of the mesh has an area"
            )
        points, sampled_faces = trimesh.sample.sample_surface(
            reference, samples, face_weight=weight, seed=DEVIATION_SEED
        )
        self.points = np.asarray(points, dtype=np.float32)
        self.normals = np.asarray(reference.face_normals[sampled_faces], dtype=np.float64)

    def of(self, vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
        import open3d as o3d

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices.astype(np.float32)),
            o3d.core.Tensor(faces.astype(np.uint32))
        ))
        closest = scene.compute_closest_points(o3d.core.Tensor(self.points))
        distances = np.linalg.norm(self.points - closest["points"].numpy(), axis=1).astype(np.float64)

        # Angle between the surface the reference had there and the one that replaced it
        triangles = vertices[faces[closest["primitive_ids"].numpy().astype(np.int64)]]
        result_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        result_normals /= np.maximum(np.linalg.norm(result_normals, axis=1, keepdims=True), 1e-20)
        cosine = np.abs((self.normals * result_normals).sum(axis=1))
        angles = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

        return {
            "maxPercent": round(float(distances.max()) / self.diagonal * 100.0, 4),
            "percentilePercent": round(
                float(np.percentile(distances, NORMAL_BUDGET_PERCENTILE)) / self.diagonal * 100.0, 4
            ),
            "rmsPercent": round(float(np.sqrt((distances ** 2).mean())) / self.diagonal * 100.0, 4),
            "normalDeviationDegrees": round(float(np.percentile(angles, NORMAL_BUDGET_PERCENTILE)), 3),
            "normalMedianDegrees": round(float(np.median(angles)), 3)
        }


class SourceUVProjector:
    """
    Reads the UVs of a mesh whose topology no longer exists.

    `merge` collapses edges, so the reduced mesh has new vertices with no UV of their own. This
    projects any 3D point onto the original mesh (closest point, open3d BVH) and interpolates the
    original UVs there, which is what the Step 4 bake needs to fetch colours out of the original
    texture: per texel it is exact, so a reduced triangle spanning two islands of the original
    atlas still samples the right pixels on both sides.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        import open3d as o3d

        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        if uv is None or len(uv) != len(mesh.vertices):
            raise PipelineAbort(
                "Cannot project UVs from the original mesh: it has no UV per vertex to project"
            )
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.area_faces = np.asarray(mesh.area_faces, dtype=np.float64)
        self.diagonal = float(np.linalg.norm(self.vertices.max(axis=0) - self.vertices.min(axis=0)))
        self.uv = np.asarray(uv, dtype=np.float64)
        # The model's hard edges live in these: a glTF mesh marks a crease by splitting the vertex
        # and giving each side its own normal, so they have to be carried over as well
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(self.faces.astype(np.uint32))
        ))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The original mesh's UV at the point closest to each of `points` (n, 3) -> (n, 2)."""
        return self._interpolate(points, self.uv)

    def _interpolate(self, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """`values` (one row per original vertex) read at the point of the original mesh closest
        to each of `points`."""
        import open3d as o3d

        closest = self.scene.compute_closest_points(
            o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        )
        triangles = closest["primitive_ids"].numpy().astype(np.int64)
        uv_bary = closest["primitive_uvs"].numpy().astype(np.float64)
        weights = np.column_stack([1.0 - uv_bary.sum(axis=1), uv_bary])
        return (values[self.faces[triangles]] * weights[:, :, None]).sum(axis=1)

    def corner_normals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        The original mesh's shading normal at each corner of each face of `mesh`, (F, 3, 3).

        Per corner, not per vertex: a crease is precisely where the two sides disagree, so a single
        normal per position would average the very discontinuity that draws the edge. Each corner is
        sampled a little inside its own face, which keeps it on that side of the crease.

        One sample per corner, which is only as good as where that one sample lands. See
        `footprint_corner_normals`, which averages the whole patch the corner replaced instead and
        is what Step 3 actually uses; this remains the fallback for corners no patch reaches.
        """
        triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
        inset = triangles * (1.0 - CORNER_NORMAL_INSET) + triangles.mean(axis=1, keepdims=True) * CORNER_NORMAL_INSET
        normals = self._interpolate(inset.reshape(-1, 3), self.vertex_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        return normals.reshape(len(mesh.faces), 3, 3)

    def footprint_corner_normals(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
        """
        (corner normals (F, 3, 3), fraction of corners covered) for `mesh`, averaged over the
        patch of the original surface each corner replaced.

        A collapse replaces a whole patch of the original with one triangle, so the normal that
        triangle should carry is what that patch averaged, weighted by how much surface each of its
        faces contributed. `corner_normals` instead reads a single point per corner, and a single
        point lands where it lands: on a model with fine panel lines it drops onto the far side of
        one often enough to leave individual triangles lit unlike every neighbour they have -
        measured on zelvaron at 15,652 faces, 71 faces more than 40 degrees away from their own
        neighbourhood, against 1 here, and 378 over 20 degrees against 21.

        Each original face contributes four samples carrying its own normal and a quarter of its
        AREA, so a large triangle counts for more than a small one. Each sample is projected onto
        this mesh and pushed towards the corner of the triangle it landed nearest
        (FOOTPRINT_CORNER_SHARPNESS); spreading it evenly over all three corners instead blurs the
        shading noticeably.

        Two filters keep a sample from voting on a patch it never belonged to: it must land on a
        face pointing the same way it does (the closest face to an interior sample is routinely the
        far side of a wall, whose normal then cancels the near side's out - half of them on
        zelvaron), and within FOOTPRINT_MAX_DISTANCE of the surface.

        That first filter compares against the face's winding, so it also quietly turns the result
        to agree with it - and a collapse leaves plenty of faces wound the wrong way round (67,383
        of gravilux's 80,931). Agreeing with a wrong winding hides it from `align_faces_outward`,
        which repairs a face precisely by noticing that the two disagree, so the face stays
        inside-out and is culled: the model comes out speckled with holes. Each corner therefore
        takes its side from the single-point sample, which is read off the original and knows
        nothing about how this mesh happens to be wound.
        """
        import open3d as o3d

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = vertices[faces]
        geometric = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        geometric /= np.maximum(np.linalg.norm(geometric, axis=1, keepdims=True), 1e-20)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32))
        ))

        source = self.vertices[self.faces]
        source_normals = self.vertex_normals[self.faces]
        centre, centre_normal = source.mean(axis=1), source_normals.mean(axis=1)
        points = np.concatenate(
            [centre] + [source[:, c] * (1.0 - FOOTPRINT_SAMPLE_INSET) + centre * FOOTPRINT_SAMPLE_INSET
                        for c in range(3)]
        )
        normals = np.concatenate(
            [centre_normal] + [source_normals[:, c] * (1.0 - FOOTPRINT_SAMPLE_INSET)
                               + centre_normal * FOOTPRINT_SAMPLE_INSET for c in range(3)]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        weights = np.tile(self.area_faces / FOOTPRINT_SAMPLES_PER_FACE, FOOTPRINT_SAMPLES_PER_FACE)

        accumulated = np.zeros((len(faces), 3, 3))
        total = np.zeros((len(faces), 3))
        for start in range(0, len(points), FOOTPRINT_CHUNK):
            chunk = points[start:start + FOOTPRINT_CHUNK]
            found = scene.compute_closest_points(o3d.core.Tensor(chunk.astype(np.float32)))
            face_of = found["primitive_ids"].numpy().astype(np.int64)
            uv = found["primitive_uvs"].numpy()
            sample_normals = normals[start:start + FOOTPRINT_CHUNK]
            distance = np.linalg.norm(chunk - found["points"].numpy(), axis=1)
            usable = (
                (sample_normals * geometric[face_of]).sum(axis=1) >= FOOTPRINT_MIN_AGREEMENT
            ) & (distance <= FOOTPRINT_MAX_DISTANCE * self.diagonal)
            if not usable.any():
                continue
            face_of, uv, sample_normals = face_of[usable], uv[usable], sample_normals[usable]
            bary = np.column_stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]])
            bary = np.clip(bary, 0.0, None) ** FOOTPRINT_CORNER_SHARPNESS
            bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-20)
            share = weights[start:start + FOOTPRINT_CHUNK][usable][:, None] * bary
            for corner in range(3):
                np.add.at(accumulated[:, corner, :], face_of, sample_normals * share[:, corner][:, None])
                np.add.at(total[:, corner], face_of, share[:, corner])

        point_sampled = self.corner_normals(mesh)
        result = point_sampled.copy()
        lengths = np.linalg.norm(accumulated, axis=2)
        covered = (total > 1e-12) & (lengths > 1e-12)
        result[covered] = accumulated[covered] / lengths[covered][:, None]
        opposed = (result * point_sampled).sum(axis=2) < 0
        result[opposed] = -result[opposed]
        return result, float(covered.mean())

    def vertex_uv(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Per-vertex UVs for `mesh`, projected from the original. Used where a per-vertex UV is
        required (texel density, canvas planning, tangent frames); the bake itself samples per
        texel through __call__."""
        return self(np.asarray(mesh.vertices, dtype=np.float64))


def restore_hard_edges(mesh: trimesh.Trimesh, projector: "SourceUVProjector") -> trimesh.Trimesh:
    """
    Gives `mesh` back the shading normals of the model it was reduced from, hard edges included.

    A glTF model draws a crease by splitting the vertex there and giving each side its own normal;
    rebuilding a mesh from bare positions and faces (which is all an edge collapse returns) averages
    the two sides back together, and the crease stops reading as an edge - the surface looks lit
    from the wrong place even though the geometry, the texture and the colours are all correct.
    Here each face corner takes its normal from its own side of the crease, and a position whose
    corners disagree by more than HARD_EDGE_DEGREES keeps one vertex per distinct normal.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    corner_normals, _ = projector.footprint_corner_normals(mesh)
    # Vertices are split by (position, normal direction); the quantisation sets how close two
    # normals have to be to share a vertex, i.e. what still counts as one smooth surface
    quantum = max(np.sin(np.radians(HARD_EDGE_DEGREES)), 1e-6)
    keys = np.column_stack([
        faces.reshape(-1),
        np.round(corner_normals.reshape(-1, 3) / quantum).astype(np.int64)
    ])
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    flat_normals = corner_normals.reshape(-1, 3)
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[faces.reshape(-1)[first]],
        faces=inverse.reshape(-1, 3),
        vertex_normals=flat_normals[first],
        process=False
    )
    return out


def align_faces_outward(mesh: trimesh.Trimesh) -> Dict[str, int]:
    """
    Repairs faces whose winding and shading normals point to opposite sides, in place.

    Single-sided rendering draws a face only from the side its winding faces, and lights it by its
    normals. When the two disagree the face either vanishes (a hole you can see through) or is lit
    from behind (a black triangle) - and which of the two is wrong cannot be decided by comparing
    them to each other. So the outside is measured instead: rays are fired from the face to both
    sides, and whichever side a ray escapes from is the one outside. Then the side that disagrees
    with it is corrected - the winding by reversing the triangle, the normals by giving that face
    its own corner vertices, so no neighbour is touched. A face fully enclosed by the model is
    invisible either way; its winding is simply made to agree with its normals.
    """
    import open3d as o3d

    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    triangles = vertices[faces]
    geometric = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    geometric /= np.maximum(np.linalg.norm(geometric, axis=1, keepdims=True), 1e-20)
    shading = normals[faces].sum(axis=1)
    shading /= np.maximum(np.linalg.norm(shading, axis=1, keepdims=True), 1e-20)
    disagree = np.nonzero((geometric * shading).sum(axis=1) < 0)[0]
    if len(disagree) == 0:
        return {"windingFlipped": 0, "normalsFlipped": 0}

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32))
    ))
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    offset = diagonal * RAY_ORIGIN_OFFSET
    directions = _sphere_dirs(OUTWARD_TEST_VIEWS)

    def escapes(centre: np.ndarray, side: np.ndarray) -> float:
        """Fraction of rays leaving `centre` towards `side` that never hit the model again."""
        hemisphere = directions[directions @ side > 0.2]
        if len(hemisphere) == 0:
            return 0.0
        origin = (centre + offset * side).astype(np.float32)
        rays = np.concatenate(
            [np.repeat(origin[None, :], len(hemisphere), axis=0), hemisphere.astype(np.float32)], axis=1
        )
        return float(np.isinf(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()).mean())

    flip_winding, flip_normals = [], []
    for face in disagree:
        centre = triangles[face].mean(axis=0)
        towards_normals = escapes(centre, shading[face])
        towards_winding = escapes(centre, geometric[face])
        if towards_winding > towards_normals:
            flip_normals.append(face)     # the winding already faces out; the normals are inverted
        else:
            flip_winding.append(face)     # the normals face out, or the face is buried

    if flip_winding:
        faces[flip_winding] = faces[flip_winding][:, ::-1]
    if flip_normals:
        # Its own corner vertices, so inverting these normals cannot disturb a neighbouring face
        extra = np.arange(len(vertices), len(vertices) + 3 * len(flip_normals)).reshape(-1, 3)
        corners = faces[flip_normals]
        vertices = np.vstack([vertices, vertices[corners].reshape(-1, 3)])
        normals = np.vstack([normals, -normals[corners].reshape(-1, 3)])
        # Everything else stored per vertex has to grow with them, or the mesh no longer describes
        # one UV per vertex and Step 4 refuses it
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        if uv is not None and len(uv) == len(np.asarray(mesh.vertices)):
            mesh.visual.uv = np.vstack([np.asarray(uv), np.asarray(uv)[corners].reshape(-1, 2)])
        faces[flip_normals] = extra

    mesh.vertices = vertices
    mesh.faces = faces
    mesh.vertex_normals = normals
    return {"windingFlipped": len(flip_winding), "normalsFlipped": len(flip_normals)}


def _mesh_arrays(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


# Where a ray leaves a candidate face from: its centroid, then one point biased towards each
# corner, so a face that is only reachable through a slit at one of its edges still counts
RAY_SAMPLE_WEIGHTS = (
    (1 / 3, 1 / 3, 1 / 3),
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6)
)
RAY_CHUNK_FACES = 150000       # rays are built per chunk to bound peak memory
RAY_ORIGIN_OFFSET = 1e-4       # lift off the surface, as a fraction of the bounding box diagonal
# How far a corner is pulled towards its face centre before the original normal is read there
CORNER_NORMAL_INSET = 0.12
# Samples each original face contributes to the footprint average, and how far the three off-centre
# ones sit from their corner. Four is enough to reach into a reduced triangle's own corners.
FOOTPRINT_SAMPLES_PER_FACE = 4
FOOTPRINT_SAMPLE_INSET = 0.25
# A sample is pushed towards the corner it landed nearest by raising its barycentric weight to this
# power. At 1 a sample near the middle votes equally for all three corners and the shading blurs;
# measured on zelvaron the count of badly lit faces stops improving past 8.
FOOTPRINT_CORNER_SHARPNESS = 8
# A sample may only vote on a face pointing the same way it does, and lying this close (as a
# fraction of the source bounding box diagonal). Both reject samples from surfaces this face never
# replaced - above all the interior, whose nearest outward face is the far side of a wall.
FOOTPRINT_MIN_AGREEMENT = 0.5
FOOTPRINT_MAX_DISTANCE = 0.01
FOOTPRINT_CHUNK = 400000
# Corner normals further apart than this keep separate vertices, which is what draws a hard edge
HARD_EDGE_DEGREES = 15.0
# Directions tried when deciding which side of a face is the outside of the model
OUTWARD_TEST_VIEWS = 64
# Rounds of "a hidden face with at least two visible neighbours counts as visible"
VISIBLE_RIM_ROUNDS = 2
# Creeping into crevices: each round re-tests the hidden faces that touch a visible one, with far
# more ray directions than the first pass uses. Only that frontier is tested, so the cost stays low
VISIBLE_CREVICE_ROUNDS = 6
VISIBLE_CREVICE_VIEWS = 256


def _ray_visible(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidates: np.ndarray,
    views: int
) -> np.ndarray:
    """
    Of the `candidates` (faces the z-buffer never saw), the ones a ray can still escape from.

    The raster pass decides visibility on a pixel grid, so a face smaller than a pixel loses every
    pixel it shares with a neighbour and reads as hidden however open it is - on a 290k-face model
    at 448px that misclassifies about 9% of the mesh, and the faces it eats line the narrow gaps
    between parts, which is exactly where holes are noticed. This second pass has no grid: it
    shoots rays out of each candidate in the same directions and keeps every face at least one ray
    leaves without hitting the model again.
    """
    import open3d as o3d

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    offset = diagonal * RAY_ORIGIN_OFFSET
    directions = _sphere_dirs(views).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32))
    ))

    escaped = np.zeros(len(candidates), dtype=bool)
    candidate_triangles = triangles[candidates]
    candidate_normals = normals[candidates]
    for weights in RAY_SAMPLE_WEIGHTS:
        points = (candidate_triangles * np.asarray(weights)[None, :, None]).sum(axis=1)
        # Both sides: the winding of an AI shell is not a reliable guide to where a camera may be
        for side in (1.0, -1.0):
            origins = (points + side * offset * candidate_normals).astype(np.float32)
            for start in range(0, len(origins), RAY_CHUNK_FACES):
                chunk = origins[start:start + RAY_CHUNK_FACES]
                rays = np.concatenate([
                    np.repeat(chunk, len(directions), axis=0),
                    np.tile(directions, (len(chunk), 1))
                ], axis=1)
                hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                escaped[start:start + len(chunk)] |= np.isinf(hits.reshape(len(chunk), len(directions))).any(axis=1)
    return escaped


def _face_adjacency(vertices: np.ndarray, faces: np.ndarray):
    """Symmetric face-to-face adjacency over shared edges, on position-welded topology: a glTF mesh
    splits its vertices at every UV seam, and faces either side of a seam are neighbours all the
    same."""
    from scipy.sparse import csr_matrix

    _, welded = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    welded_faces = welded[faces]
    edges = np.sort(np.concatenate([
        welded_faces[:, [0, 1]], welded_faces[:, [1, 2]], welded_faces[:, [2, 0]]
    ]), axis=1)
    face_ids = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((face_ids, edges[:, 1], edges[:, 0]))
    sorted_edges, sorted_faces = edges[order], face_ids[order]
    shared = (sorted_edges[1:] == sorted_edges[:-1]).all(axis=1)
    adjacency = csr_matrix(
        (np.ones(int(shared.sum()), dtype=bool), (sorted_faces[:-1][shared], sorted_faces[1:][shared])),
        shape=(len(faces), len(faces))
    )
    return adjacency + adjacency.T


def _visible_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int,
    resolution: int
) -> np.ndarray:
    """Boolean mask of the faces a camera can reach: the closest surface in at least one of `views`
    z-buffer renderings, plus the ones _ray_visible finds that the pixel grid was too coarse to
    catch, plus the rims below."""
    seen_px, _ = face_visibility(vertices, faces, views=views, resolution=resolution)
    visible = seen_px > 0
    candidates = np.nonzero(~visible)[0]
    if len(candidates):
        visible[candidates] = _ray_visible(vertices, faces, candidates, views)

    adjacency = _face_adjacency(vertices, faces)

    # A slot in a helmet or the gap between two parts is only open along a narrow cone, and the
    # directions of the first pass are spread over the whole sphere, so they miss it and the whole
    # slot reads as buried. Walking inwards from the surface one ring at a time, re-testing just
    # that frontier with many more directions, follows the opening down into the crevice. Testing
    # every face that way would cost minutes; the frontier is a couple of thousand faces.
    for _ in range(VISIBLE_CREVICE_ROUNDS):
        frontier = np.nonzero((~visible) & (adjacency @ visible.astype(np.int8) >= 1))[0]
        if len(frontier) == 0:
            break
        reached = _ray_visible(vertices, faces, frontier, VISIBLE_CREVICE_VIEWS)
        if not reached.any():
            break
        visible[frontier[reached]] = True

    # A face walled in by visible surface on two or more sides belongs to that surface as far as
    # the eye is concerned - it lines the narrow gaps between parts, where every deleted face
    # leaves a hole that is plainly visible from outside.
    for _ in range(VISIBLE_RIM_ROUNDS):
        rim = (~visible) & ((adjacency @ visible.astype(np.int8)) >= 2)
        if not rim.any():
            break
        visible |= rim
    return visible


# ---------------------------------------------------------------------------------------------
# MeshLab engine (pymeshlab)
# ---------------------------------------------------------------------------------------------

def _meshlab_set(vertices: np.ndarray, faces: np.ndarray):
    """A single-mesh MeshSet whose per-face scalar holds each face's index in `faces`, so the
    surviving original indices can be read back after MeshLab's deletion filters."""
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(vertices, dtype=np.float64),
        face_matrix=np.asarray(faces, dtype=np.int32)
    ))
    mesh_set.compute_scalar_by_function_per_face(q="fi")
    return mesh_set


def _meshlab_surviving(mesh_set) -> np.ndarray:
    return np.asarray(mesh_set.current_mesh().face_scalar_array(), dtype=np.int64)


def _meshlab_remove(
    vertices: np.ndarray,
    faces: np.ndarray,
    op: str,
    isolated_min_faces: int,
    detected: Dict[str, int]
) -> np.ndarray:
    """Indices (into `faces`) of the faces MeshLab's filters for `op` keep. One operation per call,
    so the caller sees exactly what each one costs and can veto part of it."""
    mesh_set = _meshlab_set(vertices, faces)

    if op == "repair":
        mesh_set.meshing_remove_duplicate_faces()
        mesh_set.meshing_remove_null_faces()
        mesh_set.meshing_repair_non_manifold_edges(method="Remove Faces")
        mesh_set.meshing_repair_non_manifold_vertices()
        mesh_set.meshing_remove_unreferenced_vertices()
    elif op == "self_intersection":
        mesh_set.compute_selection_by_self_intersections_per_face()
        mesh_set.meshing_remove_selected_faces()
    elif op == "isolated":
        # Weld coincident vertices first: UV seams split a mesh loaded by vertex index into
        # hundreds of pieces that are not disconnected in space. Welding removes no face, so the
        # per-face indices tracked above stay valid.
        mesh_set.meshing_remove_duplicate_vertices()
        mesh_set.meshing_remove_connected_component_by_face_number(mincomponentsize=int(isolated_min_faces))
    else:
        raise ValueError(f"MeshLab has no removal step for '{op}'")

    return np.sort(_meshlab_surviving(mesh_set))


def _meshlab_merge(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric edge collapse towards `target_faces`. Boundaries are free to move: AI shells are
    open meshes whose boundary loops otherwise stop the collapse far above the target."""
    mesh_set = _meshlab_set(vertices, faces)
    # Weld first, or the collapse tears the surface apart: a glTF mesh splits its vertices at every
    # UV seam, and to the decimator each of those seams looks like a border it may pull free. On a
    # test model welding takes the result from 201 disconnected pieces with 3433 border edges to 15
    # pieces with 258 - which is also four times fewer UV islands for Step 4 to pack.
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        qualitythr=0.3,
        preservenormal=True,
        preserveboundary=False,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True
    )
    mesh = mesh_set.current_mesh()
    return np.asarray(mesh.vertex_matrix(), dtype=np.float64), np.asarray(mesh.face_matrix(), dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# CGAL engine (optimizer/cgal/build/mesh_repair)
# ---------------------------------------------------------------------------------------------

def _cgal_run(
    vertices: np.ndarray,
    faces: np.ndarray,
    args: Sequence[str],
    want_kept_faces: bool
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Runs the CGAL helper on (vertices, faces) with `args`, returning its JSON summary and either
    the surviving original face indices or the resulting mesh."""
    with tempfile.TemporaryDirectory(prefix="cgal_mesh_repair_") as tmp:
        tmp_dir = Path(tmp)
        in_ply, out_ply, kept_bin = tmp_dir / "in.ply", tmp_dir / "out.ply", tmp_dir / "kept.bin"
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(in_ply)

        cmd = [str(CGAL_HELPER), "--in", str(in_ply), "--out", str(out_ply), *args]
        if want_kept_faces:
            cmd += ["--kept-faces", str(kept_bin)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout or "").strip()
            reason = next(
                (line for line in output.splitlines() if line.startswith("Fatal mesh_repair error:")),
                f"mesh_repair exited with code {proc.returncode}"
            )
            raise PipelineAbort(f"CGAL face reduction failed: {reason.split(':', 1)[-1].strip()}")

        summary_line = next((l.strip() for l in reversed(proc.stdout.splitlines()) if l.strip().startswith("{")), None)
        if summary_line is None:
            raise PipelineAbort("CGAL face reduction failed: mesh_repair printed no JSON summary")
        try:
            summary = json.loads(summary_line)
        except json.JSONDecodeError as err:
            raise PipelineAbort(f"CGAL face reduction failed: invalid JSON summary from mesh_repair ({err})") from err

        kept = None
        if want_kept_faces:
            if not kept_bin.exists():
                raise PipelineAbort(f"CGAL face reduction failed: mesh_repair wrote no {kept_bin.name}")
            kept = np.fromfile(kept_bin, dtype=np.int32).astype(np.int64)

        mesh = None
        if not want_kept_faces:
            if not out_ply.exists():
                raise PipelineAbort(f"CGAL face reduction failed: mesh_repair wrote no {out_ply.name}")
            merged = trimesh.load(out_ply, force="mesh", process=False)
            mesh = (np.asarray(merged.vertices, dtype=np.float64), np.asarray(merged.faces, dtype=np.int64))

        return summary, kept, mesh


def _cgal_remove(
    vertices: np.ndarray,
    faces: np.ndarray,
    op: str,
    isolated_min_faces: int,
    detected: Dict[str, int]
) -> np.ndarray:
    """Indices (into `faces`) of the faces the CGAL helper's `op` keeps, one operation per call."""
    if op == "repair":
        args = ["--repair"]
    elif op == "self_intersection":
        args = ["--self-intersection"]
    elif op == "isolated":
        args = ["--isolated-min-faces", str(int(isolated_min_faces))]
    else:
        raise ValueError(f"The CGAL helper has no removal step for '{op}'")

    summary, kept, _ = _cgal_run(vertices, faces, args, want_kept_faces=True)
    if kept is None:
        raise PipelineAbort("CGAL face reduction failed: mesh_repair returned no surviving face indices")
    if op == "repair":
        # Non-manifold vertices are repaired by splitting them, which costs no face; the helper
        # still reports how many it found
        detected["nonManifoldVertexSplit"] = int(summary.get("detected", {}).get("nonManifoldVertex", 0))
    return np.sort(kept)


def _cgal_merge(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    _, _, mesh = _cgal_run(
        vertices, faces, ["--merge-target-faces", str(int(target_faces))], want_kept_faces=False
    )
    if mesh is None:
        raise PipelineAbort("CGAL face reduction failed: mesh_repair returned no merged mesh")
    return mesh


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------

# What "nothing was removed" measures as, so the budget check has every field it reads
_NO_DEVIATION = {
    "maxPercent": 0.0, "percentilePercent": 0.0, "rmsPercent": 0.0,
    "normalDeviationDegrees": 0.0, "normalMedianDegrees": 0.0
}


def _within_budget(measured: Dict[str, float], budget: float, normal_budget: float) -> bool:
    """Whether a candidate is inside both budgets: the surface and the shading at the percentile,
    and no point anywhere further than MAX_DEVIATION_FACTOR times the surface budget."""
    return (
        measured["percentilePercent"] <= budget
        and measured["maxPercent"] <= budget * MAX_DEVIATION_FACTOR
        and measured["normalDeviationDegrees"] <= normal_budget
    )


def _merge_search(
    engine: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    deviation: _Deviation,
    budget: float,
    normal_budget: float,
    stats: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    """The smallest face count the engine can collapse to while both the surface deviation and the
    shading deviation from the input stay within budget, found by bisecting the target face count.
    Returns None when even the least aggressive attempt exceeds a budget (the caller then keeps the
    unmerged mesh)."""
    merge = _meshlab_merge if engine == "meshlab" else _cgal_merge
    n_faces = len(faces)
    low = max(4, int(n_faces * MERGE_MIN_RATIO))   # most aggressive target considered
    high = n_faces                                  # the input itself: deviation 0, always feasible
    best: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = None
    attempts = []

    for _ in range(MERGE_SEARCH_ITERATIONS):
        target = int(round(math.sqrt(low * high)))  # geometric midpoint: face counts span decades
        if target >= high or target <= low:
            break
        merged_vertices, merged_faces = merge(vertices, faces, target)
        measured = deviation.of(merged_vertices, merged_faces)
        attempts.append({
            "targetFaces": target,
            "faces": int(len(merged_faces)),
            "deviationMaxPercent": measured["maxPercent"],
            "deviationPercentilePercent": measured["percentilePercent"],
            "normalDeviationDegrees": measured["normalDeviationDegrees"]
        })
        if _within_budget(measured, budget, normal_budget):
            best = (merged_vertices, merged_faces, measured)
            high = target
        else:
            low = target

    stats["mergeAttempts"] = attempts
    return best


def reduce_faces(
    mesh: trimesh.Trimesh,
    engine: str = "cgal",
    ops: Sequence[str] = DEFAULT_OPS,
    quality_budget_percent: float = DEFAULT_QUALITY_BUDGET_PERCENT,
    normal_budget_degrees: float = DEFAULT_NORMAL_BUDGET_DEGREES,
    isolated_min_faces: int = DEFAULT_ISOLATED_MIN_FACES,
    hidden_views: int = DEFAULT_HIDDEN_VIEWS,
    hidden_resolution: int = DEFAULT_HIDDEN_RESOLUTION,
    stats: Optional[Dict[str, Any]] = None,
    pre_collapse: Optional[Dict[str, Any]] = None
) -> trimesh.Trimesh:
    """
    Repairs and reduces `mesh` with `engine`, keeping the result within `quality_budget_percent`.
    `normal_budget_degrees` is how far the result may turn away from the original surface.
    Returns a new mesh; `stats` (when given) records what each operation removed, the measured
    deviation and whether the UVs survived. `pre_collapse` (when given) receives, under "mesh", the
    mesh as the removals left it, with the source UVs still on it - only set when a collapse then
    threw those UVs away, because that is the surface a caller has to measure the source against.
    Raises PipelineAbort when the engine is unavailable, when the removal operations alone exceed
    the quality budget, or when they would leave no face at all.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unsupported engine '{engine}' (expected one of {', '.join(ENGINES)})")
    ops = validate_ops(ops)
    available, reason = engine_available(engine)
    if not available:
        raise PipelineAbort(reason)
    if not quality_budget_percent > 0:
        raise ValueError(f"quality_budget_percent must be a positive number, got {quality_budget_percent!r}")
    normal_budget_degrees = validate_normal_budget(normal_budget_degrees)

    record: Dict[str, Any] = stats if stats is not None else {}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_input = len(faces)
    removed = {"repair": 0, "selfIntersection": 0, "isolated": 0, "hidden": 0}
    # Defects an engine repaired without removing a face (reported, never charged to the budget)
    detected: Dict[str, int] = {}
    record.update({
        "engine": engine,
        "ops": list(ops),
        "qualityBudgetPercent": quality_budget_percent,
        "normalBudgetDegrees": normal_budget_degrees,
        "facesBefore": n_input,
        "removed": removed,
        "detected": detected
    })
    if n_input == 0:
        raise PipelineAbort("Cannot reduce faces: the mesh has no face")

    # Which faces a camera can see: the quality budget is measured on them, and the `hidden`
    # operation removes the rest. Rendered once, on the input mesh, before anything is removed.
    visible = _visible_faces(vertices, faces, hidden_views, hidden_resolution)
    record["hiddenViews"] = int(hidden_views)
    record["hiddenResolution"] = int(hidden_resolution)
    record["visibleFaces"] = int(visible.sum())
    if not visible.any():
        raise PipelineAbort(
            f"Face reduction ({engine}): no face is visible from any of the {hidden_views} views, "
            f"so the quality of the result cannot be measured"
        )

    deviation = _Deviation(vertices, faces, visible)
    remover = _meshlab_remove if engine == "meshlab" else _cgal_remove
    kept = np.arange(n_input, dtype=np.int64)
    # What each operation cost on its own, so an abort can name the one to turn down
    op_deviation: Dict[str, Dict[str, float]] = {}
    record["opDeviation"] = op_deviation

    def apply_mask(surviving: np.ndarray) -> trimesh.Trimesh:
        face_mask = np.zeros(n_input, dtype=bool)
        face_mask[surviving] = True
        out = mesh.copy()
        if not face_mask.all():
            out.update_faces(face_mask)
            out.remove_unreferenced_vertices()
        return out

    # 1-4. One operation at a time, each measured against the mesh this step was given
    for op in ops:
        if op == "merge":
            continue
        if op == "hidden":
            dropped = np.nonzero(~visible[kept])[0]
        else:
            surviving = remover(vertices, faces[kept], op, isolated_min_faces, detected)
            dropped = np.setdiff1d(np.arange(len(kept), dtype=np.int64), surviving, assume_unique=True)
            # Only what no camera sees may go. These operations judge a face on its geometry
            # alone - a sliver, a duplicate, a component below the size threshold, one half of an
            # intersecting pair - and will cut one that is on screen, which leaves a hole. The
            # quality budget cannot veto that: a three-triangle hole moves no sampled point and
            # turns no normal, so it measures as free. It stayed invisible in testing too, because
            # while the rest of the model is intact you see the surface behind it and the pinhole
            # reads as a shading blemish; `hidden` then takes that surface away as unreachable and
            # the hole turns black. Measured on dinoki: repair cut 3 visible slivers at the eye rim
            # and the pair left 622 black pixels over 64 views of the head, which is what the user
            # sees as two torn triangles in the eyelid.
            seen = int(visible[kept[dropped]].sum())
            if seen:
                detected[f"{'selfIntersecting' if op == 'self_intersection' else op}VisibleKept"] = seen
            dropped = dropped[~visible[kept[dropped]]]

        removed["selfIntersection" if op == "self_intersection" else op] = int(len(dropped))
        if len(dropped) == len(kept):
            raise PipelineAbort(
                f"Face reduction ({engine}): the '{op}' operation would remove all "
                f"{len(kept)} remaining faces; turn it off or check the input model"
            )
        kept = np.delete(kept, dropped)
        op_deviation[op] = deviation.of(*_mesh_arrays(apply_mask(kept)))

    removal_deviation = (
        op_deviation[list(op_deviation)[-1]] if op_deviation else _NO_DEVIATION
    )
    reduced = apply_mask(kept)
    record["removalDeviation"] = removal_deviation
    if not _within_budget(removal_deviation, quality_budget_percent, normal_budget_degrees):
        # What each operation added on top of the ones before it, so the report names the one to
        # turn down rather than the one that happened to run last
        running, increments = 0.0, {}
        for name, measure in op_deviation.items():
            increments[name] = round(measure["maxPercent"] - running, 4)
            running = measure["maxPercent"]
        costs = ", ".join(f"{name} +{value:.3f}%" for name, value in increments.items())
        worst = max(increments, key=increments.get)
        raise PipelineAbort(
            f"Face reduction ({engine}): removing faces moved the surface by "
            f"{removal_deviation['maxPercent']:.3f}% of the bounding box diagonal, over the "
            f"{quality_budget_percent:g}% quality budget. Each operation added: {costs} "
            f"(removed {json.dumps(removed)}). Turn off '{worst}' or raise the budget"
        )

    # 5. Merge: the most aggressive collapse that still fits in the budget
    record["uvInvalidated"] = False
    final_deviation = removal_deviation
    if "merge" in ops:
        best = _merge_search(
            engine, np.asarray(reduced.vertices, dtype=np.float64), np.asarray(reduced.faces, dtype=np.int64),
            deviation, quality_budget_percent, normal_budget_degrees, record
        )
        if best is not None:
            merged_vertices, merged_faces, final_deviation = best
            # Collapsing edges rewrites the topology: no UV of the input survives it. Hand the
            # caller the mesh as it was here - removals done, source UVs intact - so it can still
            # measure the source texture against the surface that survives, not the one that went.
            if pre_collapse is not None:
                pre_collapse["mesh"] = reduced
            reduced = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False)
            record["uvInvalidated"] = True
            record["mergedFaces"] = int(len(merged_faces))

    n_output = len(reduced.faces)
    record.update({
        "facesAfter": int(n_output),
        "facesRemoved": int(n_input - n_output),
        "faceReductionPercent": round((1.0 - n_output / n_input) * 100.0, 2),
        "deviation": final_deviation,
        "qualityKeptPercent": round(100.0 - final_deviation["percentilePercent"], 3),
        "withinNormalBudget": final_deviation.get("normalDeviationDegrees", 0.0) <= normal_budget_degrees
    })
    return reduced
