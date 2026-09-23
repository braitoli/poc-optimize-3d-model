"""
test_face_reduce.py

Step 3: face repair & reduction (optimizer/core/face_reduce.py), for both engines.
Covers what each operation is allowed to remove, the quality budget that bounds all of them, and
the UV projection the Step 4 bake needs once `merge` has rewritten the topology.
"""

import unittest
from pathlib import Path

import numpy as np
import trimesh

from optimizer.core.errors import PipelineAbort
from optimizer.core.face_reduce import (
    AUTO_NORMAL_BUDGET,
    AUTO_NORMAL_BUDGET_FACTOR,
    validate_normal_factor,
    AUTO_NORMAL_BUDGET_RANGE,
    CGAL_HELPER,
    HARD_EDGE_DEGREES,
    MAX_DEVIATION_FACTOR,
    align_faces_outward,
    auto_normal_budget,
    DEFAULT_OPS,
    ENGINES,
    OPS,
    SourceUVProjector,
    engine_available,
    reduce_faces,
    restore_hard_edges,
    validate_normal_budget,
    validate_ops
)
from tests.fixtures import create_mock_glb

REPO_ROOT = Path(__file__).resolve().parents[1]
CGAL_AVAILABLE = engine_available("cgal")[0]
# A sphere turns too fast for the default shading budget to let any collapse through, and these
# tests are about what the merge does, not about how much shading the budget allows
LOOSE_SHADING = 30.0


def textured_sphere(subdivisions: int = 3) -> trimesh.Trimesh:
    """An icosphere with spherical UVs and a glTF PBR material, as the pipeline hands it to Step 3."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    normalized = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    uv = np.column_stack([
        0.5 + np.arctan2(normalized[:, 0], normalized[:, 2]) / (2 * np.pi),
        0.5 - np.arcsin(normalized[:, 1]) / np.pi
    ])
    from PIL import Image
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (64, 64), (120, 160, 90)),
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


def sphere_with_junk() -> trimesh.Trimesh:
    """A textured sphere plus a small free-floating box inside it: the box is both an isolated
    component (12 faces) and invisible from outside."""
    sphere = textured_sphere()
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vertices = np.vstack([sphere.vertices, box.vertices])
    faces = np.vstack([sphere.faces, np.asarray(box.faces) + len(sphere.vertices)])
    uv = np.vstack([np.asarray(sphere.visual.uv), np.zeros((len(box.vertices), 2))])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=sphere.visual.material.copy())
    return mesh


class TestOptionValidation(unittest.TestCase):
    def test_validate_ops_returns_canonical_order(self):
        self.assertEqual(validate_ops(["merge", "repair"]), ("repair", "merge"))

    def test_unknown_op_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            validate_ops(["repair", "decimate"])
        self.assertIn("decimate", str(ctx.exception))

    def test_default_ops_leave_self_intersection_out(self):
        # Removing every face of an intersecting pair deletes large parts of an AI shell, so the
        # operation exists but is opt-in
        self.assertIn("self_intersection", OPS)
        self.assertNotIn("self_intersection", DEFAULT_OPS)

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            engine_available("open3d")
        with self.assertRaises(ValueError):
            reduce_faces(textured_sphere(), engine="open3d")

    def test_non_positive_quality_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            reduce_faces(textured_sphere(), ops=("repair",), quality_budget_percent=0)

    def test_meshlab_engine_is_available(self):
        available, reason = engine_available("meshlab")
        self.assertTrue(available, reason)

    def test_cgal_engine_reports_the_helper_it_needs(self):
        available, reason = engine_available("cgal")
        if not available:
            self.assertIn("mesh_repair", reason)
            self.assertIn("build.sh", reason)
        else:
            self.assertTrue(CGAL_HELPER.exists())


class TestRemovalOperations(unittest.TestCase):
    """The removal operations keep every surviving face's vertices and UVs untouched."""

    def _run(self, engine: str, mesh: trimesh.Trimesh, ops, **kwargs):
        stats = {}
        reduced = reduce_faces(mesh, engine=engine, ops=ops, stats=stats, **kwargs)
        return reduced, stats

    def test_clean_mesh_loses_nothing(self):
        mesh = textured_sphere()
        for engine in ENGINES:
            if engine == "cgal" and not CGAL_AVAILABLE:
                continue
            with self.subTest(engine=engine):
                reduced, stats = self._run(engine, mesh, ("repair", "isolated"))
                self.assertEqual(len(reduced.faces), len(mesh.faces))
                self.assertEqual(stats["facesRemoved"], 0)
                self.assertEqual(stats["deviation"]["maxPercent"], 0.0)

    def test_isolated_removes_the_small_component(self):
        mesh = sphere_with_junk()
        for engine in ENGINES:
            if engine == "cgal" and not CGAL_AVAILABLE:
                continue
            with self.subTest(engine=engine):
                reduced, stats = self._run(engine, mesh, ("isolated",), isolated_min_faces=25)
                self.assertEqual(stats["removed"]["isolated"], 12)
                self.assertEqual(len(reduced.faces), len(mesh.faces) - 12)

    def test_hidden_removes_what_no_view_can_see(self):
        mesh = sphere_with_junk()
        reduced, stats = self._run("meshlab", mesh, ("hidden",))
        # The box sits inside the sphere: invisible from every direction, and nothing else is
        self.assertEqual(stats["removed"]["hidden"], 12)
        self.assertEqual(len(reduced.faces), len(mesh.faces) - 12)
        # Removing geometry a camera cannot see does not move the visible surface at all
        self.assertEqual(stats["deviation"]["maxPercent"], 0.0)

    def test_surviving_faces_keep_their_uvs(self):
        mesh = sphere_with_junk()
        original_uv = {
            tuple(np.round(mesh.vertices[v], 6)): tuple(np.round(mesh.visual.uv[v], 6))
            for v in range(len(mesh.vertices))
        }
        reduced, stats = self._run("meshlab", mesh, ("isolated", "hidden"))
        self.assertFalse(stats["uvInvalidated"])
        self.assertEqual(len(reduced.visual.uv), len(reduced.vertices))
        for vertex, uv in zip(reduced.vertices, reduced.visual.uv):
            key = tuple(np.round(vertex, 6))
            self.assertIn(key, original_uv)
            self.assertEqual(tuple(np.round(uv, 6)), original_uv[key])

    def test_engines_agree_on_a_junk_component(self):
        if not CGAL_AVAILABLE:
            self.skipTest(f"the CGAL helper is not built at {CGAL_HELPER}")
        mesh = sphere_with_junk()
        counts = {}
        for engine in ENGINES:
            _, stats = self._run(engine, mesh, ("repair", "isolated"), isolated_min_faces=25)
            counts[engine] = stats["facesAfter"]
        self.assertEqual(counts["meshlab"], counts["cgal"])


class TestAutoShadingBudget(unittest.TestCase):
    """The default shading budget is read off the mesh: a fixed angle buys a coarse model a
    fraction of an edge's worth of collapse and a dense one several, so it is not a setting that
    can be carried from one model to the next."""

    @staticmethod
    def budget_of(mesh: trimesh.Trimesh):
        return auto_normal_budget(np.asarray(mesh.vertices), np.asarray(mesh.faces))

    def test_it_reads_the_angle_the_surface_already_turns_per_edge(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        budget, per_edge = self.budget_of(mesh)
        angles = np.degrees(mesh.face_adjacency_angles)
        self.assertAlmostEqual(per_edge, float(np.median(angles[angles <= HARD_EDGE_DEGREES])), places=6)
        self.assertAlmostEqual(budget, per_edge * AUTO_NORMAL_BUDGET_FACTOR, places=6)

    def test_a_finer_mesh_of_the_same_shape_gets_a_smaller_budget(self):
        # Subdividing halves how far the surface turns per edge, so the same shape may be turned
        # half as far - which is what makes the setting mean the same thing on both
        coarse, _ = self.budget_of(trimesh.creation.icosphere(subdivisions=2, radius=1.0))
        fine, _ = self.budget_of(trimesh.creation.icosphere(subdivisions=4, radius=1.0))
        self.assertLess(fine, coarse)

    def test_creases_are_not_read_as_tessellation(self):
        # A box is all creases: there is no smooth surface to measure, so it falls back to the floor
        budget, per_edge = self.budget_of(trimesh.creation.box())
        self.assertEqual(budget, AUTO_NORMAL_BUDGET_RANGE[0])
        self.assertEqual(per_edge, 0.0)

    def test_the_result_is_clamped(self):
        for mesh in (trimesh.creation.icosphere(subdivisions=1), trimesh.creation.icosphere(subdivisions=5)):
            with self.subTest(faces=len(mesh.faces)):
                budget, _ = self.budget_of(mesh)
                self.assertGreaterEqual(budget, AUTO_NORMAL_BUDGET_RANGE[0])
                self.assertLessEqual(budget, AUTO_NORMAL_BUDGET_RANGE[1])

    def test_an_angle_or_auto_is_accepted_and_nothing_else(self):
        self.assertEqual(validate_normal_budget(AUTO_NORMAL_BUDGET), AUTO_NORMAL_BUDGET)
        self.assertEqual(validate_normal_budget(8), 8)
        self.assertEqual(validate_normal_budget(90), 90)
        for bad in ("AUTO", "8", 0, -1, 90.5, True, None):
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    validate_normal_budget(bad)

    def test_a_run_records_the_angle_it_resolved_and_what_it_read(self):
        mesh = textured_sphere(subdivisions=3)
        expected, per_edge = self.budget_of(mesh)
        stats = {}
        reduce_faces(mesh, engine="cgal" if CGAL_AVAILABLE else "meshlab", ops=("repair",),
                     normal_budget_degrees=AUTO_NORMAL_BUDGET, stats=stats)
        self.assertTrue(stats["normalBudgetAuto"])
        self.assertAlmostEqual(stats["normalBudgetDegrees"], round(expected, 2), places=2)
        self.assertAlmostEqual(stats["perEdgeTurnDegrees"], round(per_edge, 2), places=2)

    def test_the_factor_scales_the_budget_and_is_recorded(self):
        mesh = textured_sphere(subdivisions=3)
        _, per_edge = self.budget_of(mesh)
        for factor in (1.0, 2.0):
            with self.subTest(factor=factor):
                budget, _ = auto_normal_budget(np.asarray(mesh.vertices), np.asarray(mesh.faces), factor)
                self.assertAlmostEqual(budget, min(per_edge * factor, AUTO_NORMAL_BUDGET_RANGE[1]), places=6)
                stats = {}
                reduce_faces(textured_sphere(subdivisions=3), engine="cgal" if CGAL_AVAILABLE else "meshlab",
                             ops=("repair",), normal_budget_degrees=AUTO_NORMAL_BUDGET,
                             normal_budget_factor=factor, stats=stats)
                self.assertEqual(stats["normalBudgetFactor"], factor)
                self.assertAlmostEqual(stats["normalBudgetDegrees"], round(budget, 2), places=2)

    def test_the_factor_must_be_a_positive_number(self):
        self.assertEqual(validate_normal_factor(7), 7.0)
        for bad in (0, -1, 31, True, "3", None):
            with self.subTest(factor=bad):
                with self.assertRaises(ValueError):
                    validate_normal_factor(bad)

    def test_a_pinned_angle_is_used_as_given_and_reads_nothing(self):
        stats = {}
        reduce_faces(textured_sphere(subdivisions=3), engine="cgal" if CGAL_AVAILABLE else "meshlab",
                     ops=("repair",), normal_budget_degrees=11.5, stats=stats)
        self.assertFalse(stats["normalBudgetAuto"])
        self.assertEqual(stats["normalBudgetDegrees"], 11.5)
        self.assertIsNone(stats["perEdgeTurnDegrees"])
        self.assertIsNone(stats["normalBudgetFactor"])


class TestQualityBudget(unittest.TestCase):
    def test_merge_stays_within_the_budget(self):
        mesh = textured_sphere(subdivisions=4)  # 5120 faces
        stats = {}
        reduced = reduce_faces(mesh, engine="meshlab", ops=("merge",), quality_budget_percent=2.0,
                     normal_budget_degrees=LOOSE_SHADING, stats=stats)
        self.assertLess(len(reduced.faces), len(mesh.faces))
        # The budget holds at the percentile, with the worst point anywhere capped well above it
        self.assertLessEqual(stats["deviation"]["percentilePercent"], 2.0)
        self.assertLessEqual(stats["deviation"]["maxPercent"], 2.0 * MAX_DEVIATION_FACTOR)
        self.assertTrue(stats["uvInvalidated"])
        # It reports every target it measured, so the chosen one can be audited
        self.assertTrue(stats["mergeAttempts"])
        for attempt in stats["mergeAttempts"]:
            self.assertIn("deviationMaxPercent", attempt)

    def test_a_tighter_budget_keeps_more_faces(self):
        mesh = textured_sphere(subdivisions=4)
        loose, tight = {}, {}
        reduce_faces(mesh, engine="meshlab", ops=("merge",), quality_budget_percent=2.0,
                     normal_budget_degrees=LOOSE_SHADING, stats=loose)
        reduce_faces(mesh, engine="meshlab", ops=("merge",), quality_budget_percent=0.05,
                     normal_budget_degrees=LOOSE_SHADING, stats=tight)
        self.assertGreater(tight["facesAfter"], loose["facesAfter"])
        self.assertLessEqual(tight["deviation"]["percentilePercent"], 0.05)

    def test_the_same_model_reduces_to_the_same_mesh_every_run(self):
        mesh = textured_sphere(subdivisions=4)
        runs = []
        for _ in range(2):
            stats = {}
            reduce_faces(mesh, engine="meshlab", ops=("merge",), quality_budget_percent=2.0,
                     normal_budget_degrees=LOOSE_SHADING, stats=stats)
            runs.append((stats["facesAfter"], stats["deviation"]["maxPercent"]))
        self.assertEqual(runs[0], runs[1])

    def test_a_visible_component_is_kept_when_the_budget_cannot_pay_for_it(self):
        # Two separate spheres, both plainly visible. Removing the smaller one as an "isolated"
        # component leaves nothing anywhere near where it stood, so the visible surface would move
        # far - and rather than failing the run over it, the step keeps it.
        big = textured_sphere(subdivisions=3)
        small = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
        small.apply_translation([4.0, 0.0, 0.0])
        vertices = np.vstack([big.vertices, small.vertices])
        faces = np.vstack([big.faces, np.asarray(small.faces) + len(big.vertices)])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.vstack([np.asarray(big.visual.uv), np.zeros((len(small.vertices), 2))]),
            material=big.visual.material.copy()
        )

        stats = {}
        reduced = reduce_faces(
            mesh,
            engine="meshlab",
            ops=("isolated",),
            isolated_min_faces=len(small.faces) + 1,
            quality_budget_percent=1.0,
            stats=stats
        )
        self.assertEqual(stats["removed"]["isolated"], 0)
        self.assertEqual(stats["detected"]["isolatedVisibleKept"], len(small.faces))
        self.assertEqual(len(reduced.faces), len(mesh.faces))
        self.assertLessEqual(stats["deviation"]["percentilePercent"], 1.0)

    def test_a_hidden_component_is_removed_whatever_the_budget(self):
        # The box inside the sphere costs nothing to remove: no view ever saw it
        stats = {}
        mesh = sphere_with_junk()
        reduce_faces(
            mesh,
            engine="meshlab",
            ops=("isolated",),
            isolated_min_faces=25,
            quality_budget_percent=1e-6,
            stats=stats
        )
        self.assertEqual(stats["removed"]["isolated"], 12)
        self.assertNotIn("isolatedVisibleKept", stats["detected"])
        self.assertEqual(stats["deviation"]["maxPercent"], 0.0)

    def test_self_intersection_never_cuts_a_face_you_can_see(self):
        # Two boxes crossing in mid-air: every face of the intersection is visible from outside
        a = trimesh.creation.box(extents=(2.0, 0.4, 0.4))
        b = trimesh.creation.box(extents=(0.4, 2.0, 0.4))
        vertices = np.vstack([a.vertices, b.vertices])
        faces = np.vstack([a.faces, np.asarray(b.faces) + len(a.vertices)])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(vertices), 2)), material=textured_sphere().visual.material.copy()
        )

        stats = {}
        reduced = reduce_faces(mesh, engine="meshlab", ops=("self_intersection",), stats=stats)
        self.assertEqual(stats["removed"]["selfIntersection"], 0)
        self.assertGreater(stats["detected"]["selfIntersectingVisibleKept"], 0)
        self.assertEqual(len(reduced.faces), len(mesh.faces))
        self.assertEqual(stats["deviation"]["maxPercent"], 0.0)

    def test_stats_report_what_was_removed_and_how_much_quality_is_left(self):
        stats = {}
        mesh = sphere_with_junk()
        reduce_faces(mesh, engine="meshlab", ops=DEFAULT_OPS, quality_budget_percent=2.0, stats=stats)
        self.assertEqual(stats["engine"], "meshlab")
        self.assertEqual(stats["ops"], list(DEFAULT_OPS))
        self.assertEqual(stats["facesBefore"], len(mesh.faces))
        self.assertEqual(set(stats["removed"]), {"repair", "selfIntersection", "isolated", "hidden"})
        self.assertEqual(stats["facesBefore"] - stats["facesAfter"], stats["facesRemoved"])
        # Quality kept is read at the same percentile as the budget, not at the max
        self.assertEqual(
            stats["qualityKeptPercent"], round(100.0 - stats["deviation"]["percentilePercent"], 3)
        )


class TestMergeKeepsTheSurfaceTogether(unittest.TestCase):
    """A glTF mesh splits its vertices at every UV seam. If the collapse runs on that split
    topology it treats each seam as a border it may pull free, shredding the model into loose
    patches - which is worse geometry and, because every patch becomes its own UV island needing
    its own gutter, a far more expensive texture atlas."""

    @staticmethod
    def topology(mesh: trimesh.Trimesh):
        """(connected components, boundary edges) of the mesh welded by position."""
        from scipy.sparse import csgraph, csr_matrix

        _, welded = np.unique(np.round(np.asarray(mesh.vertices), 6), axis=0, return_inverse=True)
        faces = welded[np.asarray(mesh.faces)]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        graph = csr_matrix(
            (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
            shape=(welded.max() + 1,) * 2
        )
        labels = csgraph.connected_components(graph, directed=False)[1]
        return len(set(labels[np.unique(faces)])), int((counts == 1).sum())

    @staticmethod
    def seam_split(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """The same surface with every triangle given its own three vertices: the extreme case of
        what UV seams do to a glTF mesh."""
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)[faces].reshape(-1, 3)
        split = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)
        split.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(mesh.visual.uv)[faces].reshape(-1, 2),
            material=mesh.visual.material.copy()
        )
        return split

    def test_a_seam_split_mesh_survives_the_collapse_in_one_piece(self):
        original = textured_sphere(subdivisions=4)
        split = self.seam_split(original)
        self.assertEqual(len(split.vertices), 3 * len(split.faces))  # nothing is shared

        merged = reduce_faces(split, engine="meshlab", ops=("merge",), quality_budget_percent=2.0,
                              normal_budget_degrees=LOOSE_SHADING)
        components, boundary_edges = self.topology(merged)
        # The sphere is one closed surface, and simplifying it must not change that
        self.assertEqual(components, 1)
        self.assertEqual(boundary_edges, 0)
        self.assertLess(len(merged.faces), len(split.faces))


class TestHardEdgesSurviveTheCollapse(unittest.TestCase):
    """A glTF model draws a crease by splitting the vertex and giving each side its own normal.
    An edge collapse returns bare positions and faces, so without putting those normals back the
    crease lights up as a smooth surface - the model looks lit from the wrong place even though
    its geometry, UVs and colours are all correct."""

    @staticmethod
    def flat_shaded_box() -> trimesh.Trimesh:
        """A box whose every face carries its own normals: six hard edges, as a real model has."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        faces = np.asarray(box.faces)
        vertices = np.asarray(box.vertices)[faces].reshape(-1, 3)
        split = trimesh.Trimesh(
            vertices=vertices,
            faces=np.arange(len(vertices)).reshape(-1, 3),
            process=False
        )
        split.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(vertices), 2)), material=textured_sphere().visual.material.copy()
        )
        return split

    @staticmethod
    def sharp_positions(mesh: trimesh.Trimesh, degrees: float = 20.0) -> int:
        """Positions where two normals of the mesh disagree by more than `degrees`."""
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        _, groups, counts = np.unique(np.round(vertices, 5), axis=0, return_inverse=True, return_counts=True)
        sharp = 0
        for group in np.nonzero(counts > 1)[0]:
            members = normals[np.nonzero(groups == group)[0]]
            spread = np.degrees(np.arccos(np.clip((members @ members.T).min(), -1.0, 1.0)))
            sharp += spread > degrees
        return int(sharp)

    def test_a_welded_mesh_gets_its_creases_back(self):
        original = self.flat_shaded_box()
        self.assertGreater(self.sharp_positions(original), 0)

        # What an edge collapse hands back: the same surface with one averaged normal per position
        welded = trimesh.Trimesh(vertices=original.vertices, faces=original.faces, process=True)
        self.assertEqual(self.sharp_positions(welded), 0)

        restored = restore_hard_edges(welded, SourceUVProjector(original))
        self.assertEqual(len(restored.faces), len(welded.faces))
        self.assertGreater(self.sharp_positions(restored), 0)
        # Each corner takes the normal of its own side, so a box corner reads as a right angle again
        corner_spread = []
        vertices = np.asarray(restored.vertices)
        normals = np.asarray(restored.vertex_normals)
        _, groups, counts = np.unique(np.round(vertices, 5), axis=0, return_inverse=True, return_counts=True)
        for group in np.nonzero(counts > 1)[0]:
            members = normals[np.nonzero(groups == group)[0]]
            corner_spread.append(np.degrees(np.arccos(np.clip((members @ members.T).min(), -1.0, 1.0))))
        self.assertGreater(max(corner_spread), 80.0)

    def test_a_smooth_surface_is_not_broken_into_facets(self):
        # The sphere has no crease anywhere, so nothing may be split and the shading stays smooth
        sphere = textured_sphere(subdivisions=3)
        welded = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces, process=True)
        restored = restore_hard_edges(welded, SourceUVProjector(sphere))
        self.assertEqual(self.sharp_positions(restored, degrees=20.0), 0)


class TestWindingMatchesTheNormals(unittest.TestCase):
    """The pipeline renders single-sided, so a face wound against the normals it is shaded by is
    culled: you look through the model and the gap reads as a black triangle on the surface."""

    @staticmethod
    def reversed_faces(mesh: trimesh.Trimesh) -> int:
        faces = np.asarray(mesh.faces)
        triangles = np.asarray(mesh.vertices)[faces]
        geometric = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        shading = np.asarray(mesh.vertex_normals)[faces].sum(axis=1)
        return int(((geometric * shading).sum(axis=1) < 0).sum())

    def test_a_face_wound_inwards_is_turned_around(self):
        # Winding reversed, normals still pointing out: the outside is on the normals' side, so it
        # is the triangle that gets turned around
        mesh = textured_sphere(subdivisions=3)
        faces = np.asarray(mesh.faces).copy()
        faces[:5] = faces[:5][:, ::-1]
        broken = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=faces,
            vertex_normals=np.asarray(mesh.vertex_normals), process=False
        )
        self.assertEqual(self.reversed_faces(broken), 5)

        fixed = align_faces_outward(broken)
        self.assertEqual(fixed["windingFlipped"], 5)
        self.assertEqual(fixed["normalsFlipped"], 0)
        self.assertEqual(self.reversed_faces(broken), 0)
        # The normals carry the model's hard edges, so they are left exactly as they were
        self.assertTrue(np.allclose(np.asarray(broken.vertex_normals), np.asarray(mesh.vertex_normals)))

    def test_a_normal_pointing_inwards_is_turned_around_instead(self):
        # Winding already faces out and the normals are inverted: turning the triangle around would
        # cull it and leave a hole, so the normals are what must be corrected
        mesh = textured_sphere(subdivisions=3)
        normals = np.asarray(mesh.vertex_normals).copy()
        faces = np.asarray(mesh.faces)
        inverted = np.unique(faces[:3])
        normals[inverted] *= -1.0
        broken = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=faces.copy(),
            vertex_normals=normals, process=False
        )
        before = np.asarray(broken.faces).copy()
        self.assertGreater(self.reversed_faces(broken), 0)

        fixed = align_faces_outward(broken)
        self.assertEqual(fixed["windingFlipped"], 0)
        self.assertGreater(fixed["normalsFlipped"], 0)
        self.assertEqual(self.reversed_faces(broken), 0)
        # No triangle was turned around, so nothing can have been culled away
        self.assertEqual(len(broken.faces), len(before))
        self.assertTrue(np.array_equal(
            np.sort(np.asarray(broken.vertices)[np.asarray(broken.faces)].reshape(-1, 3), axis=0),
            np.sort(np.asarray(mesh.vertices)[before].reshape(-1, 3), axis=0)
        ))

    def test_a_consistent_mesh_is_left_alone(self):
        mesh = textured_sphere(subdivisions=3)
        before = np.asarray(mesh.faces).copy()
        self.assertEqual(align_faces_outward(mesh), {"windingFlipped": 0, "normalsFlipped": 0})
        self.assertTrue(np.array_equal(np.asarray(mesh.faces), before))


class TestSourceUVProjector(unittest.TestCase):
    """After `merge` the mesh has no UV of its own; Step 4 reads the original atlas through this."""

    def test_projection_reproduces_the_source_uv(self):
        mesh = textured_sphere()
        projector = SourceUVProjector(mesh)
        # Triangle centroids are unambiguous (a vertex on a UV seam has several valid UVs)
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        expected = np.asarray(mesh.visual.uv)[mesh.faces].mean(axis=1)
        error = np.linalg.norm(projector(centroids) - expected, axis=1)
        self.assertLess(error.max(), 1e-4)

    def test_projection_covers_a_merged_mesh(self):
        mesh = textured_sphere(subdivisions=4)
        projector = SourceUVProjector(mesh)
        merged = reduce_faces(mesh, engine="meshlab", ops=("merge",), quality_budget_percent=2.0,
                              normal_budget_degrees=LOOSE_SHADING)
        uv = projector.vertex_uv(merged)
        self.assertEqual(uv.shape, (len(merged.vertices), 2))
        self.assertTrue(np.isfinite(uv).all())

    def test_a_mesh_without_uvs_cannot_be_projected_from(self):
        with self.assertRaises(PipelineAbort):
            SourceUVProjector(trimesh.creation.icosphere(subdivisions=2))


class TestReducedModelStaysTextured(unittest.TestCase):
    """The colours a merged model shows must still be the colours of the original texture."""

    def test_baked_colours_match_the_original_surface(self):
        import tempfile
        import open3d as o3d
        from optimizer.step_pipeline import StepPipeline

        with tempfile.TemporaryDirectory(prefix="face_reduce_bake_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=4)
            StepPipeline(
                texture_format="original",
                verbose=False,
                stream_events=False,
                skip_steps=[5, 6],
                reduce_normal_budget=LOOSE_SHADING
            ).run(source, tmp_dir / "out")
            baked = trimesh.load(tmp_dir / "out" / "step_04_texture_baked.glb", force="mesh", process=False)
            # Step 1 grounds the model at Y=0, so the reference is its output, not the raw input:
            # both meshes have to stand in the same place for a closest-point comparison
            original = trimesh.load(tmp_dir / "out" / "step_01_cleaned_grounded.glb", force="mesh", process=False)

        def colour_at(mesh, points):
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
            ))
            hit = scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32)))
            triangles = hit["primitive_ids"].numpy().astype(np.int64)
            bary = hit["primitive_uvs"].numpy().astype(np.float64)
            weights = np.column_stack([1.0 - bary.sum(axis=1), bary])
            uv = (np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)[triangles]] * weights[:, :, None]).sum(axis=1) % 1.0
            image = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64)
            px = np.clip((uv[:, 0] * (image.shape[1] - 1)).astype(int), 0, image.shape[1] - 1)
            py = np.clip(((1.0 - uv[:, 1]) * (image.shape[0] - 1)).astype(int), 0, image.shape[0] - 1)
            return image[py, px]

        self.assertLess(len(baked.faces), len(original.faces))
        points = baked.sample(20000)
        difference = np.abs(colour_at(baked, points) - colour_at(original, points)).mean(axis=1)
        # The flat-coloured regions of the mock texture must survive the re-chart intact; only the
        # few texels straddling one of its painted borders may disagree
        self.assertLess(float(np.median(difference)), 4.0)
        self.assertLess(float((difference > 48).mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
