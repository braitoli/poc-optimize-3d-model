"""
uvatlas.py

Microsoft UVAtlas Iso-chart Parameterization & Unwrapping Integration.
Integrates Microsoft UVAtlas via Open3D C++ tensor pipeline and optional native CLI tool.
Enforces Zero-Decimation Policy by automatically repairing non-manifold vertices & edges
without removing any geometric triangles.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any, List
from collections import defaultdict
import numpy as np
import trimesh

# Try importing Open3D
_OPEN3D_AVAILABLE = False
_OPEN3D_UVATLAS_AVAILABLE = False
try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
    if hasattr(o3d, "t") and hasattr(o3d.t, "geometry") and hasattr(o3d.t.geometry.TriangleMesh, "compute_uvatlas"):
        _OPEN3D_UVATLAS_AVAILABLE = True
except ImportError:
    o3d = None


def is_open3d_uvatlas_available() -> bool:
    """Checks if Open3D with Microsoft UVAtlas C++ support is available."""
    return _OPEN3D_UVATLAS_AVAILABLE


def get_uvatlas_cli_path() -> Optional[str]:
    """
    Checks if a native UVAtlas CLI binary (uvatlas, uvatlastool, UVAtlasTool)
    is available in PATH or specified via UVATLAS_BIN / UVATLAS_CLI environment variables.
    """
    env_path = os.environ.get("UVATLAS_BIN") or os.environ.get("UVATLAS_CLI")
    if env_path and shutil.which(env_path):
        return env_path
    for name in ("uvatlas", "uvatlastool", "UVAtlas", "UVAtlasTool"):
        p = shutil.which(name)
        if p:
            return p
    return None


def is_uvatlas_available() -> Tuple[bool, str]:
    """
    Checks if Microsoft UVAtlas unwrap can be performed.
    Returns:
        (is_available, backend_name)
        backend_name is one of 'open3d', 'cli:<path>', or 'none'.
    """
    if is_open3d_uvatlas_available():
        return True, "open3d"
    cli = get_uvatlas_cli_path()
    if cli:
        return True, f"cli:{cli}"
    return False, "none"


def ensure_manifold_zero_decimation(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_iters: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Guarantees 2-manifold surface geometry required by Microsoft UVAtlas algorithm
    while strictly adhering to Rule 11 (Zero-Decimation Policy):
      - 100% faces are preserved (0 faces dropped).
      - Non-manifold edges (>2 adjacent faces) are split by duplicating edge vertices.
      - Non-manifold pinch vertices (multiple disjoint face fans) are split into separate sheets.

    Returns:
        (manifold_vertices, manifold_faces, vmapping_to_orig)
        where vmapping_to_orig maps each new vertex index back to the original vertex index.
    """
    verts = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64).copy()
    vmapping = np.arange(len(verts), dtype=np.int64)

    if not _OPEN3D_AVAILABLE:
        return verts, faces, vmapping

    leg = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces)
    )
    if leg.is_edge_manifold() and leg.is_vertex_manifold():
        return verts, faces, vmapping

    # 1. Split non-manifold edges (>2 incident faces)
    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
            edge_to_faces[e].append((fi, i))

    for e, flist in edge_to_faces.items():
        if len(flist) > 2:
            for fi, pos in flist[2:]:
                v1 = faces[fi, pos]
                v2 = faces[fi, (pos + 1) % 3]

                new_v1 = len(verts)
                verts = np.vstack([verts, [verts[v1]]])
                vmapping = np.append(vmapping, vmapping[v1])
                faces[fi, pos] = new_v1

                new_v2 = len(verts)
                verts = np.vstack([verts, [verts[v2]]])
                vmapping = np.append(vmapping, vmapping[v2])
                faces[fi, (pos + 1) % 3] = new_v2

    # 2. Split non-manifold vertices (multiple disjoint face fans at a single vertex)
    vert_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            vert_to_faces[int(v)].append(fi)

    for v, inc_faces in vert_to_faces.items():
        if len(inc_faces) <= 1:
            continue
        adj = defaultdict(set)
        for i in range(len(inc_faces)):
            f1 = faces[inc_faces[i]]
            for j in range(i + 1, len(inc_faces)):
                f2 = faces[inc_faces[j]]
                shared = set(f1) & set(f2)
                if len(shared) == 2 and v in shared:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        components: List[List[int]] = []
        for i in range(len(inc_faces)):
            if i not in visited:
                comp = []
                q = [i]
                visited.add(i)
                while q:
                    curr = q.pop()
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
                components.append(comp)

        if len(components) > 1:
            for comp in components[1:]:
                new_v = len(verts)
                verts = np.vstack([verts, [verts[v]]])
                vmapping = np.append(vmapping, vmapping[v])
                for idx in comp:
                    fi = inc_faces[idx]
                    pos = np.where(faces[fi] == v)[0][0]
                    faces[fi, pos] = new_v

    return verts, faces, vmapping


def unwrap_mesh_uvatlas_open3d(
    mesh: trimesh.Trimesh,
    target_res: int = 1024,
    gutter: float = 2.0,
    max_stretch: float = 0.1667,
    parallel_partitions: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes Microsoft UVAtlas unwrap using Open3D's C++ tensor pipeline.
    Preserves 100% faces and ensures all UV coordinates are strictly inside [0, 1].

    Returns:
        vertices_unwrapped: (N, 3) 3D vertex positions
        faces_unwrapped: (F, 3) triangle face indices
        uv_unwrapped: (N, 2) normalized UV coordinates in [0, 1]
        vmapping: (N,) mapping to original mesh.vertices indices
        stats: dict containing stretch, chart_count, and execution metadata
    """
    if not _OPEN3D_UVATLAS_AVAILABLE:
        raise RuntimeError(
            "Open3D UVAtlas is not available. Please install Open3D: pip install open3d"
        )

    t_start = time.perf_counter()
    n_faces = len(mesh.faces)

    # Adaptive parallel_partitions based on face count
    if parallel_partitions is None:
        if n_faces > 100_000:
            parallel_partitions = 4
        elif n_faces > 40_000:
            parallel_partitions = 2
        else:
            parallel_partitions = 1

    # 1. Zero-decimation manifold repair
    t_repair_start = time.perf_counter()
    verts_man, faces_man, vmap_orig = ensure_manifold_zero_decimation(
        mesh.vertices, mesh.faces
    )
    t_repair = time.perf_counter() - t_repair_start

    # 2. Convert to Open3D Tensor TriangleMesh
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(verts_man.astype(np.float32)),
        o3d.core.Tensor(faces_man.astype(np.int64))
    )

    # 3. Compute Microsoft UVAtlas
    t_uv_start = time.perf_counter()
    try:
        res = tmesh.compute_uvatlas(
            size=int(target_res),
            gutter=float(gutter),
            max_stretch=float(max_stretch),
            parallel_partitions=int(parallel_partitions)
        )
    except Exception as e:
        # Fallback: if compute_uvatlas failed due to residual non-manifoldness,
        # retry with parallel_partitions=1 and slightly relaxed max_stretch
        try:
            res = tmesh.compute_uvatlas(
                size=int(target_res),
                gutter=float(gutter),
                max_stretch=0.33,
                parallel_partitions=1
            )
        except Exception as retry_err:
            raise RuntimeError(
                f"Microsoft UVAtlas unwrapping failed: {retry_err} (initial error: {e})"
            ) from retry_err

    t_uv = time.perf_counter() - t_uv_start
    actual_stretch, chart_count, partition_count = res

    # 4. Extract per-face UV coordinates: (F, 3, 2)
    tri_uvs = tmesh.triangle["texture_uvs"].numpy()
    F = len(faces_man)

    # 5. Fast vertex welding along seams
    flat_v_man = faces_man.flatten()
    flat_uvs = np.clip(tri_uvs.reshape(-1, 2), 0.0, 1.0)
    flat_pos = verts_man[flat_v_man]
    flat_orig = vmap_orig[flat_v_man]

    # Structured array key: (vertex_id, round(u*1e6), round(v*1e6))
    dt = np.dtype([("vid", np.int64), ("u", np.int64), ("v", np.int64)])
    keys = np.empty(len(flat_v_man), dtype=dt)
    keys["vid"] = flat_v_man
    keys["u"] = np.round(flat_uvs[:, 0] * 1_000_000).astype(np.int64)
    keys["v"] = np.round(flat_uvs[:, 1] * 1_000_000).astype(np.int64)

    _, unique_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)

    faces_unwrapped = inverse.reshape(F, 3).astype(np.int64)
    vertices_unwrapped = flat_pos[unique_idx]
    uv_unwrapped = flat_uvs[unique_idx]
    vmapping = flat_orig[unique_idx]

    t_total = time.perf_counter() - t_start

    stats = {
        "uvatlas_backend": "open3d",
        "uvatlas_stretch": round(float(actual_stretch), 4),
        "uvatlas_chart_count": int(chart_count),
        "uvatlas_partition_count": int(partition_count),
        "uvatlas_repair_sec": round(t_repair, 3),
        "uvatlas_compute_sec": round(t_uv, 3),
        "uvatlas_total_sec": round(t_total, 3),
        "zero_decimation_faces_preserved": len(faces_unwrapped) == len(mesh.faces)
    }

    return vertices_unwrapped, faces_unwrapped, uv_unwrapped, vmapping, stats


def unwrap_mesh_uvatlas_cli(
    mesh: trimesh.Trimesh,
    cli_path: str,
    target_res: int = 1024,
    gutter: float = 2.0,
    max_stretch: float = 0.1667
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes Microsoft UVAtlas unwrap using native CLI binary (uvatlas / uvatlastool).
    """
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = os.path.join(tmp_dir, "input.obj")
        tmp_out = os.path.join(tmp_dir, "output.obj")

        # Export clean OBJ
        mesh.export(tmp_in, file_type="obj")

        cmd = [
            cli_path,
            "-o", tmp_out,
            "-w", str(int(target_res)),
            "-h", str(int(target_res)),
            "-g", str(float(gutter)),
            "-st", str(float(max_stretch)),
            "-y",
            tmp_in
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0 or not os.path.exists(tmp_out):
            raise RuntimeError(
                f"UVAtlas CLI failed with exit code {p.returncode}: {p.stderr or p.stdout}"
            )

        out_mesh = trimesh.load(tmp_out, file_type="obj", force="mesh", process=False)
        uvs = getattr(out_mesh.visual, "uv", None)
        if uvs is None:
            raise RuntimeError("UVAtlas CLI produced an OBJ without UV coordinates.")

        vmapping = np.arange(len(out_mesh.vertices), dtype=np.int64)
        t_total = time.perf_counter() - t_start

        stats = {
            "uvatlas_backend": f"cli:{cli_path}",
            "uvatlas_total_sec": round(t_total, 3),
            "zero_decimation_faces_preserved": len(out_mesh.faces) == len(mesh.faces)
        }

        return np.asarray(out_mesh.vertices), np.asarray(out_mesh.faces), np.asarray(uvs), vmapping, stats


def unwrap_mesh_uvatlas(
    mesh: trimesh.Trimesh,
    target_res: int = 1024,
    gutter: float = 2.0,
    max_stretch: float = 0.1667,
    parallel_partitions: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified entry point for Microsoft UVAtlas unwrapping.
    Automatically prioritizes Open3D tensor pipeline, falling back to CLI tool.
    """
    avail, backend = is_uvatlas_available()
    if not avail:
        raise RuntimeError(
            "Microsoft UVAtlas is not available. Please install Open3D in your Python environment "
            "(pip install open3d) or place the uvatlas / uvatlastool binary in PATH."
        )

    if backend == "open3d":
        return unwrap_mesh_uvatlas_open3d(
            mesh=mesh,
            target_res=target_res,
            gutter=gutter,
            max_stretch=max_stretch,
            parallel_partitions=parallel_partitions
        )
    elif backend.startswith("cli:"):
        cli_path = backend.split(":", 1)[1]
        return unwrap_mesh_uvatlas_cli(
            mesh=mesh,
            cli_path=cli_path,
            target_res=target_res,
            gutter=gutter,
            max_stretch=max_stretch
        )
    else:
        raise RuntimeError(f"Unknown UVAtlas backend: {backend}")
