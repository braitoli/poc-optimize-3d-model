"""
uvatlas_wrapper.py

Python Wrapper and Bridge for Microsoft UVAtlas (Iso-charts parameterization).
Provides an interface 100% compatible with xatlas.Atlas:
  (vmapping, new_faces, new_uvs, metadata) = uvatlas_unwrap(vertices, faces, ...)

Under the hood:
  Uses the native Open3D integration of Microsoft UVAtlas (Zhou et al., 2004),
  which provides pre-compiled ARM64 (Apple Silicon) & x86_64 binary wheels.
  Includes non-manifold diagnosis and optional seamless fallback to xatlas.
"""

from typing import Tuple, Dict, Any, Optional
import time
import logging
import numpy as np

logger = logging.getLogger("uvatlas_wrapper")


def is_mesh_manifold(vertices: np.ndarray, faces: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """
    Checks if a triangle mesh satisfies strict 2-manifold requirements for UVAtlas:
      1. Every edge has <= 2 incident faces.
      2. Every vertex has a single connected fan of incident faces (no bowties).
    """
    try:
        import open3d as o3d
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
        legacy.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))

        nme = legacy.get_non_manifold_edges(allow_boundary_edges=True)
        nmv = legacy.get_non_manifold_vertices()
        edge_manifold = len(nme) == 0
        vert_manifold = len(nmv) == 0
        is_manifold = edge_manifold and vert_manifold

        return is_manifold, {
            "is_manifold": is_manifold,
            "non_manifold_edges_count": len(nme),
            "non_manifold_vertices_count": len(nmv),
            "is_edge_manifold": edge_manifold,
            "is_vertex_manifold": vert_manifold,
        }
    except Exception as e:
        logger.warning(f"Failed to check manifold status: {e}")
        return False, {"error": str(e), "is_manifold": False}


def uvatlas_unwrap(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_res: int = 1024,
    gutter: float = 4.0,
    max_stretch: float = 0.16667,
    parallel_partitions: int = 1,
    fallback_to_xatlas: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unwraps mesh geometry using Microsoft UVAtlas (Iso-charts algorithm).
    
    Args:
        vertices: (N, 3) float array of 3D vertex positions.
        faces: (M, 3) int array of triangle vertex indices.
        target_res: Target texture canvas size (width = height = target_res).
        gutter: Spacing between UV islands in pixels (minimum 4.0 px enforced).
        max_stretch: Maximum allowed surface stretch parameter [0.0, 1.0].
        parallel_partitions: Partitions for multithreaded processing (>1 enables parallelization).
        fallback_to_xatlas: If True, falls back to xatlas if UVAtlas fails (e.g. non-manifold mesh).

    Returns:
        vmapping: (K,) int array mapping each output vertex back to its original vertex index in `vertices`.
        new_faces: (M, 3) int array of new face indices into `vmapping` and `new_uvs`.
        new_uvs: (K, 2) float array of normalized UV coordinates in [0, 1].
        metadata: Dict containing execution time, chart count, stretch, and engine info.
    """
    eff_gutter = max(4.0, float(gutter))
    v_arr = np.ascontiguousarray(vertices, dtype=np.float32)
    f_arr = np.ascontiguousarray(faces, dtype=np.int64)
    n_verts_orig = len(v_arr)
    n_faces = len(f_arr)

    t0 = time.perf_counter()

    try:
        import open3d as o3d

        # Pre-check manifold status to prevent Open3D C++ segmentation fault
        is_m, diag = is_mesh_manifold(vertices, faces)
        if not is_m:
            raise ValueError(
                f"Mesh is not 2-manifold (non_manifold_edges={diag.get('non_manifold_edges_count', 0)}, "
                f"non_manifold_verts={diag.get('non_manifold_vertices_count', 0)})"
            )

        o3d_mesh = o3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = o3d.core.Tensor(v_arr)
        o3d_mesh.triangle.indices = o3d.core.Tensor(f_arr)

        # Call UVAtlas Iso-charts parameterization
        max_stretch_out, num_charts, num_partitions = o3d_mesh.compute_uvatlas(
            size=int(target_res),
            gutter=float(eff_gutter),
            max_stretch=float(max_stretch),
            parallel_partitions=int(parallel_partitions)
        )

        elapsed = time.perf_counter() - t0
        texture_uvs = o3d_mesh.triangle["texture_uvs"].numpy()  # (M, 3, 2)

        # Convert per-corner UVs (M, 3, 2) into welded indexed mesh format (vmapping, new_faces, new_uvs)
        # matching the standard xatlas interface:
        corner_vids = f_arr.reshape(-1)          # (M*3,)
        corner_uvs = texture_uvs.reshape(-1, 2)   # (M*3, 2)

        # Quantize UVs slightly to 1e-6 to avoid floating-point duplicate vertices at chart boundaries
        corner_uvs_round = np.round(corner_uvs, decimals=6)

        # Structured composite key for fast unique indexing
        dtype = [('v', np.int64), ('u', np.float32), ('v_', np.float32)]
        keys = np.empty(len(corner_vids), dtype=dtype)
        keys['v'] = corner_vids
        keys['u'] = corner_uvs_round[:, 0]
        keys['v_'] = corner_uvs_round[:, 1]

        _, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)

        vmapping = corner_vids[first_idx].astype(np.int64)
        new_uvs = corner_uvs[first_idx].astype(np.float64)
        new_faces = inverse.reshape(-1, 3).astype(np.int64)

        metadata = {
            "engine": "uvatlas",
            "requested_engine": "uvatlas",
            "fallback_used": False,
            "elapsed_seconds": round(elapsed, 4),
            "num_charts": int(num_charts),
            "max_stretch": float(max_stretch_out),
            "num_partitions": int(num_partitions),
            "target_resolution": int(target_res),
            "original_vertices": n_verts_orig,
            "recharted_vertices": len(vmapping),
            "faces_count": n_faces
        }

        logger.info(
            f"[UVAtlas] Successfully unwrap in {elapsed:.3f}s: {num_charts} charts, "
            f"stretch={max_stretch_out:.4f}, vertices {n_verts_orig} -> {len(vmapping)}"
        )
        return vmapping, new_faces, new_uvs, metadata

    except Exception as exc:
        logger.warning(f"[UVAtlas] Execution failed: {exc}")
        if not fallback_to_xatlas:
            raise RuntimeError(f"UVAtlas failed: {exc}") from exc

        # Fallback to xatlas
        logger.info("[UVAtlas] Falling back to xatlas unwrap backend...")
        import xatlas

        t_fb0 = time.perf_counter()
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_arr, np.ascontiguousarray(f_arr, dtype=np.uint32))
        atlas.generate()
        t_fb1 = time.perf_counter()

        vmapping_fb, indices_fb, uvs_fb = atlas[0]
        vmapping = np.asarray(vmapping_fb, dtype=np.int64)
        new_faces = np.asarray(indices_fb, dtype=np.int64)
        new_uvs = np.asarray(uvs_fb, dtype=np.float64)

        metadata = {
            "engine": "xatlas",
            "requested_engine": "uvatlas",
            "fallback_used": True,
            "fallback_reason": str(exc),
            "elapsed_seconds": round(t_fb1 - t_fb0, 4),
            "num_charts": int(atlas.chart_count),
            "atlas_count": int(atlas.atlas_count),
            "utilization_percent": round(float(atlas.utilization * 100.0), 2),
            "target_resolution": int(target_res),
            "original_vertices": n_verts_orig,
            "recharted_vertices": len(vmapping),
            "faces_count": n_faces
        }

        logger.info(
            f"[UVAtlas->xatlas fallback] Unwrap in {metadata['elapsed_seconds']:.3f}s: "
            f"{atlas.chart_count} charts, util={metadata['utilization_percent']}%"
        )
        return vmapping, new_faces, new_uvs, metadata
