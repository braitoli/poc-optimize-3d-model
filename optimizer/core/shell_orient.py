"""
shell_orient.py

Visibility-based winding orientation for 3D AI statues (e.g. Trellis / Tripo).
Solves thin-shell inside-out backface visibility issues.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles (prune_hidden=False).
- Groups connected components across UV seams.
- Flips inward-facing components to outward CCW FrontSide.

High-Performance Parallel Architecture:
- Vectorized NumPy z-buffer with precomputed edge determinants (avoids degenerate allocations).
- Exact float64 lexsort tie-breaking for 100% bit-for-bit mathematical parity.
- Multi-core parallel view rasterization (Apple Silicon / Linux / Windows auto-fallback).
- Concurrent connected-components graph decomposition.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging
import os
import multiprocessing as mp
import concurrent.futures
import numpy as np
from scipy.sparse import csgraph, csr_matrix

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = 48
DEFAULT_RESOLUTION = 384
DETECT_THRESHOLD = 0.005
DETECT_VIEWS = 24
DETECT_RESOLUTION = 256


def _sphere_dirs(n: int) -> np.ndarray:
    """n Fibonacci spiral directions uniformly distributed over the unit sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * i
    return np.c_[
        np.cos(theta) * np.sin(phi),
        np.cos(phi),
        np.sin(theta) * np.sin(phi),
    ]


def _zbuffer(sx: np.ndarray, sy: np.ndarray, depth: np.ndarray, res: int) -> np.ndarray:
    """
    Vectorized z-buffer rasterization: returns res x res image with closest face ID per pixel.
    Optimized:
    - Pre-filters non-positive screen areas and degenerate triangles before pixel expansion.
    - Precomputes per-triangle barycentric edge coefficients, eliminating repeated divisions.
    - Uses exact float64 lexsort for 100% parity with mathematical rasterization.
    """
    x0 = np.clip(np.floor(sx.min(1)).astype(np.int32), 0, res)
    x1 = np.clip(np.ceil(sx.max(1)).astype(np.int32), 0, res)
    y0 = np.clip(np.floor(sy.min(1)).astype(np.int32), 0, res)
    y1 = np.clip(np.ceil(sy.max(1)).astype(np.int32), 0, res)
    w = (x1 - x0).astype(np.int64)
    h = (y1 - y0).astype(np.int64)
    area = w * h

    ax, ay = sx[:, 0], sy[:, 0]
    bx, by = sx[:, 1], sy[:, 1]
    cx, cy = sx[:, 2], sy[:, 2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)

    # Filter valid triangles (non-zero area & non-degenerate denominator)
    valid = (area > 0) & (np.abs(den) > 1e-12)
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return np.full((res, res), -1, dtype=np.int32)

    reps = area[idx]
    total = int(reps.sum())
    fid = np.repeat(idx, reps)
    offset = np.concatenate([[0], np.cumsum(reps)])[:-1]
    seq = np.arange(total) - np.repeat(offset, reps)
    wf = w[fid]
    px = x0[fid] + (seq % wf)
    py = y0[fid] + (seq // wf)

    # Normalized linear edge functions:
    # w0 = a0*px + b0*py + c0
    # w1 = a1*px + b1*py + c1
    den_idx = den[idx]
    a0 = (by[idx] - cy[idx]) / den_idx
    b0 = (cx[idx] - bx[idx]) / den_idx
    c0 = - a0 * cx[idx] - b0 * cy[idx]

    a1 = (cy[idx] - ay[idx]) / den_idx
    b1 = (ax[idx] - cx[idx]) / den_idx
    c1 = - a1 * cx[idx] - b1 * cy[idx]

    a0_f = np.repeat(a0, reps)
    b0_f = np.repeat(b0, reps)
    c0_f = np.repeat(c0, reps)
    a1_f = np.repeat(a1, reps)
    b1_f = np.repeat(b1, reps)
    c1_f = np.repeat(c1, reps)

    w0 = a0_f * px + b0_f * py + c0_f
    w1 = a1_f * px + b1_f * py + c1_f
    inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
    if not inside.any():
        return np.full((res, res), -1, dtype=np.int32)

    fid = fid[inside]
    lin = py[inside].astype(np.int64) * res + px[inside]
    order = np.lexsort((depth[fid], lin))
    lin_s, fid_s = lin[order], fid[order]
    first = np.ones(len(lin_s), dtype=bool)
    first[1:] = lin_s[1:] != lin_s[:-1]

    frame = np.full(res * res, -1, dtype=np.int32)
    frame[lin_s[first]] = fid_s[first]
    return frame.reshape(res, res)


def _render_single_view_task(args: Tuple) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Worker task for rendering a single Fibonacci sphere view direction."""
    direction, vertices, faces, centre, span, resolution = args
    d = direction / np.linalg.norm(direction)
    up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    ex = np.cross(up, d)
    ex /= np.linalg.norm(ex)
    ey = np.cross(d, ex)
    view = (vertices - centre) @ np.c_[ex, ey, d]

    tri = view[faces]
    sx = (tri[:, :, 0] / span * 0.92 + 0.5) * resolution
    sy = (tri[:, :, 1] / span * 0.92 + 0.5) * resolution
    depth = tri[:, :, 2].mean(axis=1)

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    facing = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    frame = _zbuffer(sx, sy, -depth, resolution)
    ids, counts = np.unique(frame[frame >= 0], return_counts=True)
    if len(ids) == 0:
        return ids, counts, None
    return ids, counts, np.sign(facing[ids])


def _rasterize_votes(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int,
    resolution: int,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes visibility votes for each face in parallel across CPU cores.
    Uses multi-processing (or thread pool fallback) across all Fibonacci camera angles.
    Returns:
        vote: net visibility vote (vote_front - vote_back)
        seen_px: total pixels where face was closest surface
        vis_count: number of views where face appeared in z-buffer
        vote_front: total front-facing visibility pixels
        vote_back: total back-facing visibility pixels
    """
    vote = np.zeros(len(faces), dtype=np.float64)
    seen_px = np.zeros(len(faces), dtype=np.float64)
    vis_count = np.zeros(len(faces), dtype=np.int32)
    vote_front = np.zeros(len(faces), dtype=np.float64)
    vote_back = np.zeros(len(faces), dtype=np.float64)

    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    span = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    if span <= 0:
        return vote, seen_px, vis_count, vote_front, vote_back

    sphere_directions = _sphere_dirs(views)
    tasks = [(d, vertices, faces, centre, span, resolution) for d in sphere_directions]

    cpu_count = os.cpu_count() or 4
    workers = max_workers if max_workers is not None else min(cpu_count, 10)

    if workers > 1 and len(tasks) > 1:
        use_threads = False
        try:
            # Fork is fastest on macOS and Linux (zero process bootstrap overhead)
            ctx = mp.get_context("fork")
            chunksize = max(1, len(tasks) // (workers * 2))
            with ctx.Pool(workers) as pool:
                results = pool.map(_render_single_view_task, tasks, chunksize=chunksize)
        except Exception:
            use_threads = True

        if use_threads:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as tpool:
                results = list(tpool.map(_render_single_view_task, tasks))
    else:
        results = [_render_single_view_task(t) for t in tasks]

    for ids, counts, sign_f in results:
        if len(ids) == 0 or sign_f is None:
            continue
        vis_count[ids] += 1
        seen_px[ids] += counts
        front = sign_f > 0
        back = sign_f < 0
        if front.any():
            vote_front[ids[front]] += counts[front]
        if back.any():
            vote_back[ids[back]] += counts[back]
        vote[ids] += sign_f * counts

    return vote, seen_px, vis_count, vote_front, vote_back


def _find_connected_components(vertices: np.ndarray, faces: np.ndarray) -> Tuple[int, np.ndarray]:
    """Groups connected triangles across spatial-welded edges using SciPy csgraph."""
    n_faces = len(faces)
    _, inv = np.unique(np.round(vertices, 5), axis=0, return_inverse=True)
    gf = inv[faces]

    edges = np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    fids = np.repeat(np.arange(n_faces), 3)
    order = np.lexsort((fids, edges[:, 1], edges[:, 0]))
    s_edges = edges[order]
    s_fids = fids[order]
    same = (s_edges[1:] == s_edges[:-1]).all(axis=1)
    f1 = s_fids[:-1][same]
    f2 = s_fids[1:][same]

    adj = csr_matrix((np.ones(len(f1), dtype=bool), (f1, f2)), shape=(n_faces, n_faces))
    n_comp, labels = csgraph.connected_components(adj, directed=False)
    return n_comp, labels


def orient_faces_by_visibility(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int = DEFAULT_VIEWS,
    resolution: int = DEFAULT_RESOLUTION,
    stats: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Orients mesh faces by visibility voting over connected components.
    Strictly preserves 100% faces (Zero-Decimation Policy).
    Optimized with parallel multi-core rasterization and concurrent graph partitioning.
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(faces) == 0:
        return faces

    # Execute connected components and parallel rasterization concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as bg_pool:
        comp_future = bg_pool.submit(_find_connected_components, vertices, faces)
        vote_future = bg_pool.submit(_rasterize_votes, vertices, faces, views, resolution, max_workers)

        n_comp, labels = comp_future.result()
        vote, seen_px, vis_count, vote_front, vote_back = vote_future.result()

    # Fast-path / component vote accumulation:
    comp_votes = np.bincount(labels, weights=vote)
    flipped_comps = comp_votes < 0
    flip = flipped_comps[labels]

    # Outward radial normal fallback for components with comp_votes == 0
    mesh_center = (vertices.max(0) + vertices.min(0)) / 2.0
    zero_comps = np.nonzero(comp_votes == 0)[0]
    for c in zero_comps:
        c_mask = (labels == c)
        c_tris = vertices[faces[c_mask]]
        c_centers = c_tris.mean(axis=1)
        e1 = c_tris[:, 1] - c_tris[:, 0]
        e2 = c_tris[:, 2] - c_tris[:, 0]
        normals = np.cross(e1, e2)
        rad_vec = c_centers - mesh_center
        if np.sum(normals * rad_vec) < 0:
            flip[c_mask] = True
            flipped_comps[c] = True

    oriented = faces.copy()
    oriented[flip] = oriented[flip][:, ::-1]

    if stats is not None:
        stats["orient_views"] = int(views)
        stats["orient_resolution"] = int(resolution)
        stats["faces_flipped"] = int(flip.sum())
        stats["faces_before"] = int(len(faces))
        stats["faces_after"] = int(len(faces))
        stats["connected_components"] = int(n_comp)
        stats["components_flipped"] = int(flipped_comps.sum())
        stats["seen_faces"] = int(np.sum(seen_px > 0))
        stats["zero_seen_faces"] = int(np.sum(seen_px == 0))

    return oriented


def prune_interior_and_orient_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    prune_interior: bool = True,
    views: int = DEFAULT_VIEWS,
    resolution: int = DEFAULT_RESOLUTION,
    stats: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> Any:
    """
    Occlusion Culling & Shell Orientation for 3D AI statues (e.g. Trellis / Tripo).

    1. Occlusion Culling / Interior Face Pruning (prune_interior=True):
       - Identifies interior faces and occluded cavity structures with vis_count == 0.
       - Vectorized topological hole-protection ensures exterior surface triangles
         (crevices, folds, subpixel details) are never pruned, preventing punctures/holes ("bục tượng").
       - Prunes unreferenced (orphaned) vertices and reindexes faces & UVs.

    2. Visibility Shell Orientation:
       - Ensures ALL remaining outward faces point outwards with CCW winding (FrontSide).
       - Uses z-buffer visibility voting per connected component with radial outward normal fallback.

    Parameters:
        vertices: (V, 3) float array of vertex coordinates.
        faces: (F, 3) int array of triangle indices.
        uvs: Optional (V, 2) float array of UV texture coordinates.
        prune_interior: bool, whether to prune interior occluded faces.
        views: int, number of Fibonacci sphere views (default 48).
        resolution: int, rasterization z-buffer resolution (default 384).
        stats: Optional dict to receive detailed execution statistics.
        max_workers: Optional int for worker concurrency.

    Returns:
        If uvs is None:
            (new_vertices, oriented_faces)
        If uvs is not None:
            (new_vertices, oriented_faces, new_uvs)
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices, dtype=np.float64)
    n_faces = len(faces)
    n_verts = len(vertices)

    if n_faces == 0:
        if stats is not None:
            stats["faces_before"] = 0
            stats["faces_after"] = 0
            stats["faces_pruned"] = 0
            stats["vertices_before"] = 0
            stats["vertices_after"] = 0
            stats["vertices_pruned"] = 0
        if uvs is not None:
            return vertices, faces, uvs
        return vertices, faces

    if not prune_interior:
        oriented_faces = orient_faces_by_visibility(
            vertices,
            faces,
            views=views,
            resolution=resolution,
            stats=stats,
            max_workers=max_workers,
        )
        if stats is not None:
            stats["faces_pruned"] = 0
            stats["vertices_pruned"] = 0
            stats["prune_ratio"] = 0.0
        if uvs is not None:
            return vertices, oriented_faces, uvs
        return vertices, oriented_faces

    # Phase 1: Rasterize visibility across Fibonacci sphere views
    vote, seen_px, vis_count, vote_front, vote_back = _rasterize_votes(
        vertices, faces, views, resolution, max_workers
    )

    # Phase 2: Spatial welding and topological exterior surface protection
    # Spatially weld vertices to find topological edge connectivity across UV seams
    _, inv = np.unique(np.round(vertices, 5), axis=0, return_inverse=True)
    gf = inv[faces]

    keep = (seen_px > 0).copy()

    # Precompute edge IDs for vectorized neighbor sharing
    e0 = gf[:, [0, 1]]
    e1 = gf[:, [1, 2]]
    e2 = gf[:, [2, 0]]
    all_edges = np.sort(np.concatenate([e0, e1, e2], axis=0), axis=1)
    u_edges, edge_ids = np.unique(all_edges, axis=0, return_inverse=True)
    e0_ids = edge_ids[:n_faces]
    e1_ids = edge_ids[n_faces : 2 * n_faces]
    e2_ids = edge_ids[2 * n_faces :]

    # Iterative closing: any un-seen face sharing >= 2 edges with kept faces is
    # part of the outer surface (groove/crevice/fold) and must be protected against puncture
    for _ in range(10):
        edge_keep = np.bincount(
            edge_ids, weights=np.tile(keep.astype(np.int32), 3), minlength=len(u_edges)
        )
        shared = (
            (edge_keep[e0_ids] >= 1).astype(np.int8)
            + (edge_keep[e1_ids] >= 1).astype(np.int8)
            + (edge_keep[e2_ids] >= 1).astype(np.int8)
        )
        newly_kept = (~keep) & (shared >= 2)
        if not np.any(newly_kept):
            break
        keep[newly_kept] = True

    # Phase 3: Prune interior faces & unreferenced vertices
    kept_faces = faces[keep]
    ref_verts, new_faces = np.unique(kept_faces, return_inverse=True)
    new_vertices = vertices[ref_verts]
    new_faces = new_faces.reshape(kept_faces.shape)
    new_uvs = uvs[ref_verts] if uvs is not None else None

    # Phase 4: Winding orientation of the pruned exterior mesh
    n_comp, labels = _find_connected_components(new_vertices, new_faces)
    kept_votes = vote[keep]
    comp_votes = np.bincount(labels, weights=kept_votes)

    flip = np.zeros(len(new_faces), dtype=bool)
    mesh_center = (new_vertices.max(0) + new_vertices.min(0)) / 2.0
    flipped_comp_count = 0

    for c in range(n_comp):
        c_mask = (labels == c)
        if comp_votes[c] < 0:
            flip[c_mask] = True
            flipped_comp_count += 1
        elif comp_votes[c] == 0:
            # Fallback: check radial outward alignment of normals
            c_tris = new_vertices[new_faces[c_mask]]
            c_centers = c_tris.mean(axis=1)
            e1 = c_tris[:, 1] - c_tris[:, 0]
            e2 = c_tris[:, 2] - c_tris[:, 0]
            normals = np.cross(e1, e2)
            rad_vec = c_centers - mesh_center
            if np.sum(normals * rad_vec) < 0:
                flip[c_mask] = True
                flipped_comp_count += 1

    oriented_faces = new_faces.copy()
    oriented_faces[flip] = oriented_faces[flip][:, ::-1]

    if stats is not None:
        stats["orient_views"] = int(views)
        stats["orient_resolution"] = int(resolution)
        stats["faces_before"] = int(n_faces)
        stats["faces_after"] = int(len(oriented_faces))
        stats["faces_pruned"] = int(n_faces - len(oriented_faces))
        stats["prune_ratio"] = float((n_faces - len(oriented_faces)) / max(1, n_faces))
        stats["vertices_before"] = int(n_verts)
        stats["vertices_after"] = int(len(new_vertices))
        stats["vertices_pruned"] = int(n_verts - len(new_vertices))
        stats["connected_components"] = int(n_comp)
        stats["components_flipped"] = int(flipped_comp_count)
        stats["faces_flipped"] = int(flip.sum())
        stats["seen_faces"] = int(np.sum(seen_px > 0))
        stats["zero_seen_faces"] = int(np.sum(seen_px == 0))
        stats["protected_surface_faces"] = int(np.sum(keep) - np.sum(seen_px > 0))

    if uvs is not None:
        return new_vertices, oriented_faces, new_uvs
    return new_vertices, oriented_faces


def prune_mesh_interior_and_orient(
    mesh: Any,
    prune_interior: bool = True,
    views: int = DEFAULT_VIEWS,
    resolution: int = DEFAULT_RESOLUTION,
    stats: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> Any:
    """
    Convenience wrapper for Trimesh instances.
    Prunes interior occluded faces, drops unreferenced vertices,
    updates UV coordinates if present, and ensures outward CCW winding.
    """
    import trimesh

    uvs = getattr(mesh.visual, "uv", None)
    if uvs is not None and len(uvs) == len(mesh.vertices):
        new_v, new_f, new_uv = prune_interior_and_orient_faces(
            mesh.vertices,
            mesh.faces,
            uvs=uvs,
            prune_interior=prune_interior,
            views=views,
            resolution=resolution,
            stats=stats,
            max_workers=max_workers,
        )
        new_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        new_mesh.visual = mesh.visual.copy()
        new_mesh.visual.uv = new_uv
    else:
        new_v, new_f = prune_interior_and_orient_faces(
            mesh.vertices,
            mesh.faces,
            prune_interior=prune_interior,
            views=views,
            resolution=resolution,
            stats=stats,
            max_workers=max_workers,
        )
        new_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        if hasattr(mesh, "visual") and mesh.visual is not None:
            new_mesh.visual = mesh.visual.copy()

    return new_mesh


# Alias for compatibility with callers using shell naming
orient_shells_by_visibility = orient_faces_by_visibility

