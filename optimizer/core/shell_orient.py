"""
shell_orient.py

Visibility-based winding orientation for 3D AI statues (e.g. Trellis / Tripo).
Solves thin-shell inside-out backface visibility issues.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles (prune_hidden=False).
- Votes per UV piece (components joined by shared vertex indices, no positional welding),
  so oppositely-wound pieces meeting at a seam are oriented independently.
- Flips inward-facing components to outward CCW FrontSide.

High-Performance Parallel Architecture:
- Vectorized NumPy z-buffer with precomputed edge determinants (avoids degenerate allocations).
- Exact float64 lexsort tie-breaking for 100% bit-for-bit mathematical parity.
- Multi-core parallel view rasterization (process pool; a thread pool with identical output, logged,
  when the process pool cannot start).
- Concurrent connected-components graph decomposition.
"""

from typing import Tuple, Dict, Any, Optional
import logging
import os
import multiprocessing as mp
import concurrent.futures
import numpy as np
from scipy.sparse import csgraph, csr_matrix

from optimizer.core.errors import PipelineAbort

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = 48
DEFAULT_RESOLUTION = 384


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
    depth is per-vertex (F, 3); each pixel's depth is barycentrically interpolated (smallest wins).
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
    w0 = w0[inside]
    w1 = w1[inside]
    d = depth[fid]
    pix_depth = w0 * d[:, 0] + w1 * d[:, 1] + (1.0 - w0 - w1) * d[:, 2]
    lin = py[inside].astype(np.int64) * res + px[inside]
    order = np.lexsort((pix_depth, lin))
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
    depth = tri[:, :, 2]

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
    Uses a fork process pool across all Fibonacci camera angles; if that pool cannot start, a thread
    pool renders the same views (identical output) and a warning is logged. Worker errors propagate.
    A zero-size (or non-finite) bounding box raises PipelineAbort.
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
    if not np.isfinite(span) or span <= 0:
        raise PipelineAbort(
            f"Cannot orient faces: the mesh bounding box has zero size (extent {span}), so no view sees any face"
        )

    sphere_directions = _sphere_dirs(views)
    tasks = [(d, vertices, faces, centre, span, resolution) for d in sphere_directions]

    cpu_count = os.cpu_count() or 4
    workers = max_workers if max_workers is not None else min(cpu_count, 10)

    if workers > 1 and len(tasks) > 1:
        pool = None
        try:
            # Fork is fastest on macOS and Linux (zero process bootstrap overhead)
            pool = mp.get_context("fork").Pool(workers)
        except (ValueError, OSError) as err:
            # Perf-only fallback with identical output: no fork start method (ValueError), or the OS
            # refused the processes / semaphores (OSError). Errors inside the workers are not caught.
            logger.warning(
                "Process pool unavailable (%s: %s); rendering %d views on %d threads instead",
                type(err).__name__, err, len(tasks), workers
            )

        if pool is not None:
            chunksize = max(1, len(tasks) // (workers * 2))
            with pool:
                results = pool.map(_render_single_view_task, tasks, chunksize=chunksize)
        else:
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


def _find_connected_components(
    vertices: np.ndarray, faces: np.ndarray, weld: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Groups connected triangles by shared edges using SciPy csgraph.
    weld=True joins edges across spatially-welded vertices (UV seams);
    weld=False uses shared vertex indices only (one component per UV piece).
    """
    n_faces = len(faces)
    if weld:
        _, inv = np.unique(np.round(vertices, 5), axis=0, return_inverse=True)
        gf = inv[faces]
    else:
        gf = faces

    edges = np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    # Edge rows are stacked [e01; e12; e20], so row r belongs to face r % n_faces.
    fids = np.tile(np.arange(n_faces), 3)
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
    Orients mesh faces by visibility voting per UV piece (index-connected component).
    Pieces are not welded across seams, so neighbouring pieces with opposite winding
    are each flipped independently. Pieces never seen by the z-buffer (vote == 0)
    keep their source winding.
    Strictly preserves 100% faces (Zero-Decimation Policy).
    Optimized with parallel multi-core rasterization and concurrent graph partitioning.
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(faces) == 0:
        return faces

    # Execute connected components and parallel rasterization concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as bg_pool:
        comp_future = bg_pool.submit(_find_connected_components, vertices, faces, False)
        vote_future = bg_pool.submit(_rasterize_votes, vertices, faces, views, resolution, max_workers)

        n_comp, labels = comp_future.result()
        vote, seen_px, vis_count, vote_front, vote_back = vote_future.result()

    # Fast-path / component vote accumulation:
    comp_votes = np.bincount(labels, weights=vote)
    flipped_comps = comp_votes < 0
    flip = flipped_comps[labels]

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
