"""
shell_orient.py

Visibility-based winding orientation for 3D AI statues (e.g. Trellis / Tripo).
Solves thin-shell inside-out backface visibility issues.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles (prune_hidden=False).
- Groups connected components across UV seams.
- Flips inward-facing components to outward CCW FrontSide.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging
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
    """Vectorized z-buffer rasterization: returns res x res image with closest face ID per pixel."""
    x0 = np.clip(np.floor(sx.min(1)).astype(np.int32), 0, res)
    x1 = np.clip(np.ceil(sx.max(1)).astype(np.int32), 0, res)
    y0 = np.clip(np.floor(sy.min(1)).astype(np.int32), 0, res)
    y1 = np.clip(np.ceil(sy.max(1)).astype(np.int32), 0, res)
    w = (x1 - x0).astype(np.int64)
    h = (y1 - y0).astype(np.int64)
    area = w * h
    idx = np.nonzero(area > 0)[0]
    if len(idx) == 0:
        return np.full((res, res), -1, dtype=np.int32)

    reps = area[idx]
    total = int(reps.sum())
    fid = np.repeat(idx, reps)
    offset = np.concatenate([[0], np.cumsum(reps)])[:-1]
    seq = np.arange(total) - np.repeat(offset, reps)
    wf = np.repeat(w[idx], reps)
    px = np.repeat(x0[idx], reps) + (seq % wf)
    py = np.repeat(y0[idx], reps) + (seq // wf)

    ax, ay = sx[fid, 0], sy[fid, 0]
    bx, by = sx[fid, 1], sy[fid, 1]
    cx, cy = sx[fid, 2], sy[fid, 2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    good = np.abs(den) > 1e-12
    w0 = np.zeros(total)
    w1 = np.zeros(total)
    w0[good] = ((by - cy) * (px - cx) + (cx - bx) * (py - cy))[good] / den[good]
    w1[good] = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy))[good] / den[good]
    inside = good & (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
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


def _rasterize_votes(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes visibility votes for each face."""
    vote = np.zeros(len(faces), dtype=np.float64)
    seen_px = np.zeros(len(faces), dtype=np.float64)

    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    span = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    if span <= 0:
        return vote, seen_px

    for direction in _sphere_dirs(views):
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
            continue
        seen_px[ids] += counts
        vote[ids] += np.sign(facing[ids]) * counts

    return vote, seen_px


def orient_faces_by_visibility(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int = DEFAULT_VIEWS,
    resolution: int = DEFAULT_RESOLUTION,
    stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Orients mesh faces by visibility voting over connected components.
    Strictly preserves 100% faces (Zero-Decimation).
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(faces) == 0:
        return faces

    vote, _ = _rasterize_votes(vertices, faces, views, resolution)

    # Connected Component Winding:
    # Group connected triangles across edges using spatial-welded vertex indices
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

    return oriented
