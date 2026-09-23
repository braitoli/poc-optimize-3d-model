"""
uv_baker.py

UV Atlas Re-charting, Direct Barycentric Baking, and 16px Boundary Dilation.
Maximizes texture canvas utilization, Texel Density, and surface sharpness.
Ensures zero black edge bleeding during GPU texture mipmapping.
"""

import math
from typing import Callable, Tuple, Optional, Dict, List, Any
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csgraph, csr_matrix
import trimesh
import xatlas
from optimizer.core.errors import PipelineAbort
from optimizer.core.uvatlas import unwrap_mesh_uvatlas
from optimizer.core.texture_utils import optimize_mesh_texture_for_export


def _sample_texture_bilinear(image_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Samples RGB or RGBA image at normalized UV coordinates [0, 1] using bilinear interpolation."""
    h, w = image_rgb.shape[:2]
    # Clamp UVs strictly to [0.0, 1.0] to prevent wrapping artifacts (white seams from canvas border)
    u_norm = np.clip(uv[:, 0], 0.0, 1.0)
    v_norm = np.clip(uv[:, 1], 0.0, 1.0)

    u = u_norm * (w - 1)
    v = (1.0 - v_norm) * (h - 1)

    x0 = np.floor(u).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(v).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)

    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]

    c00 = image_rgb[y0, x0].astype(np.float32)
    c10 = image_rgb[y0, x1].astype(np.float32)
    c01 = image_rgb[y1, x0].astype(np.float32)
    c11 = image_rgb[y1, x1].astype(np.float32)

    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bot * fy


def _rasterize_uv_atlas(
    faces: np.ndarray,
    uv: np.ndarray,
    dim: int,
    max_batch_samples: int = 4000000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterizes UV atlas into a (dim x dim) grid using memory-bounded direct canvas buffering.
    Eliminates heavy list concatenation and large-array sorting for extreme speed.
    Returns:
      sel: 1D flat pixel indices covered by triangles (sorted)
      fid: triangle index covering each selected pixel
      bary: barycentric weights (N, 3) for each selected pixel
    """
    if len(faces) == 0 or len(uv) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))

    tri_uv = uv[faces]  # (N, 3, 2)
    px = tri_uv[:, :, 0] * dim
    py = (1.0 - tri_uv[:, :, 1]) * dim

    min_x = np.clip(np.floor(px.min(axis=1) - 0.5).astype(np.int32), 0, dim - 1)
    max_x = np.clip(np.ceil(px.max(axis=1) - 0.5).astype(np.int32), 0, dim - 1)
    min_y = np.clip(np.floor(py.min(axis=1) - 0.5).astype(np.int32), 0, dim - 1)
    max_y = np.clip(np.ceil(py.max(axis=1) - 0.5).astype(np.int32), 0, dim - 1)

    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    areas = span_x * span_y

    valid = areas > 0
    if not np.any(valid):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))

    idx = np.where(valid)[0]
    reps = areas[idx]

    canvas_fid = np.full(dim * dim, -1, dtype=np.int64)
    canvas_bary = np.zeros((dim * dim, 3), dtype=np.float64)

    cum = np.concatenate([[0], np.cumsum(reps)])
    total_samples = cum[-1]
    batch_start_sample = 0

    while batch_start_sample < total_samples:
        batch_end_sample = min(batch_start_sample + max_batch_samples, total_samples)
        t_start = np.searchsorted(cum, batch_start_sample, side='right') - 1
        t_end = np.searchsorted(cum, batch_end_sample, side='right')
        t_end = max(t_end, t_start + 1)
        t_end = min(t_end, len(idx))

        b_idx = idx[t_start:t_end]
        b_reps = reps[t_start:t_end]
        b_total = int(np.sum(b_reps))

        b_fids = np.repeat(b_idx, b_reps)
        b_offsets = np.concatenate([[0], np.cumsum(b_reps)])[:-1]
        b_seq = np.arange(b_total) - np.repeat(b_offsets, b_reps)
        b_sx = span_x[b_fids]

        gx = min_x[b_fids] + (b_seq % b_sx)
        gy = min_y[b_fids] + (b_seq // b_sx)

        p0x, p0y = px[b_fids, 0], py[b_fids, 0]
        p1x, p1y = px[b_fids, 1], py[b_fids, 1]
        p2x, p2y = px[b_fids, 2], py[b_fids, 2]

        det = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
        nonzero = np.abs(det) > 1e-10

        gx_s = gx[nonzero]
        gy_s = gy[nonzero]
        ff = b_fids[nonzero]
        d = det[nonzero]

        p0x_s, p0y_s = p0x[nonzero], p0y[nonzero]
        p1x_s, p1y_s = p1x[nonzero], p1y[nonzero]
        p2x_s, p2y_s = p2x[nonzero], p2y[nonzero]

        # Evaluate barycentric coordinates at exact pixel center (gx + 0.5, gy + 0.5)
        gx_c = gx_s.astype(np.float64) + 0.5
        gy_c = gy_s.astype(np.float64) + 0.5

        w0 = ((p1y_s - p2y_s) * (gx_c - p2x_s) + (p2x_s - p1x_s) * (gy_c - p2y_s)) / d
        w1 = ((p2y_s - p0y_s) * (gx_c - p2x_s) + (p0x_s - p2x_s) * (gy_c - p2y_s)) / d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if np.any(inside):
            pix_idx = gy_s[inside] * dim + gx_s[inside]
            canvas_fid[pix_idx] = ff[inside]
            canvas_bary[pix_idx, 0] = w0[inside]
            canvas_bary[pix_idx, 1] = w1[inside]
            canvas_bary[pix_idx, 2] = w2[inside]

        batch_start_sample = cum[t_end]

    # Sub-pixel Triangle Splatting: Ensure 100% triangles have at least one rasterized texel
    assigned_triangles = np.unique(canvas_fid[canvas_fid >= 0])
    unassigned = np.setdiff1d(idx, assigned_triangles)

    if len(unassigned) > 0:
        c_px = px[unassigned].mean(axis=1)
        c_py = py[unassigned].mean(axis=1)
        splat_gx = np.clip(np.floor(c_px).astype(np.int64), 0, dim - 1)
        splat_gy = np.clip(np.floor(c_py).astype(np.int64), 0, dim - 1)
        splat_pix = splat_gy * dim + splat_gx

        unocc = canvas_fid[splat_pix] < 0
        if np.any(unocc):
            canvas_fid[splat_pix[unocc]] = unassigned[unocc]
            canvas_bary[splat_pix[unocc]] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        rem = canvas_fid[splat_pix] != unassigned
        if np.any(rem):
            canvas_fid[splat_pix[rem]] = unassigned[rem]
            canvas_bary[splat_pix[rem]] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    sel = np.where(canvas_fid >= 0)[0]
    if len(sel) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))
    return sel, canvas_fid[sel], canvas_bary[sel]


def dilate_texture(image_rgb: np.ndarray, mask_covered: np.ndarray, padding: int = 16) -> np.ndarray:
    """
    Dilates covered pixel colors into non-covered background by `padding` pixels.
    Eliminates dark / black edge artifacts when generating GPU mipmaps.
    Optimized: Restricts distance transform to the bounding box of active dilation boundaries.
    """
    if mask_covered.all():
        return image_rgb
    if not np.any(mask_covered):
        return image_rgb

    uncovered = ~mask_covered
    if not np.any(uncovered):
        return image_rgb

    h, w = mask_covered.shape[:2]

    # Quick 1D projections to find bounding ranges
    c_rows = np.any(mask_covered, axis=1)
    c_cols = np.any(mask_covered, axis=0)
    c_rmin, c_rmax = np.where(c_rows)[0][[0, -1]]
    c_cmin, c_cmax = np.where(c_cols)[0][[0, -1]]

    u_rows = np.any(uncovered, axis=1)
    u_cols = np.any(uncovered, axis=0)
    u_rmin, u_rmax = np.where(u_rows)[0][[0, -1]]
    u_cmin, u_cmax = np.where(u_cols)[0][[0, -1]]

    # Tightest sub-region where dilation can possibly have an effect
    sub_rmin = max(0, max(u_rmin - padding, c_rmin - padding))
    sub_rmax = min(h, min(u_rmax + padding, c_rmax + padding) + 1)
    sub_cmin = max(0, max(u_cmin - padding, c_cmin - padding))
    sub_cmax = min(w, min(u_cmax + padding, c_cmax + padding) + 1)

    if sub_rmin >= sub_rmax or sub_cmin >= sub_cmax:
        return image_rgb

    sub_uncovered = uncovered[sub_rmin:sub_rmax, sub_cmin:sub_cmax]
    if not np.any(sub_uncovered):
        return image_rgb

    sub_img = image_rgb[sub_rmin:sub_rmax, sub_cmin:sub_cmax]
    dist, near = ndimage.distance_transform_edt(sub_uncovered, return_distances=True, return_indices=True)
    grow = sub_uncovered & (dist <= padding)

    out = image_rgb.copy()
    sub_out = sub_img.copy()
    sub_out[grow] = sub_img[near[0][grow], near[1][grow]]
    out[sub_rmin:sub_rmax, sub_cmin:sub_cmax] = sub_out
    return out


def get_adaptive_chart_options(n_faces: int, merge_level: int = 0) -> xatlas.ChartOptions:
    """
    Configures xatlas.ChartOptions with adaptive iterations and micro-chart merging
    to achieve ultra-fast unwrapping even on large meshes (>200k faces like Gravilux).

    `merge_level` above 0 lets xatlas grow charts further before it cuts a new one: the cost
    ceiling doubles per level while the penalties that stop a chart from spreading (normal
    deviation, straightness, both seam weights) are halved. Fewer, larger islands mean less chart
    boundary, so less of the canvas goes into gutter padding - at the price of more UV distortion,
    which _fit_xatlas answers by packing at a higher texel density. plan_uv_canvas charts at
    several levels and keeps the one that needs the smallest canvas.
    """
    c_opts = xatlas.ChartOptions()
    if n_faces > 100_000:
        c_opts.max_iterations = 1
        c_opts.max_cost = 4.0
        c_opts.normal_deviation_weight = 0.5
        c_opts.roundness_weight = 0.0
        c_opts.straightness_weight = 1.0
        c_opts.normal_seam_weight = 1.0
        c_opts.texture_seam_weight = 0.25
        c_opts.fix_winding = False
    elif n_faces > 40_000:
        c_opts.max_iterations = 2
        c_opts.max_cost = 3.0
        c_opts.normal_deviation_weight = 1.0
        c_opts.roundness_weight = 0.005
        c_opts.straightness_weight = 3.0
        c_opts.normal_seam_weight = 2.0
        c_opts.texture_seam_weight = 0.5
        c_opts.fix_winding = True
    else:
        c_opts.max_iterations = 4
        c_opts.max_cost = 2.0
        c_opts.normal_deviation_weight = 2.0
        c_opts.roundness_weight = 0.01
        c_opts.straightness_weight = 6.0
        c_opts.normal_seam_weight = 4.0
        c_opts.texture_seam_weight = 0.5
        c_opts.fix_winding = True

    if merge_level > 0:
        factor = float(2 ** merge_level)
        c_opts.max_cost *= factor
        c_opts.normal_deviation_weight /= factor
        c_opts.straightness_weight /= factor
        c_opts.normal_seam_weight /= factor
        c_opts.texture_seam_weight /= factor
    return c_opts


def get_adaptive_pack_options(n_faces: int, target_res: int, padding: int = 2) -> xatlas.PackOptions:
    """
    Configures xatlas.PackOptions with strict non-rotation.
    Eliminates diagonal chart rotation blur ('chống nhòe do xoay chéo')
    and preserves exact source pixel grid alignment.
    """
    p_opts = xatlas.PackOptions()
    p_opts.resolution = target_res
    p_opts.padding = padding
    p_opts.bilinear = True
    p_opts.bruteForce = False
    # Anti-blur: zero chart rotation to prevent diagonal pixel resampling blurring
    p_opts.rotate_charts = False
    p_opts.rotate_charts_to_axis = False
    return p_opts


SIZE_MODES = ("exact", "pot-up", "pot-down")
UNWRAP_METHODS = ("xatlas", "uvatlas")

CANVAS_BLOCK_PX = 4           # KTX2 / GPU block size: canvas sides are multiples of 4
CANVAS_MARGIN_PX = 4          # border kept free around the islands by maximize_uv_bounds
XATLAS_PADDING_PX = 2         # xatlas chart padding (+1 bilinear texel) in final-canvas pixels
MIN_TEXEL_DENSITY_RATIO = 0.97  # exact / pot-up must keep T_new / T_src >= this
MAX_PACK_PASSES = 4           # bound on every sizing loop (xatlas re-packs, UVAtlas passes)
ISLAND_MERGE_LEVELS = 3       # chart aggressiveness levels tried when merging islands (0 = as-is)
# UVAtlas merges charts by tolerating more stretch instead; one value per merge level
UVATLAS_MERGE_STRETCH = (0.1667, 0.3333, 0.5)
UVATLAS_INITIAL_FILL = 0.5    # first guess of the UV area fraction UVAtlas fills (gutter 4 px)


def _check_source_uv(mesh: trimesh.Trimesh, source_uv: np.ndarray) -> np.ndarray:
    """Source UVs as (V, 2) float64, one finite UV per mesh vertex; anything else raises PipelineAbort."""
    uv = np.asarray(source_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != len(mesh.vertices):
        raise PipelineAbort(
            f"Got {len(uv) if uv.ndim else 0} source UVs for {len(mesh.vertices)} vertices "
            f"(shape {uv.shape}): expected one (u, v) per vertex"
        )
    bad = ~np.isfinite(uv).all(axis=1)
    if bad.any():
        raise PipelineAbort(f"{int(bad.sum())} source UVs are non-finite (NaN / inf)")
    return uv


def _check_no_nan(values: np.ndarray, what: str) -> np.ndarray:
    """Sampled texel values must be numbers; a NaN would otherwise be painted as a made-up colour."""
    nan = np.isnan(values)
    if nan.any():
        raise PipelineAbort(f"Sampling the {what} produced {int(nan.any(axis=-1).sum())} NaN texels")
    return values


def uv_boundary_length(uv: np.ndarray, faces: np.ndarray) -> float:
    """
    Total length of the atlas's island outlines, in the units `uv` is given in (texels when the
    UVs are in atlas pixels). An edge used by a single triangle is an island border: it is where
    the packer has to spend gutter padding and where the GPU sees a seam, so the smaller this is,
    the less of the canvas is wasted on padding.
    """
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    unique_edges = np.unique(edges, axis=0)[counts == 1]
    if len(unique_edges) == 0:
        return 0.0
    segments = uv[unique_edges[:, 0]] - uv[unique_edges[:, 1]]
    return float(np.linalg.norm(segments, axis=1).sum())


def _count_uv_islands(faces: np.ndarray) -> int:
    """Number of connected components of the UV layout: one per island the packer has to place."""
    n_vertices = int(faces.max()) + 1 if len(faces) else 0
    if n_vertices == 0:
        return 0
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices)
    )
    return int(csgraph.connected_components(graph, directed=False)[0])


def _uv_triangle_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def texel_density(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int
) -> float:
    """
    Average texel density in texels per 3D unit:
        T = sqrt( sum_f uvArea_f * W * H / sum_f area3D_f )
    Summed per face, so overlapping or mirrored UVs count once per use (a re-chart gives every
    face its own space).
    """
    faces = np.asarray(faces, dtype=np.int64)
    area_3d = float(trimesh.triangles.area(np.asarray(vertices, dtype=np.float64)[faces]).sum())
    if area_3d <= 0.0:
        raise ValueError("Mesh has zero surface area: texel density is undefined")
    area_uv = float(_uv_triangle_areas(uv, faces).sum())
    return math.sqrt(area_uv * float(width) * float(height) / area_3d)


def _round_up_to_block(size: float) -> int:
    return int(math.ceil(size / CANVAS_BLOCK_PX)) * CANVAS_BLOCK_PX


def _pow2_at_least(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _pow2_at_most(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


def _layout_texel_density(layout: Dict[str, Any]) -> float:
    canvas = layout["canvas"]
    return texel_density(layout["vertices"], layout["faces"], layout["uv"], canvas, canvas)


def _xatlas_natural_atlas(
    mesh: trimesh.Trimesh,
    texels_per_unit: float,
    merge_level: int = 0
) -> Dict[str, Any]:
    """
    Charts and packs the mesh with xatlas at `texels_per_unit` and resolution 0, i.e. one atlas
    whose size xatlas grows until every chart fits. Returns the layout in atlas pixels.
    """
    n_faces = len(mesh.faces)
    c_opts = get_adaptive_chart_options(n_faces, merge_level)
    p_opts = get_adaptive_pack_options(n_faces, target_res=0, padding=XATLAS_PADDING_PX)
    p_opts.texels_per_unit = float(texels_per_unit)

    atlas = xatlas.Atlas()
    # 3D mesh unwrap (not add_uv_mesh): no giant sliver triangles or unassigned (0, 0) UVs
    atlas.add_mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    )
    atlas.generate(chart_options=c_opts, pack_options=p_opts)
    if atlas.atlas_count != 1:
        raise RuntimeError(
            f"xatlas packed {atlas.atlas_count} atlases at {texels_per_unit:.3f} texels/unit "
            f"(expected exactly one)"
        )

    vmapping, indices, uv = atlas[0]
    vmapping = np.asarray(vmapping, dtype=np.int64)
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float64)[vmapping],
        "faces": np.asarray(indices, dtype=np.int64),
        "vmapping": vmapping,
        # xatlas normalises UVs by the atlas width / height: back to atlas pixels
        "uv_px": np.asarray(uv, dtype=np.float64) * np.array([atlas.width, atlas.height], dtype=np.float64),
        "extent": max(int(atlas.width), int(atlas.height)),
        "texels_per_unit": float(texels_per_unit),
        "meta": {
            "xatlas_chart_count": int(atlas.chart_count),
            "xatlas_atlas_count": int(atlas.atlas_count),
            "xatlas_atlas_size": f"{int(atlas.width)}x{int(atlas.height)}",
            "xatlas_texels_per_unit": round(float(texels_per_unit), 4),
            "xatlas_utilization_percent": round(float(atlas.utilization * 100.0), 2),
            "xatlas_merge_level": int(merge_level),
        }
    }


def _place_on_canvas(natural: Dict[str, Any], canvas: int) -> Dict[str, Any]:
    """Puts an xatlas pixel layout on a canvas x canvas texture; maximize_uv_bounds then scales the
    islands uniformly (never below their packed size when the atlas fits) to fill the canvas."""
    uv = maximize_uv_bounds(natural["uv_px"] / float(canvas), target_res=canvas, padding_px=CANVAS_MARGIN_PX)
    return {
        "vertices": natural["vertices"],
        "faces": natural["faces"],
        "vmapping": natural["vmapping"],
        "uv": uv,
        "canvas": canvas,
        "meta": dict(natural["meta"]),
    }


def _uvatlas_layout(
    mesh: trimesh.Trimesh,
    canvas: int,
    gutter: float,
    merge_level: int = 0
) -> Dict[str, Any]:
    """Microsoft UVAtlas unwrap packed into a canvas x canvas square (gutter in canvas pixels).
    UVAtlas merges charts by tolerating more stretch, so that is what the merge level raises."""
    vertices, faces, uv, vmapping, meta = unwrap_mesh_uvatlas(
        mesh=mesh,
        target_res=canvas,
        gutter=max(4.0, float(gutter)),
        max_stretch=UVATLAS_MERGE_STRETCH[merge_level]
    )
    uv = maximize_uv_bounds(np.asarray(uv, dtype=np.float64), target_res=canvas, padding_px=CANVAS_MARGIN_PX)
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "vmapping": np.asarray(vmapping, dtype=np.int64),
        "uv": uv,
        "canvas": canvas,
        "meta": {**dict(meta), "uvatlas_merge_level": int(merge_level)},
    }


def _fit_xatlas(mesh: trimesh.Trimesh, t_src: float, merge_level: int = 0) -> Dict[str, Any]:
    """
    Exact fit with xatlas: pack at texels_per_unit = T_src (xatlas scales every chart to that
    density) into a single atlas sized to fit, then S = max(atlas side) + border margins, rounded
    up to the block size. xatlas can land slightly under the requested density; if T_new / T_src
    falls below the minimum, re-pack at a proportionally higher density (bounded).
    """
    tpu = t_src
    tried = []
    for _ in range(MAX_PACK_PASSES):
        natural = _xatlas_natural_atlas(mesh, tpu, merge_level)
        fit = _round_up_to_block(natural["extent"] + 2 * CANVAS_MARGIN_PX)
        layout = _place_on_canvas(natural, fit)
        ratio = _layout_texel_density(layout) / t_src
        tried.append((fit, round(ratio, 4)))
        if ratio >= MIN_TEXEL_DENSITY_RATIO:
            return {"fit": fit, "natural": natural, "layouts": {fit: layout}, "passes": len(tried)}
        tpu /= ratio
    raise RuntimeError(
        f"xatlas could not pack at {MIN_TEXEL_DENSITY_RATIO} of the source texel density "
        f"within {MAX_PACK_PASSES} passes (canvas, ratio): {tried}"
    )


def _fit_uvatlas(
    mesh: trimesh.Trimesh,
    t_src: float,
    source_px_area: float,
    gutter: float,
    merge_level: int = 0
) -> Dict[str, Any]:
    """
    Exact fit with UVAtlas. UVAtlas always scales its charts to fill the square it is given, so the
    density grows about linearly with S; the fixed-pixel gutter makes the filled fraction grow
    slightly with S as well. Iterate S <- S / (T_new / T_src) (bounded) and keep the smallest S
    whose layout reaches the minimum density ratio.
    """
    canvas = _round_up_to_block(math.sqrt(source_px_area / UVATLAS_INITIAL_FILL))
    layouts: Dict[int, Dict[str, Any]] = {}
    ratios: Dict[int, float] = {}
    for _ in range(MAX_PACK_PASSES):
        layout = _uvatlas_layout(mesh, canvas, gutter, merge_level)
        ratio = _layout_texel_density(layout) / t_src
        if ratio <= 0.0:
            raise RuntimeError(f"UVAtlas produced a layout with zero UV area at {canvas}x{canvas}")
        layouts[canvas] = layout
        ratios[canvas] = round(ratio, 4)
        next_canvas = _round_up_to_block(canvas / ratio)
        if next_canvas in layouts:
            break
        canvas = next_canvas

    fitting = [s for s, r in ratios.items() if r >= MIN_TEXEL_DENSITY_RATIO]
    if not fitting:
        raise RuntimeError(
            f"UVAtlas did not reach {MIN_TEXEL_DENSITY_RATIO} of the source texel density "
            f"within {MAX_PACK_PASSES} passes (canvas: ratio): {ratios}"
        )
    return {"fit": min(fitting), "layouts": layouts, "passes": len(ratios)}


# ---------------------------------------------------------------------------
# Flat-colour swatches: faces whose source colour is uniform need no texels of
# their own. They are kept out of the chart pass and share one small swatch per
# colour, so the canvas only has to hold the faces that actually carry detail.
# ---------------------------------------------------------------------------

FLAT_SWATCH_PX = 8            # side of one colour swatch, in final-canvas pixels
# Gutter around every swatch. With the swatch side this makes the cell pitch a multiple of the KTX2
# block size and every cell block-aligned, so no 4x4 block ever straddles two swatch colours.
FLAT_SWATCH_GUTTER_PX = 4
FLAT_SAMPLES_PER_EDGE = 4     # barycentric grid per face: (n+1)(n+2)/2 = 15 samples


def _barycentric_grid(n: int = FLAT_SAMPLES_PER_EDGE) -> np.ndarray:
    """(S, 3) barycentric weights of a regular grid over a triangle, corners included."""
    pts = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.asarray(pts, dtype=np.float64)


def _face_sample_uv(
    mesh: trimesh.Trimesh,
    source_uv: np.ndarray,
    bary: np.ndarray,
    source_uv_sampler: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """(F, S, 2) source UV of every sample point: interpolated, or looked up on the textured
    original surface when Step 3 collapsed the edges this mesh is made of."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if source_uv_sampler is None:
        return np.einsum("sk,fkj->fsj", bary, source_uv[faces])
    pos = np.einsum("sk,fkj->fsj", bary, np.asarray(mesh.vertices, dtype=np.float64)[faces])
    return np.clip(source_uv_sampler(pos.reshape(-1, 3)), 0.0, 1.0).reshape(len(faces), len(bary), 2)


def _slot_images(mat: Any) -> Dict[str, np.ndarray]:
    """Every texture slot the bake re-creates, as (H, W, 3) uint8. A face counts as flat only if
    it is flat in all of them: a flat base colour over a detailed normal map is not flat."""
    imgs: Dict[str, np.ndarray] = {}
    for slot in ("baseColorTexture", "normalTexture") + _RESAMPLED_TEXTURE_SLOTS:
        rgb = _slot_rgb(mat, slot)
        if rgb is not None:
            imgs[slot] = rgb
    return imgs


def classify_flat_faces(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    tolerance: float,
    min_group_faces: int,
    material: Any = None,
    source_uv_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Finds the faces whose source colour is uniform and groups them.

    A face is flat when, over a barycentric grid of samples, every re-baked texture slot varies by
    at most `tolerance` (0..255, per channel). Adjacent flat faces whose mean colours are within
    `tolerance` grow into one group, and groups whose colours agree share a swatch. The guarantee
    is then enforced directly rather than argued from those steps: every sample of every swatched
    face is compared against the swatch colour it would get, and a face with a sample further than
    `tolerance` away is put back with the detailed faces. So no point of a swatched face ever moves
    further than `tolerance`, whatever region growing did.
    Groups smaller than `min_group_faces` are left to the chart pass
    (cutting a tiny hole out of the mesh adds more chart boundary than the texels it saves).

    Returns {flat, group, cluster, cluster_colors, stats}: per-face masks / ids over `mesh.faces`
    (-1 = not flat), one colour per cluster and slot, and the numbers Step 4 reports.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)
    slots = _slot_images(material) if material is not None else {}
    if "baseColorTexture" not in slots:
        slots = {"baseColorTexture": np.asarray(source_image.convert("RGB"), dtype=np.uint8), **slots}

    bary = _barycentric_grid()
    sample_uv = _face_sample_uv(mesh, source_uv, bary, source_uv_sampler)
    flat = np.ones(n_faces, dtype=bool)
    means: Dict[str, np.ndarray] = {}
    means_samples: Dict[str, np.ndarray] = {}
    for slot, rgb in slots.items():
        values = _sample_texture_bilinear(rgb, sample_uv.reshape(-1, 2)).reshape(n_faces, len(bary), 3)
        flat &= (values.max(axis=1) - values.min(axis=1)).max(axis=1) <= tolerance
        means[slot] = values.mean(axis=1)
        means_samples[slot] = values.astype(np.float32)  # kept for the enforcement pass below

    base_mean = means["baseColorTexture"]
    group = np.full(n_faces, -1, dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) and flat.any():
        same = (
            flat[adjacency[:, 0]] & flat[adjacency[:, 1]]
            & (np.abs(base_mean[adjacency[:, 0]] - base_mean[adjacency[:, 1]]).max(axis=1) <= tolerance)
        )
        edges = adjacency[same]
        graph = csr_matrix(
            (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
            shape=(n_faces, n_faces)
        )
        _, labels = csgraph.connected_components(graph, directed=False)

        # Keep the groups tight before they are clustered; the hard bound is enforced afterwards
        n_labels = int(labels.max()) + 1
        for _ in range(2):
            counts = np.bincount(labels[flat], minlength=n_labels)
            centre = np.stack([
                np.bincount(labels[flat], weights=base_mean[flat][:, channel], minlength=n_labels)
                for channel in range(3)
            ], axis=1) / np.maximum(counts, 1)[:, None]
            drift = np.abs(base_mean - centre[labels]).max(axis=1)
            keep = flat & (drift <= tolerance / 2.0)
            if keep.sum() == flat.sum():
                break
            flat = keep

        sizes = np.bincount(labels[flat], minlength=n_labels)
        big_enough = flat & (sizes[labels] >= max(1, int(min_group_faces)))
        flat = big_enough
        if flat.any():
            group[flat] = np.unique(labels[flat], return_inverse=True)[1]

    n_groups = int(group.max()) + 1 if flat.any() else 0
    cluster = np.full(n_faces, -1, dtype=np.int64)
    cluster_colors: Dict[str, List[List[int]]] = {slot: [] for slot in slots}
    if n_groups:
        group_size = np.bincount(group[flat], minlength=n_groups)
        group_mean = {
            slot: np.stack([
                np.bincount(group[flat], weights=values[flat][:, channel], minlength=n_groups)
                for channel in range(3)
            ], axis=1) / np.maximum(group_size, 1)[:, None]
            for slot, values in means.items()
        }
        group_cluster = np.full(n_groups, -1, dtype=np.int64)
        centres: List[Dict[str, np.ndarray]] = []
        for g in np.argsort(-group_size):          # biggest group defines a cluster's colour
            for c, centre in enumerate(centres):
                if all(
                    np.abs(group_mean[slot][g] - centre[slot]).max() <= tolerance / 2.0
                    for slot in slots
                ):
                    group_cluster[g] = c
                    break
            else:
                centres.append({slot: group_mean[slot][g] for slot in slots})
                group_cluster[g] = len(centres) - 1
        cluster[flat] = group_cluster[group[flat]]
        colour = {
            slot: np.clip(np.round(np.stack([c[slot] for c in centres])), 0, 255)
            for slot in slots
        }

        # Hard guarantee: drop any face that has a sample further than `tolerance` from the colour
        # its swatch would paint. Whatever the grouping did, what is left respects the promise.
        over = np.zeros(n_faces, dtype=bool)
        for slot, values in means_samples.items():
            deviation = np.abs(values[flat] - colour[slot][cluster[flat]][:, None, :]).max(axis=(1, 2))
            over[np.where(flat)[0][deviation > tolerance]] = True
        if over.any():
            flat = flat & ~over
            group[over] = -1
            cluster[over] = -1
            # A group that fell below the minimum, and a swatch nothing points at any more, go too
            if flat.any():
                sizes = np.bincount(group[flat], minlength=n_groups)
                too_small = flat & (sizes[group] < max(1, int(min_group_faces)))
                flat = flat & ~too_small
                group[too_small] = -1
                cluster[too_small] = -1
            used = np.unique(cluster[flat]) if flat.any() else np.array([], dtype=np.int64)
            renumber = np.full(len(centres), -1, dtype=np.int64)
            renumber[used] = np.arange(len(used))
            cluster[flat] = renumber[cluster[flat]]
            colour = {slot: colour[slot][used] for slot in slots}

        cluster_colors = {
            slot: colour[slot].astype(np.uint8).tolist() for slot in slots
        }

    n_clusters = len(cluster_colors.get("baseColorTexture", []))
    area = trimesh.triangles.area(np.asarray(mesh.vertices, dtype=np.float64)[faces])
    return {
        "flat": flat,
        "group": group,
        "cluster": cluster,
        "cluster_colors": cluster_colors,
        "stats": {
            "tolerance": float(tolerance),
            "minGroupFaces": int(min_group_faces),
            "slotsChecked": list(slots),
            "flatFaces": int(flat.sum()),
            "faces": int(n_faces),
            "flatFacesPercent": round(float(flat.mean()) * 100.0, 2),
            "flatAreaPercent": round(float(area[flat].sum() / max(area.sum(), 1e-12)) * 100.0, 2),
            "groups": int(n_groups),
            "swatches": int(n_clusters),
        }
    }


def _swatch_strip_px(n_clusters: int, island_side: int) -> int:
    """Height of the strip that holds `n_clusters` swatches under an island_side-wide atlas."""
    if n_clusters <= 0:
        return 0
    cell = FLAT_SWATCH_PX + 2 * FLAT_SWATCH_GUTTER_PX
    columns = max(1, (island_side - 2 * CANVAS_MARGIN_PX) // cell)
    rows = int(math.ceil(n_clusters / columns))
    return rows * cell + CANVAS_MARGIN_PX


def swatch_cells(n_clusters: int, canvas: int, strip_px: int) -> np.ndarray:
    """
    (C, 4) pixel rectangles [x0, y0, x1, y1) of every swatch, laid out in the strip along the
    bottom of the canvas (y grows downwards, as the baked image is indexed).
    """
    cell = FLAT_SWATCH_PX + 2 * FLAT_SWATCH_GUTTER_PX
    columns = max(1, (canvas - 2 * CANVAS_MARGIN_PX) // cell)
    out = np.zeros((n_clusters, 4), dtype=np.int64)
    for c in range(n_clusters):
        col, row = c % columns, c // columns
        x0 = CANVAS_MARGIN_PX + col * cell + FLAT_SWATCH_GUTTER_PX
        y0 = canvas - strip_px + row * cell + FLAT_SWATCH_GUTTER_PX
        out[c] = (x0, y0, x0 + FLAT_SWATCH_PX, y0 + FLAT_SWATCH_PX)
    return out


def _swatch_uv(cells: np.ndarray, canvas: int) -> np.ndarray:
    """(C, 2) UV of each swatch centre, in the v-up space the baked mesh uses."""
    centre_x = (cells[:, 0] + cells[:, 2]) / 2.0
    centre_y = (cells[:, 1] + cells[:, 3]) / 2.0
    return np.stack([centre_x / canvas, 1.0 - centre_y / canvas], axis=1)


def _submesh_for_charting(mesh: trimesh.Trimesh, keep: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """The mesh the chart pass sees when the flat faces are held out, plus the vertex index map
    back to `mesh` (so every layout can still be read against the original vertices / UVs)."""
    kept_faces = np.asarray(mesh.faces, dtype=np.int64)[keep]
    used = np.unique(kept_faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    sub = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[used],
        faces=remap[kept_faces],
        process=False
    )
    return sub, used


def _place_in_box(layout: Dict[str, Any], canvas: int, strip_px: int) -> Dict[str, Any]:
    """Moves an island layout packed for a `canvas - strip_px` square into the top of a
    `canvas` square, leaving the bottom strip free for the swatches. The islands keep their pixel
    size, so the texel density of the layout is unchanged."""
    if strip_px <= 0:
        return layout
    box = canvas - strip_px
    scale = box / float(canvas)
    uv = np.asarray(layout["uv"], dtype=np.float64).copy()
    uv[:, 0] *= scale
    uv[:, 1] = strip_px / float(canvas) + uv[:, 1] * scale
    return {**layout, "uv": uv, "canvas": canvas}



def plan_uv_canvas(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    size_mode: str = "exact",
    unwrap_method: str = "xatlas",
    uvatlas_gutter: float = 4.0,
    density_mesh: Optional[trimesh.Trimesh] = None,
    density_uv: Optional[np.ndarray] = None,
    merge_islands: bool = True,
    flat_swatch: bool = False,
    flat_tolerance: float = 8.0,
    flat_min_group_faces: int = 16,
    source_uv_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Sizes the square re-chart canvas at the source's 1:1 average texel density (see
    texel_density; source texture W x H, source UVs per face):
      - fit_resolution: smallest S x S holding the packed islands at 1:1, S % 4 == 0.
      - final_resolution: exact -> fit; pot-up -> smallest power of two >= fit;
        pot-down -> largest power of two <= fit (islands scaled down, lossy).
    Charts and packs at the fit size; the final canvas is packed by bake_uv_plan.

    `density_mesh` / `density_uv` measure the source texel density on another mesh than the one
    being charted. Step 3 passes the mesh it collapsed: `mesh`'s UVs are then projected onto
    triangles that never existed in the source atlas, so their area says nothing about how many
    texels per world unit the source texture actually holds.

    `merge_islands` charts the mesh once per merge level (see get_adaptive_chart_options) and keeps
    the level whose islands fit the smallest canvas at 1:1 density. Merging islands removes chart
    boundary, and boundary is what costs gutter padding; where merging instead distorts the UVs so
    much that the packer needs a bigger canvas, the smaller canvas of a lower level wins on its own.
    """
    if size_mode not in SIZE_MODES:
        raise ValueError(f"Unsupported size_mode '{size_mode}' (expected one of {', '.join(SIZE_MODES)})")
    if unwrap_method not in UNWRAP_METHODS:
        raise ValueError(f"Unsupported unwrap_method '{unwrap_method}' (expected one of {', '.join(UNWRAP_METHODS)})")

    source_uv = _check_source_uv(mesh, source_uv)
    src_w, src_h = source_image.size
    if density_mesh is None:
        density_mesh, density_uv = mesh, source_uv
    else:
        density_uv = _check_source_uv(density_mesh, density_uv)
    t_src = texel_density(density_mesh.vertices, density_mesh.faces, density_uv, src_w, src_h)
    if t_src <= 0.0:
        raise ValueError("Source UVs have zero area: the canvas cannot be sized at 1:1 texel density")

    source_px_area = float(_uv_triangle_areas(density_uv, density_mesh.faces).sum()) * src_w * src_h

    def chart_levels(chart_mesh: trimesh.Trimesh, px_area: float) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        """Charts at every merge level and keeps the level whose islands fit the smallest canvas."""
        attempts: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_level = 0
        for merge_level in (range(ISLAND_MERGE_LEVELS) if merge_islands else range(1)):
            if unwrap_method == "xatlas":
                candidate = _fit_xatlas(chart_mesh, t_src, merge_level)
            else:
                candidate = _fit_uvatlas(chart_mesh, t_src, px_area, uvatlas_gutter, merge_level)
            layout = candidate["layouts"][candidate["fit"]]
            # Boundary and canvas in the same units for every level: texels of that level's own canvas
            attempts.append({
                "level": merge_level,
                "canvas": candidate["fit"],
                "islands": _count_uv_islands(layout["faces"]),
                "boundaryTexels": round(uv_boundary_length(layout["uv"] * candidate["fit"], layout["faces"]), 1)
            })
            if best is None or candidate["fit"] < best["fit"]:
                best, best_level = candidate, merge_level
            elif merge_level > 0:
                # This level already needs a bigger canvas than the best one: merging further only
                # distorts the islands more, so there is nothing to gain from another charting pass
                break
        return best, best_level, attempts

    # Flat-colour faces need no texels of their own: hold them out of the chart pass and give each
    # colour one swatch, so the canvas is sized for the faces that actually carry detail. Charting
    # without them cuts holes into the mesh, which adds chart boundary, so the two canvases are
    # measured against each other and the smaller one wins - as with the island merge levels.
    flat_info: Optional[Dict[str, Any]] = None
    flat_disabled: Optional[str] = None
    chart_vertex_map = None
    fit_info, chosen, attempts = chart_levels(mesh, source_px_area)

    if flat_swatch:
        flat_info = classify_flat_faces(
            mesh,
            source_image=source_image,
            source_uv=source_uv,
            tolerance=flat_tolerance,
            min_group_faces=flat_min_group_faces,
            material=getattr(getattr(mesh, "visual", None), "material", None),
            source_uv_sampler=source_uv_sampler
        )
        detailed = ~flat_info["flat"]
        if not detailed.any():
            flat_disabled = "every face is flat: the chart pass needs at least one detailed face"
        elif flat_info["stats"]["swatches"] == 0:
            flat_disabled = (
                f"no flat group reached {flat_min_group_faces} faces at tolerance {flat_tolerance:g}"
            )
        else:
            sub_mesh, sub_vertex_map = _submesh_for_charting(mesh, detailed)
            sub_px_area = float(
                _uv_triangle_areas(source_uv, np.asarray(mesh.faces, dtype=np.int64)[detailed]).sum()
            ) * src_w * src_h
            sub_fit, sub_chosen, sub_attempts = chart_levels(sub_mesh, sub_px_area)
            strip = _swatch_strip_px(flat_info["stats"]["swatches"], sub_fit["fit"])
            swatched_canvas = _round_up_to_block(sub_fit["fit"] + strip)
            if swatched_canvas >= fit_info["fit"]:
                # Cutting the flat faces out added more chart boundary than the swatches saved
                flat_disabled = (
                    f"canvas would not shrink: {swatched_canvas} with swatches vs {fit_info['fit']} without"
                )
            else:
                fit_info, chosen, attempts = sub_fit, sub_chosen, sub_attempts
                chart_vertex_map = sub_vertex_map
        if flat_disabled is not None:
            flat_info = None

    island_merge = {
        "enabled": bool(merge_islands),
        "method": unwrap_method,
        "chosenLevel": chosen,
        "levels": attempts,
        "islandsBefore": attempts[0]["islands"],
        "islandsAfter": attempts[chosen]["islands"],
        "boundaryTexelsBefore": attempts[0]["boundaryTexels"],
        "boundaryTexelsAfter": attempts[chosen]["boundaryTexels"],
        "boundaryReductionPercent": round(
            (1.0 - attempts[chosen]["boundaryTexels"] / attempts[0]["boundaryTexels"]) * 100.0, 2
        ) if attempts[0]["boundaryTexels"] > 0 else 0.0,
        "canvasBefore": attempts[0]["canvas"],
        "canvasAfter": attempts[chosen]["canvas"]
    }

    island_fit = fit_info["fit"]
    strip_px = _swatch_strip_px(flat_info["stats"]["swatches"], island_fit) if flat_info else 0
    fit = _round_up_to_block(island_fit + strip_px)
    final = {"exact": fit, "pot-up": _pow2_at_least(fit), "pot-down": _pow2_at_most(fit)}[size_mode]
    return {
        "size_mode": size_mode,
        "unwrap_method": unwrap_method,
        "flat": flat_info,
        "flat_disabled": flat_disabled,
        "swatch_strip_px": strip_px,
        "island_fit_resolution": island_fit,
        "chart_vertex_map": chart_vertex_map,
        "uvatlas_gutter": float(uvatlas_gutter),
        "source_resolution": (src_w, src_h),
        "texel_density_source": t_src,
        "fit_resolution": fit,
        "final_resolution": final,
        "fit_passes": fit_info["passes"],
        "island_merge": island_merge,
        # Layouts already packed, by canvas size (reused when the final canvas equals one of them)
        "layouts": fit_info["layouts"],
        # xatlas pixel layout at 1:1 (pot-up re-uses it on the larger canvas)
        "xatlas_natural": fit_info.get("natural"),
    }


def _final_layout(mesh: trimesh.Trimesh, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Layout packed for the plan's final canvas. Returns (layout, repacked_at_final_resolution).
    With flat swatches the islands are packed for the canvas minus the swatch strip, on the mesh
    without its flat faces; the layout's vmapping is mapped back to the original vertices.
    """
    canvas = plan["final_resolution"]
    strip_px = plan.get("swatch_strip_px", 0)
    box = canvas - strip_px
    flat_info = plan.get("flat")
    chart_mesh = mesh if flat_info is None else _submesh_for_charting(mesh, ~flat_info["flat"])[0]
    merge_level = plan["island_merge"]["chosenLevel"]

    def finish(layout: Dict[str, Any], repacked: bool) -> Tuple[Dict[str, Any], bool]:
        layout = _place_in_box(layout, canvas, strip_px)
        vertex_map = plan.get("chart_vertex_map")
        if vertex_map is not None:
            layout = {**layout, "vmapping": np.asarray(vertex_map, dtype=np.int64)[layout["vmapping"]]}
        return layout, repacked

    if box in plan["layouts"]:
        return finish(plan["layouts"][box], False)

    if plan["unwrap_method"] == "uvatlas":
        return finish(_uvatlas_layout(chart_mesh, box, plan["uvatlas_gutter"], merge_level), True)

    natural = plan["xatlas_natural"]
    if box > plan["island_fit_resolution"]:
        # pot-up: the 1:1 packing fits the larger canvas as is (padding already in final pixels);
        # maximize_uv_bounds scales the islands and gutters up to fill it.
        return finish(_place_on_canvas(natural, box), False)

    # pot-down: re-pack at a lower density so the atlas fits the smaller canvas and the padding
    # stays in final-canvas pixels (scaling the 1:1 layout down would shrink the gutters).
    tried = []
    for _ in range(MAX_PACK_PASSES):
        tpu = natural["texels_per_unit"] * (box - 2 * CANVAS_MARGIN_PX) / natural["extent"] * 0.99
        natural = _xatlas_natural_atlas(chart_mesh, tpu, merge_level)
        tried.append((round(tpu, 3), natural["extent"]))
        if natural["extent"] + 2 * CANVAS_MARGIN_PX <= box:
            return finish(_place_on_canvas(natural, box), True)
    raise RuntimeError(
        f"xatlas could not fit the charts into {box}x{box} within {MAX_PACK_PASSES} passes "
        f"(texels/unit, atlas side): {tried}"
    )


def maximize_uv_bounds(
    uv: np.ndarray,
    target_res: int,
    padding_px: int = 4
) -> np.ndarray:
    """
    Bước C: Normalizes and scales UV coordinates uniformly to expand UV islands to the maximum
    extent within [0, 1] x [0, 1] of target_res, with safe margins so charts do not
    touch edges or overlap each other.
    """
    if uv is None or len(uv) == 0:
        return uv
    uv_out = np.array(uv, dtype=np.float64, copy=True)
    margin = float(padding_px) / float(max(target_res, 1))

    u_min, v_min = np.min(uv_out, axis=0)
    u_max, v_max = np.max(uv_out, axis=0)

    span_u = max(u_max - u_min, 1e-7)
    span_v = max(v_max - v_min, 1e-7)

    avail = max(1.0 - 2.0 * margin, 1e-4)
    scale = min(avail / span_u, avail / span_v)

    offset_u = margin + (avail - span_u * scale) / 2.0
    offset_v = margin + (avail - span_v * scale) / 2.0

    uv_out[:, 0] = offset_u + (uv_out[:, 0] - u_min) * scale
    uv_out[:, 1] = offset_v + (uv_out[:, 1] - v_min) * scale
    return uv_out


# Non tangent-space PBR slots: resampled into the new UV layout like the base colour
_RESAMPLED_TEXTURE_SLOTS = ("metallicRoughnessTexture", "occlusionTexture", "emissiveTexture")
_FLAT_NORMAL_RGB = (128, 128, 255)


def _slot_rgb(mat: Any, slot: str) -> Optional[np.ndarray]:
    """(H, W, 3) uint8 pixels of a material texture slot; None if the slot is empty. Fails fast otherwise."""
    img = getattr(mat, slot, None)
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        raise TypeError(f"Material {slot} is a {type(img).__name__}, expected a PIL image")
    try:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as err:
        raise RuntimeError(f"Cannot read the material {slot} image for re-baking: {err}") from err


def _face_uv_derivatives(tri_pos: np.ndarray, tri_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-face T = dp/du, B = dp/dv: the frame three.js derives when a glTF has no TANGENT attribute.
    In trimesh's v-up UV space (glTF V flipped on load) the glTF normal-map convention is
    +X = +dp/du, +Y = +dp/dv. Returns (T, B, valid); valid is False for degenerate UV triangles.
    """
    dp1 = tri_pos[:, 1] - tri_pos[:, 0]
    dp2 = tri_pos[:, 2] - tri_pos[:, 0]
    duv1 = tri_uv[:, 1] - tri_uv[:, 0]
    duv2 = tri_uv[:, 2] - tri_uv[:, 0]
    r = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    # |r| = |duv1| |duv2| sin(angle): relative test, so tiny but well-shaped UV triangles stay valid
    valid = np.abs(r) > 1e-9 * np.linalg.norm(duv1, axis=1) * np.linalg.norm(duv2, axis=1)
    inv_r = (1.0 / np.where(valid, r, 1.0))[:, None]
    t = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * inv_r
    b = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * inv_r
    return t, b, valid


def _tangent_frame(t: np.ndarray, b: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (T, B) around unit n: Gram-Schmidt T, B = cross(n, T) signed like b (keeps mirroring)."""
    t_len0 = np.linalg.norm(t, axis=1)
    t = t - n * np.sum(n * t, axis=1, keepdims=True)
    t_len = np.linalg.norm(t, axis=1)
    valid = t_len > 1e-6 * t_len0
    t = t / np.where(valid, t_len, 1.0)[:, None]
    bt = np.cross(n, t)
    bt *= np.where(np.sum(bt * b, axis=1) < 0.0, -1.0, 1.0)[:, None]
    return t, bt, valid


def _rebake_normal_map(
    normal_rgb: np.ndarray,
    tri_pos: np.ndarray,
    old_tri_uv: np.ndarray,
    new_tri_uv: np.ndarray,
    tri_normals: Optional[np.ndarray],
    fid: np.ndarray,
    bary: np.ndarray,
    src_uv: np.ndarray,
    chunk: int = 1 << 20
) -> Tuple[np.ndarray, int]:
    """
    Re-bakes a tangent-space normal map into the new UV layout. Per baked pixel: sample the old map at
    src_uv, decode (rgb / 255 * 2 - 1), to object space with the OLD per-face frame, back with the NEW
    per-face frame (same face positions, same interpolated vertex normal N), normalize, encode.
    Pixels whose old or new frame is degenerate keep the sampled value.
    tri_* are per new face: positions, old / new UVs, vertex normals (None -> face normal).
    Returns (encoded (P, 3) float in [0, 255], degenerate pixel count).
    """
    t_old, b_old, ok_old = _face_uv_derivatives(tri_pos, old_tri_uv)
    t_new, b_new, ok_new = _face_uv_derivatives(tri_pos, new_tri_uv)
    ok_face = ok_old & ok_new
    if tri_normals is None:
        face_n = np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0])

    encoded = np.empty((len(fid), 3), dtype=np.float64)
    degenerate = 0
    for s in range(0, len(fid), chunk):  # bounded memory on multi-million pixel canvases
        f = fid[s:s + chunk]
        n_ts = _sample_texture_bilinear(normal_rgb, src_uv[s:s + chunk]) / 255.0 * 2.0 - 1.0
        n = face_n[f] if tri_normals is None else np.einsum("pk,pkj->pj", bary[s:s + chunk], tri_normals[f])
        n_len = np.linalg.norm(n, axis=1)
        n = n / np.where(n_len > 0.0, n_len, 1.0)[:, None]
        t0, b0, ok0 = _tangent_frame(t_old[f], b_old[f], n)
        t1, b1, ok1 = _tangent_frame(t_new[f], b_new[f], n)
        n_os = n_ts[:, 0:1] * t0 + n_ts[:, 1:2] * b0 + n_ts[:, 2:3] * n
        out = np.stack([np.sum(n_os * t1, axis=1), np.sum(n_os * b1, axis=1), np.sum(n_os * n, axis=1)], axis=1)
        out_len = np.linalg.norm(out, axis=1)
        ok = ok_face[f] & (n_len > 0.0) & ok0 & ok1 & (out_len > 0.0)
        out = np.where(ok[:, None], out / np.where(ok, out_len, 1.0)[:, None], n_ts)
        encoded[s:s + chunk] = (out * 0.5 + 0.5) * 255.0
        degenerate += int(np.count_nonzero(~ok))
    return encoded, degenerate


def _rebake_material_slots(
    mat: Any,
    mesh: trimesh.Trimesh,
    source_uv: np.ndarray,
    orig_face_verts: np.ndarray,
    new_tri_uv: np.ndarray,
    sel: np.ndarray,
    fid: np.ndarray,
    bary: np.ndarray,
    src_uv: np.ndarray,
    covered: np.ndarray,
    padding: int,
    swatch: Optional[Tuple[np.ndarray, Dict[str, List[List[int]]]]] = None
) -> Tuple[Dict[str, Image.Image], int]:
    """
    Re-bakes every non-base-colour texture slot of the PBRMaterial `mat` into the new UV layout with the base
    colour's rasterization (sel / fid / bary / src_uv), canvas and dilation: the re-charted mesh no longer
    carries the UVs those images were painted for. normalTexture is re-encoded into the new tangent frames
    (flat-normal background); the other slots are resampled like the base colour (mean sampled background).
    Returns ({slot: new image}, degenerate normal-frame pixel count).
    """
    res = covered.shape[0]
    rebaked: Dict[str, Image.Image] = {}
    degenerate = 0
    for slot in ("normalTexture",) + _RESAMPLED_TEXTURE_SLOTS:
        slot_rgb = _slot_rgb(mat, slot)
        if slot_rgb is None:
            continue
        if slot == "normalTexture":
            vn = getattr(mesh, "vertex_normals", None)
            tri_normals = (
                np.asarray(vn, dtype=np.float64)[orig_face_verts]
                if vn is not None and len(vn) == len(mesh.vertices) else None
            )
            values, degenerate = _rebake_normal_map(
                slot_rgb, np.asarray(mesh.vertices, dtype=np.float64)[orig_face_verts],
                source_uv[orig_face_verts], new_tri_uv, tri_normals, fid, bary, src_uv
            )
            values = np.round(values)
            background = np.array(_FLAT_NORMAL_RGB, dtype=np.uint8)
        else:
            values = _check_no_nan(_sample_texture_bilinear(slot_rgb, src_uv), slot)
            background = np.mean(values, axis=0).astype(np.uint8)
        canvas = np.tile(background, (res * res, 1))
        canvas[sel] = np.clip(values, 0.0, 255.0).astype(np.uint8)
        canvas = canvas.reshape(res, res, 3)
        if swatch is not None:
            cells, colors = swatch
            # This slot is flat over every swatched face too, so one colour per swatch holds it
            _paint_swatches(canvas, np.zeros_like(covered), cells, colors[slot])
        # New PIL images: no _fast_save_data / _is_bitstream_passthrough, so the GLB gets these pixels
        rebaked[slot] = Image.fromarray(dilate_texture(canvas, covered, padding=padding), mode="RGB")
    return rebaked, degenerate


def _paint_swatches(image: np.ndarray, covered: np.ndarray, cells: np.ndarray, colors: List[List[int]]) -> None:
    """Fills every swatch cell with its colour and marks it covered, so the dilation leaves it be."""
    channels = image.shape[2]
    for c, (x0, y0, x1, y1) in enumerate(cells):
        rgb = np.asarray(colors[c], dtype=np.uint8)
        image[y0:y1, x0:x1, :3] = rgb
        if channels == 4:
            image[y0:y1, x0:x1, 3] = 255
        covered[y0:y1, x0:x1] = True


def _append_flat_faces(
    mesh: trimesh.Trimesh,
    flat_info: Dict[str, Any],
    layout: Dict[str, Any],
    cells: np.ndarray,
    canvas: int
) -> Dict[str, np.ndarray]:
    """
    Adds the flat faces back to the charted mesh, every vertex pinned to the centre of its
    colour's swatch. Their UV triangles are a point, which is what a constant colour needs: the
    GPU samples that one texel whatever the mip level, so nothing from a neighbouring island can
    bleed in. A vertex shared by two colours is duplicated, one copy per colour.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)[flat_info["flat"]]
    cluster = flat_info["cluster"][flat_info["flat"]]
    n_clusters = len(cells)
    uv_centre = _swatch_uv(cells, canvas)

    key = faces * n_clusters + cluster[:, None]      # one vertex per (original vertex, colour)
    unique_key, inverse = np.unique(key, return_inverse=True)
    vertex_source = unique_key // n_clusters
    vertex_cluster = unique_key % n_clusters

    offset = len(layout["vertices"])
    return {
        "vertices": np.vstack([layout["vertices"], np.asarray(mesh.vertices, dtype=np.float64)[vertex_source]]),
        "faces": np.vstack([layout["faces"], inverse.reshape(-1, 3) + offset]),
        "uv": np.vstack([layout["uv"], uv_centre[vertex_cluster]]),
        "vmapping": np.concatenate([np.asarray(layout["vmapping"], dtype=np.int64), vertex_source]),
    }



def bake_uv_plan(
    mesh: trimesh.Trimesh,
    plan: Dict[str, Any],
    source_image: Image.Image,
    source_uv: np.ndarray,
    dilation_padding: int = 16,
    double_sided: bool = False,
    source_uv_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[trimesh.Trimesh, Image.Image, Dict[str, Any]]:
    """
    Packs the charts for the plan's final canvas (see plan_uv_canvas) and bakes the source texture
    into it: barycentric sampling, EDT dilation into the gutters, FrontSide material
    (doubleSided=double_sided). Returns (recharted_mesh, baked_image, stats).

    `source_uv_sampler` reads the source texture's UV at arbitrary 3D points instead of
    interpolating `source_uv` over the triangle being baked. Step 3 passes one when it collapsed
    edges: `mesh` is then a mesh whose triangles never existed in the textured original, so each
    texel is looked up on the original surface rather than between this triangle's corners.
    """
    source_uv = _check_source_uv(mesh, source_uv)
    # The baked material is the source glTF PBR material with the new textures: nothing is invented
    orig_mat = getattr(getattr(mesh, "visual", None), "material", None)
    if not isinstance(orig_mat, trimesh.visual.material.PBRMaterial):
        raise PipelineAbort(
            f"The source material is {type(orig_mat).__name__}, not a glTF PBR material: "
            f"its factors cannot be carried over to the baked texture"
        )
    layout, repacked = _final_layout(mesh, plan)
    target_res = layout["canvas"]
    vertices_recharted = layout["vertices"]
    faces_recharted = layout["faces"]
    uv_recharted = layout["uv"]
    vmapping = layout["vmapping"]

    t_src = plan["texel_density_source"]
    t_new = _layout_texel_density(layout)
    density_ratio = t_new / t_src
    if plan["size_mode"] in ("exact", "pot-up") and density_ratio < MIN_TEXEL_DENSITY_RATIO:
        raise RuntimeError(
            f"{plan['size_mode']} canvas {target_res}x{target_res} holds the islands at only "
            f"{density_ratio:.3f} of the source texel density (minimum {MIN_TEXEL_DENSITY_RATIO})"
        )

    has_alpha = source_image.mode in ("RGBA", "LA") or (
        source_image.mode == "P" and "transparency" in source_image.info
    )
    has_transparency = False
    if has_alpha:
        if source_image.mode in ("RGBA", "LA"):
            alpha_arr = np.asarray(source_image.convert("RGBA"))[..., 3]
            if np.any(alpha_arr < 255):
                has_transparency = True
        else:
            has_transparency = True

    if has_transparency:
        src_img = np.asarray(source_image.convert("RGBA"), dtype=np.uint8)
        channels = 4
        out_mode = "RGBA"
    else:
        src_img = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        channels = 3
        out_mode = "RGB"

    # Bake the texture & dilate the gutters
    sel, fid, bary = _rasterize_uv_atlas(faces_recharted, uv_recharted, target_res)
    if len(sel) == 0:
        raise RuntimeError("Failed to rasterize UV atlas during baking.")

    # Map recharted faces through vmapping back to exact original mesh vertices
    orig_face_verts = np.asarray(vmapping, dtype=np.int64)[faces_recharted]  # (N, 3)
    if source_uv_sampler is None:
        raw_tri_uv = source_uv[orig_face_verts[fid]]  # (len(sel), 3, 2)
        src_uv = (raw_tri_uv * bary[:, :, None]).sum(axis=1)
    else:
        # Where each baked texel sits on the mesh, looked up on the textured original surface
        tri_pos = np.asarray(mesh.vertices, dtype=np.float64)[orig_face_verts[fid]]
        src_uv = source_uv_sampler((tri_pos * bary[:, :, None]).sum(axis=1))
    src_uv = np.clip(src_uv, 0.0, 1.0)

    colors = _check_no_nan(_sample_texture_bilinear(src_img, src_uv), "base colour texture")

    # Defensive canvas background initialization: initialize the canvas with the average
    # sampled surface color rather than stark pure white (255, 255, 255) or black (0, 0, 0).
    # This guarantees that unmapped canvas texels match the actual model surface tones,
    # eliminating white seam/crack artifacts when GPU bilinear filtering/mipmapping samples boundaries.
    if len(colors) > 0:
        # True average color of the model surface (uncontaminated by blank white canvas in source texture)
        avg_surface_color = np.mean(colors, axis=0).astype(np.uint8)
        if channels == 4 and not has_transparency:
            avg_surface_color[3] = 255
    else:
        avg_surface_color = np.mean(src_img[..., :channels], axis=(0, 1)).astype(np.uint8)

    base_flat = np.tile(avg_surface_color, (target_res * target_res, 1))
    base_flat[sel] = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    base_img = base_flat.reshape(target_res, target_res, channels)

    covered = np.zeros(target_res * target_res, dtype=bool)
    covered[sel] = True
    covered = covered.reshape(target_res, target_res)

    # Flat faces: one swatch per colour, painted and marked covered so the dilation keeps it intact
    flat_info = plan.get("flat")
    swatch: Optional[Tuple[np.ndarray, Dict[str, List[List[int]]]]] = None
    if flat_info is not None:
        cells = swatch_cells(flat_info["stats"]["swatches"], target_res, plan["swatch_strip_px"])
        swatch = (cells, flat_info["cluster_colors"])
        _paint_swatches(base_img, covered, cells, flat_info["cluster_colors"]["baseColorTexture"])

    # Mandatory EDT boundary dilation: expand edge pixels into the gutter buffer so GPU bilinear filtering
    # and mipmapping sample valid colors instead of the background canvas.
    # For UVAtlas, charts have gutter >= 4.0 px, so we mandatorily apply at least padding=8
    # (or user-specified dilation_padding if larger).
    min_dilation = 8 if plan["unwrap_method"] == "uvatlas" else 4
    eff_dilation_padding = max(min_dilation, int(dilation_padding))
    dilated_img = dilate_texture(base_img, covered, padding=eff_dilation_padding)
    dilated_pil = Image.fromarray(dilated_img, mode=out_mode)

    # Configure PBRMaterial with FrontSide rendering (doubleSided=double_sided, default False)
    # normal / metallicRoughness / occlusion / emissive maps re-baked into the NEW layout as well
    rebaked_textures, normal_degenerate_px = _rebake_material_slots(
        orig_mat, mesh, source_uv, orig_face_verts, uv_recharted[faces_recharted],
        sel, fid, bary, src_uv, covered, eff_dilation_padding, swatch=swatch
    )
    mat = orig_mat.copy()
    mat.baseColorTexture = dilated_pil
    mat.doubleSided = double_sided
    for slot, img in rebaked_textures.items():
        setattr(mat, slot, img)

    if flat_info is not None:
        combined = _append_flat_faces(mesh, flat_info, layout, swatch[0], target_res)
        vertices_recharted = combined["vertices"]
        faces_recharted = combined["faces"]
        uv_recharted = combined["uv"]
        vmapping = combined["vmapping"]

    # Preserve smooth vertex normals
    normals_recharted = None
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        normals_recharted = np.asarray(mesh.vertex_normals, dtype=np.float64)[np.asarray(vmapping, dtype=np.int64)]

    if normals_recharted is not None:
        recharted_mesh = trimesh.Trimesh(
            vertices=vertices_recharted,
            faces=faces_recharted,
            vertex_normals=normals_recharted,
            process=False
        )
    else:
        recharted_mesh = trimesh.Trimesh(
            vertices=vertices_recharted,
            faces=faces_recharted,
            process=False
        )
        _ = recharted_mesh.vertex_normals

    recharted_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_recharted, material=mat)

    # Step 3 export format: PNG LOSSLESS to eliminate compression generational loss
    dilated_pil = optimize_mesh_texture_for_export(recharted_mesh, preferred_format="PNG")

    # Comprehensive metrics recording
    mesh_area = float(mesh.area)
    covered_pixels = int(len(sel))
    total_pixels = target_res * target_res
    coverage_percent = round(float(covered_pixels / total_pixels * 100.0), 2)
    texel_density_linear = round(float(np.sqrt(covered_pixels) / np.sqrt(max(mesh_area, 1e-6))), 2)
    texel_density_area = round(float(covered_pixels / max(mesh_area, 1e-6)), 2)

    result_stats = {
        "size_mode": plan["size_mode"],
        "unwrap_method": plan["unwrap_method"],
        "fit_resolution": plan["fit_resolution"],
        "final_resolution": target_res,
        "fit_passes": plan["fit_passes"],
        "repacked_at_final_resolution": repacked,
        "texel_density_source": round(float(t_src), 4),
        "texel_density_final": round(float(t_new), 4),
        "texel_density_ratio": round(float(density_ratio), 4),
        "textureFormat": "PNG",
        "uvCoverageRatio": round(float(covered_pixels / total_pixels), 4),
        "target_resolution": target_res,
        "canvas_pixels": total_pixels,
        "covered_pixels": covered_pixels,
        "uv_coverage_ratio_percent": coverage_percent,
        "texel_density_linear": texel_density_linear,
        "texel_density_area": texel_density_area,
        "mesh_surface_area": round(mesh_area, 4),
        "dilation_padding": eff_dilation_padding,
        "flat_swatch": (
            {**flat_info["stats"], "stripPx": plan["swatch_strip_px"],
             "islandCanvas": target_res - plan["swatch_strip_px"]}
            if flat_info is not None else None
        ),
        "rebaked_texture_slots": ["baseColorTexture"] + list(rebaked_textures),
        "normal_rebake_degenerate_pixels": normal_degenerate_px,
        "double_sided": double_sided
    }
    result_stats.update(layout["meta"])

    if hasattr(recharted_mesh, "metadata"):
        recharted_mesh.metadata["uv_metrics"] = result_stats
    return recharted_mesh, dilated_pil, result_stats


def rechart_and_bake_high_density(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    size_mode: str = "exact",
    unwrap_method: str = "xatlas",
    dilation_padding: int = 16,
    double_sided: bool = False,
    uvatlas_gutter: float = 4.0,
    stats: Optional[Dict[str, Any]] = None,
    return_stats: bool = False
) -> Tuple[trimesh.Trimesh, Image.Image] | Tuple[trimesh.Trimesh, Image.Image, Dict[str, Any]]:
    """
    Re-charts the UVs (xatlas or Microsoft UVAtlas) on a canvas sized by `size_mode` at the source's
    1:1 texel density (plan_uv_canvas) and bakes the source texture into it (bake_uv_plan).
    Always re-charts: the pipeline's keep-original rule is applied by the caller.
    """
    plan = plan_uv_canvas(
        mesh,
        source_image,
        source_uv,
        size_mode=size_mode,
        unwrap_method=unwrap_method,
        uvatlas_gutter=uvatlas_gutter
    )
    recharted_mesh, dilated_pil, result_stats = bake_uv_plan(
        mesh,
        plan,
        source_image,
        source_uv,
        dilation_padding=dilation_padding,
        double_sided=double_sided
    )
    if stats is not None:
        stats.update(result_stats)
    if return_stats:
        return recharted_mesh, dilated_pil, result_stats
    return recharted_mesh, dilated_pil
