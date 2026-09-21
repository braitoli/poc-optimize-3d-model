"""
uv_baker.py

UV Atlas Re-charting, Direct Barycentric Baking, and 16px Boundary Dilation.
Maximizes texture canvas utilization, Texel Density, and surface sharpness.
Ensures zero black edge bleeding during GPU texture mipmapping.
"""

import math
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
from scipy import ndimage
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


def get_adaptive_chart_options(n_faces: int) -> xatlas.ChartOptions:
    """
    Configures xatlas.ChartOptions with adaptive iterations and micro-chart merging
    to achieve ultra-fast unwrapping even on large meshes (>200k faces like Gravilux).
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


def _xatlas_natural_atlas(mesh: trimesh.Trimesh, texels_per_unit: float) -> Dict[str, Any]:
    """
    Charts and packs the mesh with xatlas at `texels_per_unit` and resolution 0, i.e. one atlas
    whose size xatlas grows until every chart fits. Returns the layout in atlas pixels.
    """
    n_faces = len(mesh.faces)
    c_opts = get_adaptive_chart_options(n_faces)
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


def _uvatlas_layout(mesh: trimesh.Trimesh, canvas: int, gutter: float) -> Dict[str, Any]:
    """Microsoft UVAtlas unwrap packed into a canvas x canvas square (gutter in canvas pixels)."""
    vertices, faces, uv, vmapping, meta = unwrap_mesh_uvatlas(
        mesh=mesh,
        target_res=canvas,
        gutter=max(4.0, float(gutter))
    )
    uv = maximize_uv_bounds(np.asarray(uv, dtype=np.float64), target_res=canvas, padding_px=CANVAS_MARGIN_PX)
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "vmapping": np.asarray(vmapping, dtype=np.int64),
        "uv": uv,
        "canvas": canvas,
        "meta": dict(meta),
    }


def _fit_xatlas(mesh: trimesh.Trimesh, t_src: float) -> Dict[str, Any]:
    """
    Exact fit with xatlas: pack at texels_per_unit = T_src (xatlas scales every chart to that
    density) into a single atlas sized to fit, then S = max(atlas side) + border margins, rounded
    up to the block size. xatlas can land slightly under the requested density; if T_new / T_src
    falls below the minimum, re-pack at a proportionally higher density (bounded).
    """
    tpu = t_src
    tried = []
    for _ in range(MAX_PACK_PASSES):
        natural = _xatlas_natural_atlas(mesh, tpu)
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


def _fit_uvatlas(mesh: trimesh.Trimesh, t_src: float, source_px_area: float, gutter: float) -> Dict[str, Any]:
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
        layout = _uvatlas_layout(mesh, canvas, gutter)
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


def plan_uv_canvas(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    size_mode: str = "exact",
    unwrap_method: str = "xatlas",
    uvatlas_gutter: float = 4.0
) -> Dict[str, Any]:
    """
    Sizes the square re-chart canvas at the source's 1:1 average texel density (see
    texel_density; source texture W x H, source UVs per face):
      - fit_resolution: smallest S x S holding the packed islands at 1:1, S % 4 == 0.
      - final_resolution: exact -> fit; pot-up -> smallest power of two >= fit;
        pot-down -> largest power of two <= fit (islands scaled down, lossy).
    Charts and packs at the fit size; the final canvas is packed by bake_uv_plan.
    """
    if size_mode not in SIZE_MODES:
        raise ValueError(f"Unsupported size_mode '{size_mode}' (expected one of {', '.join(SIZE_MODES)})")
    if unwrap_method not in UNWRAP_METHODS:
        raise ValueError(f"Unsupported unwrap_method '{unwrap_method}' (expected one of {', '.join(UNWRAP_METHODS)})")

    source_uv = _check_source_uv(mesh, source_uv)
    src_w, src_h = source_image.size
    t_src = texel_density(mesh.vertices, mesh.faces, source_uv, src_w, src_h)
    if t_src <= 0.0:
        raise ValueError("Source UVs have zero area: the canvas cannot be sized at 1:1 texel density")

    if unwrap_method == "xatlas":
        fit_info = _fit_xatlas(mesh, t_src)
    else:
        source_px_area = float(_uv_triangle_areas(source_uv, mesh.faces).sum()) * src_w * src_h
        fit_info = _fit_uvatlas(mesh, t_src, source_px_area, uvatlas_gutter)

    fit = fit_info["fit"]
    final = {"exact": fit, "pot-up": _pow2_at_least(fit), "pot-down": _pow2_at_most(fit)}[size_mode]
    return {
        "size_mode": size_mode,
        "unwrap_method": unwrap_method,
        "uvatlas_gutter": float(uvatlas_gutter),
        "source_resolution": (src_w, src_h),
        "texel_density_source": t_src,
        "fit_resolution": fit,
        "final_resolution": final,
        "fit_passes": fit_info["passes"],
        # Layouts already packed, by canvas size (reused when the final canvas equals one of them)
        "layouts": fit_info["layouts"],
        # xatlas pixel layout at 1:1 (pot-up re-uses it on the larger canvas)
        "xatlas_natural": fit_info.get("natural"),
    }


def _final_layout(mesh: trimesh.Trimesh, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Layout packed for the plan's final canvas. Returns (layout, repacked_at_final_resolution)."""
    canvas = plan["final_resolution"]
    if canvas in plan["layouts"]:
        return plan["layouts"][canvas], False

    if plan["unwrap_method"] == "uvatlas":
        return _uvatlas_layout(mesh, canvas, plan["uvatlas_gutter"]), True

    natural = plan["xatlas_natural"]
    if canvas > plan["fit_resolution"]:
        # pot-up: the 1:1 packing fits the larger canvas as is (padding already in final pixels);
        # maximize_uv_bounds scales the islands and gutters up to fill it.
        return _place_on_canvas(natural, canvas), False

    # pot-down: re-pack at a lower density so the atlas fits the smaller canvas and the padding
    # stays in final-canvas pixels (scaling the 1:1 layout down would shrink the gutters).
    tried = []
    for _ in range(MAX_PACK_PASSES):
        tpu = natural["texels_per_unit"] * (canvas - 2 * CANVAS_MARGIN_PX) / natural["extent"] * 0.99
        natural = _xatlas_natural_atlas(mesh, tpu)
        tried.append((round(tpu, 3), natural["extent"]))
        if natural["extent"] + 2 * CANVAS_MARGIN_PX <= canvas:
            return _place_on_canvas(natural, canvas), True
    raise RuntimeError(
        f"xatlas could not fit the charts into {canvas}x{canvas} within {MAX_PACK_PASSES} passes "
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
    padding: int
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
        # New PIL images: no _fast_save_data / _is_bitstream_passthrough, so the GLB gets these pixels
        rebaked[slot] = Image.fromarray(dilate_texture(canvas.reshape(res, res, 3), covered, padding=padding), mode="RGB")
    return rebaked, degenerate


def bake_uv_plan(
    mesh: trimesh.Trimesh,
    plan: Dict[str, Any],
    source_image: Image.Image,
    source_uv: np.ndarray,
    dilation_padding: int = 16,
    double_sided: bool = False
) -> Tuple[trimesh.Trimesh, Image.Image, Dict[str, Any]]:
    """
    Packs the charts for the plan's final canvas (see plan_uv_canvas) and bakes the source texture
    into it: barycentric sampling, EDT dilation into the gutters, FrontSide material
    (doubleSided=double_sided). Returns (recharted_mesh, baked_image, stats).
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
    raw_tri_uv = source_uv[orig_face_verts[fid]]  # (len(sel), 3, 2)
    src_uv = (raw_tri_uv * bary[:, :, None]).sum(axis=1)
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
        sel, fid, bary, src_uv, covered, eff_dilation_padding
    )
    mat = orig_mat.copy()
    mat.baseColorTexture = dilated_pil
    mat.doubleSided = double_sided
    for slot, img in rebaked_textures.items():
        setattr(mat, slot, img)

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
