"""
uv_baker.py

UV Atlas Re-charting, Direct Barycentric Baking, and 16px Boundary Dilation.
Maximizes texture canvas utilization, Texel Density, and surface sharpness.
Ensures zero black edge bleeding during GPU texture mipmapping.
"""

import math
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh
import xatlas
from optimizer.core.uvatlas import unwrap_mesh_uvatlas, is_uvatlas_available
from optimizer.core.texture_utils import (
    clamp_target_resolution,
    optimize_mesh_texture_for_export,
    can_downscale_texture,
    maximize_uv_space,
    apply_bitstream_passthrough
)


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


def compute_uv_metrics(
    mesh: trimesh.Trimesh,
    target_res: int = 1024,
    uv: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Measures UV Coverage Ratio (%) and Texel Density (px/unit) for any mesh and UV layout.
    """
    if uv is None:
        uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) == 0:
        return {
            "uv_coverage_ratio_percent": 0.0,
            "texel_density_linear": 0.0,
            "texel_density_area": 0.0,
            "covered_pixels": 0,
            "canvas_pixels": target_res * target_res,
            "mesh_surface_area": round(float(mesh.area), 4)
        }

    sel, _, _ = _rasterize_uv_atlas(mesh.faces, uv, target_res)
    mesh_area = float(mesh.area)
    covered_pixels = int(len(sel))
    total_pixels = target_res * target_res

    return {
        "target_resolution": target_res,
        "canvas_pixels": total_pixels,
        "covered_pixels": covered_pixels,
        "uv_coverage_ratio_percent": round(float(covered_pixels / total_pixels * 100.0), 2),
        "texel_density_linear": round(float(np.sqrt(covered_pixels) / np.sqrt(max(mesh_area, 1e-6))), 2),
        "texel_density_area": round(float(covered_pixels / max(mesh_area, 1e-6)), 2),
        "mesh_surface_area": round(mesh_area, 4)
    }


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


def compute_original_island_pixels(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    uv: Optional[np.ndarray] = None,
    sample_dim: int = 1024
) -> Tuple[float, float]:
    """
    Calculates the actual pixel area occupied by all UV islands on the original texture:
      1. Rasterizes the original UV triangles onto a sample_dim x sample_dim grid.
      2. Computes coverage ratio = len(covered_pixels) / (sample_dim * sample_dim).
      3. Calculates orig_island_pixels = coverage_ratio * (orig_width * orig_height).
    Returns:
      (orig_island_pixels, coverage_ratio)
    """
    w, h = source_image.size
    total_pixels = float(w * h)
    if uv is None:
        uv = getattr(mesh.visual, "uv", None)

    if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
        return total_pixels, 1.0

    uv_clean = np.clip(uv, 0.0, 1.0)
    grid_dim = min(sample_dim, max(w, h))
    grid_dim = max(256, grid_dim)

    sel, _, _ = _rasterize_uv_atlas(mesh.faces, uv_clean, dim=grid_dim)
    if len(sel) == 0:
        return total_pixels, 1.0

    coverage_ratio = float(len(sel)) / float(grid_dim * grid_dim)
    orig_island_pixels = coverage_ratio * total_pixels
    return float(orig_island_pixels), float(coverage_ratio)


def select_repack_canvas_resolution(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: Optional[np.ndarray] = None,
    requested_res: Union[int, str] = "auto",
    sample_dim: int = 1024
) -> Tuple[int, Dict[str, Any]]:
    """
    Capacity Check 1:1 UV Island Packing:
    Finds the smallest canvas size S in {1024, 2048, 4096} such that UV islands
    fit into S x S while preserving their 1:1 texel scale (no island shrinking).

    - S = 1024 if orig_island_pixels <= 1_048_576 (~1M pixels, e.g. scans with 80% wasted canvas)
    - S = 2048 if 1_048_576 < orig_island_pixels <= 4_350_000 (~4.2M - 4.3M pixels, e.g. Flamibo)
    - S = 4096 if orig_island_pixels > 4_350_000

    Strictly adheres to NO-UPSCALE policy: S <= max_pot(max(w, h)).
    """
    w, h = source_image.size
    orig_max = max(w, h)
    orig_island_pixels, coverage_ratio = compute_original_island_pixels(
        mesh, source_image, uv=source_uv, sample_dim=sample_dim
    )

    max_pot = 1 << int(math.floor(math.log2(orig_max)))
    max_pot = max(256, max_pot)

    is_auto = isinstance(requested_res, str) and requested_res.lower() == "auto"

    if not is_auto:
        try:
            req_int = int(requested_res)
            selected_res = min(req_int, max_pot)
            reason = f"manual_override ({req_int})"
        except (ValueError, TypeError):
            is_auto = True

    if is_auto:
        if orig_island_pixels <= 1_048_576:
            candidate_res = 1024
            reason = f"capacity_1024 (orig_islands={orig_island_pixels:,.0f} <= 1,048,576)"
        elif orig_island_pixels <= 4_350_000:
            candidate_res = 2048
            reason = f"capacity_2048 (orig_islands={orig_island_pixels:,.0f} fits in 2048x2048 1:1)"
        else:
            candidate_res = 4096
            reason = f"capacity_4096 (orig_islands={orig_island_pixels:,.0f} > 4.35M)"

        selected_res = min(candidate_res, max_pot)
        if selected_res < candidate_res:
            reason += f" (clamped to max_pot={selected_res} by NO-UPSCALE policy)"

    details = {
        "orig_size": (w, h),
        "orig_max": orig_max,
        "orig_island_pixels": round(orig_island_pixels, 1),
        "coverage_ratio": round(coverage_ratio, 4),
        "selected_resolution": selected_res,
        "capacity_reason": reason,
        "is_auto": is_auto
    }
    return selected_res, details


def can_downscale_texture(
    mesh: trimesh.Trimesh,
    current_res: int,
    old_uv: Optional[np.ndarray],
    new_uv: np.ndarray,
    new_faces: Optional[np.ndarray] = None,
    min_res: int = 1024,
    td_threshold_ratio: float = 0.85,
    sample_dim: int = 256,
    orig_island_pixels: Optional[float] = None
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Bước B: Checks if texture resolution can be safely downscaled to the next power-of-two tier
    (e.g., 4096 -> 2048, or 2048 -> 1024) without visual loss, by comparing Texel Density
    AND strictly enforcing 1:1 island capacity check:
      TD = effective_uv_area * Res^2 / surface_area_3d

    Downscales only if:
    1. current_res in (4096, 2048) and current_res > min_res
    2. Capacity Check: downscaling to 1024 is permitted ONLY if orig_island_pixels <= 1,048,576 (~1M)
       and downscaling to 2048 is permitted if orig_island_pixels <= 4,350,000 (~4.3M)
    3. TD_new_downscaled >= td_threshold_ratio * TD_orig (e.g., >= 0.85 * TD_orig)

    Returns:
      (can_downscale, target_res, details)
    """
    mesh_area = float(max(mesh.area, 1e-6))

    # 1. Measure effective UV area for old UV
    if old_uv is not None and len(old_uv) > 0 and len(mesh.faces) > 0:
        sel_old, _, _ = _rasterize_uv_atlas(mesh.faces, old_uv, dim=sample_dim)
        effective_uv_area_old = float(len(sel_old)) / float(sample_dim * sample_dim)
    else:
        effective_uv_area_old = 0.5

    effective_uv_area_old = max(effective_uv_area_old, 1e-4)

    # 2. Measure effective UV area for new packed UV
    faces_to_check = new_faces if new_faces is not None else mesh.faces
    if new_uv is not None and len(new_uv) > 0 and len(faces_to_check) > 0:
        sel_new, _, _ = _rasterize_uv_atlas(faces_to_check, new_uv, dim=sample_dim)
        effective_uv_area_new = float(len(sel_new)) / float(sample_dim * sample_dim)
    else:
        effective_uv_area_new = 0.75

    effective_uv_area_new = max(effective_uv_area_new, 1e-4)

    # 3. Calculate original Texel Density
    # TD = effective_uv_area * Res^2 / surface_area_3d
    td_orig = (effective_uv_area_old * (current_res ** 2)) / mesh_area

    if orig_island_pixels is None:
        orig_island_pixels = effective_uv_area_old * (current_res ** 2)

    # 4. Determine next lower power-of-two tier
    if current_res >= 4096:
        downscaled_res = 2048
    elif current_res >= 2048:
        downscaled_res = 1024
    else:
        downscaled_res = current_res

    can_downscale = False
    capacity_passed = True
    if current_res > min_res and downscaled_res < current_res and downscaled_res >= min_res:
        # Enforce 1:1 capacity check
        if downscaled_res == 1024 and orig_island_pixels > 1_048_576:
            capacity_passed = False
        elif downscaled_res == 2048 and orig_island_pixels > 4_350_000:
            capacity_passed = False

        td_new_downscaled = (effective_uv_area_new * (downscaled_res ** 2)) / mesh_area
        if capacity_passed and td_new_downscaled >= td_threshold_ratio * td_orig:
            can_downscale = True
            target_res = downscaled_res
            td_final = td_new_downscaled
        else:
            can_downscale = False
            target_res = current_res
            td_final = (effective_uv_area_new * (target_res ** 2)) / mesh_area
    else:
        can_downscale = False
        target_res = current_res
        td_final = (effective_uv_area_new * (target_res ** 2)) / mesh_area

    td_delta = td_final - td_orig
    td_delta_pct = (td_delta / td_orig * 100.0) if td_orig > 0 else 0.0

    details = {
        "downscaled": can_downscale,
        "originalResolution": f"{current_res}x{current_res}",
        "finalResolution": f"{target_res}x{target_res}",
        "original_resolution": current_res,
        "final_resolution": target_res,
        "orig_island_pixels": round(float(orig_island_pixels), 1),
        "capacity_passed": capacity_passed,
        "uvCoverageRatio": round(effective_uv_area_new, 4),
        "uvCoverageRatioOrig": round(effective_uv_area_old, 4),
        "texelDensityOrig": round(td_orig, 2),
        "texelDensityFinal": round(td_final, 2),
        "texelDensityDelta": round(td_delta, 2),
        "texelDensityDeltaPercent": round(td_delta_pct, 2),
        "td_threshold_ratio": td_threshold_ratio,
        "mesh_surface_area": round(mesh_area, 4)
    }
    return can_downscale, target_res, details


def determine_safe_downscale_resolution(
    mesh: trimesh.Trimesh,
    orig_size: Tuple[int, int],
    uv: Optional[np.ndarray] = None,
    min_texel_density: float = 120.0,
    requested_res: Union[int, str] = "auto",
    sample_dim: int = 256,
    preserve_original: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Direct Pure Downscale Strategy:
    Analyzes original texture resolution and determines the target resolution.
    - If preserve_original is True and requested_res == 'auto':
      Preserves 100% original texture resolution (True Zero-Loss Direct Mode).
    - Otherwise (auto-adaptive downscale):
      4096 -> 2048, 1536/2048 -> 1024.
    - If requested_res is numeric:
      Clamps strictly <= orig_max (no upscaling).

    Guarantees 100% original UV coordinates are preserved (no chart tearing, rotation, or distortion).
    """
    w, h = orig_size
    orig_max = max(w, h)
    mesh_area = float(max(mesh.area, 1e-6))

    if uv is None:
        uv = getattr(mesh.visual, "uv", None)

    # Compute effective UV coverage on original UV coordinates
    if uv is not None and len(uv) > 0 and len(mesh.faces) > 0:
        uv_norm = uv % 1.0 if (np.any(uv < 0.0) or np.any(uv > 1.0)) else uv
        sel, _, _ = _rasterize_uv_atlas(mesh.faces, uv_norm, dim=sample_dim)
        effective_uv_area = max(float(len(sel)) / float(sample_dim * sample_dim), 0.1)
    else:
        effective_uv_area = 0.55

    # Linear Texel Densities at candidate targets (px / 3D unit)
    td_1024 = (1024.0 * np.sqrt(effective_uv_area)) / np.sqrt(mesh_area)
    td_2048 = (2048.0 * np.sqrt(effective_uv_area)) / np.sqrt(mesh_area)

    is_auto = isinstance(requested_res, str) and requested_res.lower() == "auto"

    if is_auto:
        if preserve_original:
            # True Zero-Loss: Preserve 100% original resolution bit-for-bit
            target_res = orig_max
        elif orig_max >= 4096:
            # 4K textures: Safely downscale 1 POT tier from 4096 to 2048 (2K)
            target_res = 2048
        elif orig_max > 1024:
            # 1536 or 2048: drop to 1024 (1K)
            target_res = 1024
        else:
            # <= 1024: largest POT <= orig_max
            target_res = 1 << int(math.floor(math.log2(orig_max)))
            target_res = max(256, target_res)
    else:
        target_res = clamp_target_resolution(requested_res, orig_size)

    td_final = (float(target_res) * np.sqrt(effective_uv_area)) / np.sqrt(mesh_area)
    td_orig = (float(orig_max) * np.sqrt(effective_uv_area)) / np.sqrt(mesh_area)

    details = {
        "downscaled": target_res < orig_max,
        "originalResolution": f"{orig_max}x{orig_max}",
        "finalResolution": f"{target_res}x{target_res}",
        "original_resolution": orig_max,
        "final_resolution": target_res,
        "mesh_surface_area": round(mesh_area, 4),
        "uvCoverageRatio": round(effective_uv_area, 4),
        "texelDensityOrig": round(td_orig, 2),
        "texelDensityFinal": round(td_final, 2),
        "texelDensity1024": round(td_1024, 2),
        "texelDensity2048": round(td_2048, 2),
        "texelDensityDelta": round(td_final - td_orig, 2),
        "min_texel_density_threshold": min_texel_density,
        "uvPreserved100Percent": True,
        "decision": f"{orig_max} -> {target_res} (TD={td_final:.1f} px/u)"
    }
    return target_res, details


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


def rechart_and_bake_high_density(
    mesh: trimesh.Trimesh,
    target_res: int = 1024,
    source_image: Optional[Image.Image] = None,
    source_uv: Optional[np.ndarray] = None,
    dilation_padding: int = 16,
    chart_options: Optional[Dict[str, Any]] = None,
    pack_options: Optional[Dict[str, Any]] = None,
    double_sided: bool = False,
    stats: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
    min_downscale_res: int = 1024,
    td_threshold_ratio: float = 0.85,
    unwrap_method: str = "xatlas",
    uvatlas_gutter: float = 4.0
) -> Tuple[trimesh.Trimesh, Image.Image] | Tuple[trimesh.Trimesh, Image.Image, Dict[str, Any]]:
    """
    4-Stage Adaptive UV & Resolution Optimization Pipeline:
    - Bước A: Tổ chức lại UV gom vào hình vuông [0, 1] x [0, 1] qua xatlas hoặc Microsoft UVAtlas.
    - Bước B: can_downscale_texture kiểm tra Texel Density hạ bậc độ phân giải (4096->2048, 2048->1024).
    - Bước C: maximize_uv_bounds hiệu chỉnh tọa độ UV nở rộng tối đa không gian canvas an toàn.
    - Bước D: Barycentric sampling bake texture, 16px EDT dilation & FrontSide rendering (doubleSided=False).
    """
    # 1. Resolve source image and source UVs
    if source_image is None:
        orig_mat = getattr(mesh.visual, "material", None)
        if orig_mat is not None:
            source_image = getattr(orig_mat, "baseColorTexture", None) or getattr(orig_mat, "image", None)
        if source_image is None:
            source_image = Image.new("RGB", (target_res, target_res), (200, 200, 200))

    # Defense-in-depth: Never upscale texture
    target_res = clamp_target_resolution(target_res, source_image.size)
    initial_res = target_res

    orig_island_pixels, orig_coverage_ratio = compute_original_island_pixels(
        mesh, source_image, uv=source_uv, sample_dim=1024
    )

    if source_uv is None:
        source_uv = getattr(mesh.visual, "uv", None)
        if source_uv is None or len(source_uv) == 0:
            source_uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

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

    # =========================================================================
    # BƯỚC A: Tổ chức lại UV gom vào hình vuông [0, 1] x [0, 1]
    # Padding (xatlas) / gutter (UVAtlas) are pixels of the pack canvas `res`,
    # so Bước A is re-run at the final resolution if Bước B downscales.
    # =========================================================================
    def _unwrap(res: int):
        unwrap_meta: Dict[str, Any] = {}
        use_xatlas = (unwrap_method != "uvatlas")
        if unwrap_method == "uvatlas":
            try:
                eff_uvatlas_gutter = max(4.0, float(uvatlas_gutter))
                vertices_recharted, faces_recharted, uv_recharted, vmapping, unwrap_meta = unwrap_mesh_uvatlas(
                    mesh=mesh,
                    target_res=res,
                    gutter=eff_uvatlas_gutter
                )
            except Exception as uvatlas_err:
                import logging
                logging.getLogger("uv_baker").warning(
                    f"[uv_baker] UVAtlas unwrapping failed ({uvatlas_err}). "
                    f"Falling back to robust xatlas backend..."
                )
                use_xatlas = True
                unwrap_meta = {
                    "uvatlas_fallback": True,
                    "uvatlas_fallback_reason": str(uvatlas_err)
                }

        if use_xatlas:
            n_faces = len(mesh.faces)
            c_opts = get_adaptive_chart_options(n_faces)
            if chart_options:
                for k, v in chart_options.items():
                    if hasattr(c_opts, k):
                        setattr(c_opts, k, v)

            p_opts = get_adaptive_pack_options(n_faces, target_res=res, padding=2)
            if pack_options:
                for k, v in pack_options.items():
                    if hasattr(p_opts, k):
                        setattr(p_opts, k, v)
                # Caller's texels_per_unit targets the initial canvas; rescale it for a re-pack at `res`
                if pack_options.get("texels_per_unit"):
                    p_opts.texels_per_unit = float(pack_options["texels_per_unit"]) * res / initial_res
            # Anti-blur: Always ensure rotate_charts is False to avoid diagonal resampling blur
            if not pack_options or "rotate_charts" not in pack_options:
                p_opts.rotate_charts = False
                p_opts.rotate_charts_to_axis = False

            atlas = xatlas.Atlas()
            # Use 3D mesh unwrap to eliminate giant sliver triangles and unassigned (0, 0) UV artifacts from add_uv_mesh
            atlas.add_mesh(
                np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                np.ascontiguousarray(mesh.faces, dtype=np.uint32)
            )
            atlas.generate(chart_options=c_opts, pack_options=p_opts)

            vmapping, indices, new_uv = atlas[0]
            vertices_recharted = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(vmapping, dtype=np.int64)]
            faces_recharted = np.asarray(indices, dtype=np.int64)
            uv_recharted = np.asarray(new_uv, dtype=np.float64)
            unwrap_meta.update({
                "xatlas_chart_count": int(atlas.chart_count),
                "xatlas_atlas_count": int(atlas.atlas_count),
                "xatlas_utilization_percent": round(float(atlas.utilization * 100.0), 2),
            })
        return vertices_recharted, faces_recharted, uv_recharted, vmapping, unwrap_meta

    vertices_recharted, faces_recharted, uv_recharted, vmapping, unwrap_meta = _unwrap(initial_res)

    # =========================================================================
    # BƯỚC B: Kiểm tra xem có thể downscale không (can_downscale_texture)
    # Enforces 1:1 capacity check so islands are never scaled down below 1:1
    # =========================================================================
    can_downscale, active_target_res, downscale_info = can_downscale_texture(
        mesh=mesh,
        current_res=initial_res,
        old_uv=source_uv,
        new_uv=uv_recharted,
        new_faces=faces_recharted,
        min_res=min_downscale_res,
        td_threshold_ratio=td_threshold_ratio,
        orig_island_pixels=orig_island_pixels
    )
    target_res = active_target_res

    # Gutter guarantee at the FINAL resolution: a downscale would halve the gaps between charts
    # (bleeding under GPU bilinear + mipmaps), so re-run Bước A at target_res.
    repacked_at_final_resolution = target_res < initial_res
    if repacked_at_final_resolution:
        vertices_recharted, faces_recharted, uv_recharted, vmapping, unwrap_meta = _unwrap(target_res)

    # =========================================================================
    # BƯỚC C: Hiệu chỉnh UV để tối đa hóa không gian texture
    # =========================================================================
    uv_recharted = maximize_uv_bounds(uv_recharted, target_res=target_res, padding_px=4)

    # =========================================================================
    # BƯỚC D: Nướng (Bake) texture & Lan viền 16px
    # =========================================================================
    sel, fid, bary = _rasterize_uv_atlas(faces_recharted, uv_recharted, target_res)
    if len(sel) == 0:
        raise RuntimeError("Failed to rasterize UV atlas during xatlas baking.")

    # Map recharted faces through vmapping back to exact original mesh vertices
    orig_face_verts = np.asarray(vmapping, dtype=np.int64)[faces_recharted]  # (N, 3)
    raw_tri_uv = source_uv[orig_face_verts[fid]]  # (len(sel), 3, 2)
    src_uv = (raw_tri_uv * bary[:, :, None]).sum(axis=1)
    src_uv = np.clip(src_uv, 0.0, 1.0)

    colors = _sample_texture_bilinear(src_img, src_uv)
    colors = np.nan_to_num(colors, nan=128.0)

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
    min_dilation = 8 if unwrap_method == "uvatlas" else 4
    eff_dilation_padding = max(min_dilation, int(dilation_padding))
    dilated_img = dilate_texture(base_img, covered, padding=eff_dilation_padding)
    dilated_pil = Image.fromarray(dilated_img, mode=out_mode)

    # Configure PBRMaterial with FrontSide rendering (doubleSided=double_sided, default False)
    orig_mat = getattr(mesh.visual, "material", None) if hasattr(mesh, "visual") and mesh.visual is not None else None
    if isinstance(orig_mat, trimesh.visual.material.PBRMaterial):
        mat = orig_mat.copy()
        mat.baseColorTexture = dilated_pil
        mat.doubleSided = double_sided
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=dilated_pil,
            metallicFactor=getattr(orig_mat, "metallicFactor", 0.0) if orig_mat else 0.0,
            roughnessFactor=getattr(orig_mat, "roughnessFactor", 0.8) if orig_mat else 0.8,
            doubleSided=double_sided
        )
        if orig_mat and hasattr(orig_mat, "baseColorFactor") and orig_mat.baseColorFactor is not None:
            mat.baseColorFactor = orig_mat.baseColorFactor
        if orig_mat and hasattr(orig_mat, "alphaMode") and orig_mat.alphaMode is not None:
            mat.alphaMode = orig_mat.alphaMode
        if orig_mat and hasattr(orig_mat, "alphaCutoff") and orig_mat.alphaCutoff is not None:
            mat.alphaCutoff = orig_mat.alphaCutoff

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
    opt_img = optimize_mesh_texture_for_export(recharted_mesh, preferred_format="PNG")
    if opt_img is not None:
        dilated_pil = opt_img

    # Comprehensive metrics recording
    mesh_area = float(mesh.area)
    covered_pixels = int(len(sel))
    total_pixels = target_res * target_res
    coverage_percent = round(float(covered_pixels / total_pixels * 100.0), 2)
    texel_density_linear = round(float(np.sqrt(covered_pixels) / np.sqrt(max(mesh_area, 1e-6))), 2)
    texel_density_area = round(float(covered_pixels / max(mesh_area, 1e-6)), 2)

    result_stats = {
        "downscaled": downscale_info["downscaled"],
        "originalResolution": downscale_info["originalResolution"],
        "finalResolution": downscale_info["finalResolution"],
        "original_resolution": initial_res,
        "final_resolution": target_res,
        "repacked_at_final_resolution": repacked_at_final_resolution,
        "orig_island_pixels": round(float(orig_island_pixels), 1),
        "textureFormat": "PNG",
        "uvCoverageRatio": round(float(covered_pixels / total_pixels), 4),
        "uvCoverageRatioOrig": downscale_info["uvCoverageRatioOrig"],
        "texelDensityOrig": downscale_info["texelDensityOrig"],
        "texelDensityFinal": texel_density_area,
        "texelDensityDelta": round(float(texel_density_area - downscale_info["texelDensityOrig"]), 2),
        "texelDensityDeltaPercent": round(float((texel_density_area - downscale_info["texelDensityOrig"]) / max(downscale_info["texelDensityOrig"], 1e-6) * 100.0), 2),
        "target_resolution": target_res,
        "canvas_pixels": total_pixels,
        "covered_pixels": covered_pixels,
        "uv_coverage_ratio_percent": coverage_percent,
        "texel_density_linear": texel_density_linear,
        "texel_density_area": texel_density_area,
        "mesh_surface_area": round(mesh_area, 4),
        "dilation_padding": eff_dilation_padding,
        "double_sided": double_sided
    }
    result_stats.update(unwrap_meta)

    if stats is not None:
        stats.update(result_stats)
    if hasattr(recharted_mesh, "metadata"):
        recharted_mesh.metadata["uv_metrics"] = result_stats

    if return_stats:
        return recharted_mesh, dilated_pil, result_stats
    return recharted_mesh, dilated_pil


def rebake_texture_uvatlas(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    target_res: int = 1024,
    dilation_padding: int = 16,
    gutter: float = 4.0,
    double_sided: Optional[bool] = None
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """
    Repacks UV charts with Microsoft UVAtlas and bakes new high-coverage texture.
    Preserves 100% triangles (Zero-Decimation).
    Enforces gutter >= 4.0 and mandatory EDT dilation (min 8px) to prevent white seams.
    """
    if double_sided is None:
        double_sided = True

    target_res = clamp_target_resolution(target_res, source_image.size)
    eff_padding = max(8, int(dilation_padding))
    eff_gutter = max(4.0, float(gutter))
    res = rechart_and_bake_high_density(
        mesh=mesh,
        target_res=target_res,
        source_image=source_image,
        source_uv=source_uv,
        dilation_padding=eff_padding,
        double_sided=double_sided,
        return_stats=False,
        unwrap_method="uvatlas",
        uvatlas_gutter=eff_gutter
    )
    return res[0], res[1]


def rebake_texture_xatlas(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    target_res: int = 1024,
    dilation_padding: int = 16,
    target_coverage: Optional[float] = None,
    double_sided: Optional[bool] = None
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """
    Repacks UV charts with xatlas and bakes new high-coverage texture.
    Preserves 100% triangles (Zero-Decimation).
    Maintains backward compatibility with pipeline calls (guaranteeing doubleSided=True).
    """
    if double_sided is None:
        double_sided = True

    target_res = clamp_target_resolution(target_res, source_image.size)
    pack_opts = None
    if target_coverage is not None and target_coverage < 0.99:
        area = mesh.area
        tpu = float(np.sqrt(target_coverage * target_res * target_res / max(area, 1e-4)))
        pack_opts = {"texels_per_unit": tpu}

    res = rechart_and_bake_high_density(
        mesh=mesh,
        target_res=target_res,
        source_image=source_image,
        source_uv=source_uv,
        dilation_padding=dilation_padding,
        pack_options=pack_opts,
        double_sided=double_sided,
        return_stats=False
    )
    return res[0], res[1]
