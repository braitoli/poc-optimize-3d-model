"""
uv_baker.py

UV Atlas Re-charting, Direct Barycentric Baking, and 16px Boundary Dilation.
Maximizes texture canvas utilization, Texel Density, and surface sharpness.
Ensures zero black edge bleeding during GPU texture mipmapping.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh
import xatlas
from optimizer.core.texture_utils import clamp_target_resolution, optimize_mesh_texture_for_export


def _sample_texture_bilinear(image_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Samples RGB or RGBA image at normalized UV coordinates [0, 1] using bilinear interpolation."""
    h, w = image_rgb.shape[:2]
    u = (uv[:, 0] % 1.0) * (w - 1)
    v = (1.0 - (uv[:, 1] % 1.0)) * (h - 1)

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
    max_batch_samples: int = 2000000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterizes UV atlas into a (dim x dim) grid using memory-bounded batching.
    Returns:
      sel: 1D flat pixel indices covered by triangles
      fid: triangle index covering each selected pixel
      bary: barycentric weights (N, 3) for each selected pixel
    """
    tri_uv = uv[faces]  # (N, 3, 2)
    px = tri_uv[:, :, 0] * (dim - 1)
    py = (1.0 - tri_uv[:, :, 1]) * (dim - 1)

    min_x = np.clip(np.floor(px.min(axis=1)).astype(np.int32), 0, dim - 1)
    max_x = np.clip(np.ceil(px.max(axis=1)).astype(np.int32), 0, dim - 1)
    min_y = np.clip(np.floor(py.min(axis=1)).astype(np.int32), 0, dim - 1)
    max_y = np.clip(np.ceil(py.max(axis=1)).astype(np.int32), 0, dim - 1)

    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    areas = span_x * span_y

    valid = areas > 0
    if not np.any(valid):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))

    idx = np.where(valid)[0]
    reps = areas[idx]

    all_flat_idx = []
    all_fids = []
    all_bary = []

    n_triangles = len(idx)
    batch_start = 0

    while batch_start < n_triangles:
        cum = np.cumsum(reps[batch_start:])
        batch_end = batch_start + np.searchsorted(cum, max_batch_samples, side='right')
        batch_end = max(batch_end, batch_start + 1)
        batch_end = min(batch_end, n_triangles)

        b_idx = idx[batch_start:batch_end]
        b_reps = reps[batch_start:batch_end]
        b_total = int(np.sum(b_reps))

        b_fids = np.repeat(b_idx, b_reps)
        b_offsets = np.concatenate([[0], np.cumsum(b_reps)])[:-1]
        b_seq = np.arange(b_total) - np.repeat(b_offsets, b_reps)
        b_sx = np.repeat(span_x[b_idx], b_reps)

        grid_x = np.repeat(min_x[b_idx], b_reps) + (b_seq % b_sx)
        grid_y = np.repeat(min_y[b_idx], b_reps) + (b_seq // b_sx)

        p0x, p0y = px[b_fids, 0], py[b_fids, 0]
        p1x, p1y = px[b_fids, 1], py[b_fids, 1]
        p2x, p2y = px[b_fids, 2], py[b_fids, 2]

        det = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
        nonzero_det = np.abs(det) > 1e-10

        gx = grid_x[nonzero_det]
        gy = grid_y[nonzero_det]
        ff = b_fids[nonzero_det]
        d = det[nonzero_det]

        p0x_s, p0y_s = p0x[nonzero_det], p0y[nonzero_det]
        p1x_s, p1y_s = p1x[nonzero_det], p1y[nonzero_det]
        p2x_s, p2y_s = p2x[nonzero_det], p2y[nonzero_det]

        w0 = ((p1y_s - p2y_s) * (gx - p2x_s) + (p2x_s - p1x_s) * (gy - p2y_s)) / d
        w1 = ((p2y_s - p0y_s) * (gx - p2x_s) + (p0x_s - p2x_s) * (gy - p2y_s)) / d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if np.any(inside):
            gx_in = gx[inside]
            gy_in = gy[inside]
            all_flat_idx.append(gy_in * dim + gx_in)
            all_fids.append(ff[inside])
            all_bary.append(np.column_stack([w0[inside], w1[inside], w2[inside]]))

        batch_start = batch_end

    if not all_flat_idx:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))

    flat_idx = np.concatenate(all_flat_idx)
    final_fids = np.concatenate(all_fids)
    final_bary = np.vstack(all_bary)

    # Deduplicate pixels in case multiple triangles cover the same pixel
    uniq_idx, first_idx = np.unique(flat_idx, return_index=True)
    return uniq_idx, final_fids[first_idx], final_bary[first_idx]


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
    return_stats: bool = False
) -> Tuple[trimesh.Trimesh, Image.Image] | Tuple[trimesh.Trimesh, Image.Image, Dict[str, Any]]:
    """
    Unwraps and repacks UV charts using xatlas to maximize canvas space utilization
    and Texel Density across remaining exterior faces.
    
    Bakes color from source texture with direct barycentric interpolation and 16px dilation.
    Configures PBRMaterial with FrontSide rendering (`doubleSided=False`) while maintaining
    smooth, continuous surfaces.
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

    # 2. Configure xatlas Chart Options (max iterations, boundary optimization, fix winding)
    c_opts = xatlas.ChartOptions()
    c_opts.max_iterations = 4
    c_opts.fix_winding = True
    c_opts.normal_deviation_weight = 2.0
    c_opts.roundness_weight = 0.01
    c_opts.straightness_weight = 6.0
    c_opts.normal_seam_weight = 4.0
    c_opts.texture_seam_weight = 0.5
    c_opts.max_cost = 2.0

    if chart_options:
        for k, v in chart_options.items():
            if hasattr(c_opts, k):
                setattr(c_opts, k, v)

    # 3. Configure xatlas Pack Options (high-density pack, bilinear margin, 2px chart padding)
    p_opts = xatlas.PackOptions()
    p_opts.resolution = target_res
    p_opts.padding = 2
    p_opts.bilinear = True
    p_opts.rotate_charts = True
    p_opts.rotate_charts_to_axis = True
    p_opts.bruteForce = False

    if pack_options:
        for k, v in pack_options.items():
            if hasattr(p_opts, k):
                setattr(p_opts, k, v)

    # 4. Generate high-density UV Atlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    )
    atlas.generate(chart_options=c_opts, pack_options=p_opts)

    vmapping, indices, new_uv = atlas[0]
    vertices_recharted = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(vmapping, dtype=np.int64)]
    faces_recharted = np.asarray(indices, dtype=np.int64)
    uv_recharted = np.asarray(new_uv, dtype=np.float64)

    # 5. Rasterize new atlas and sample source texture with Barycentric interpolation
    sel, fid, bary = _rasterize_uv_atlas(faces_recharted, uv_recharted, target_res)
    if len(sel) == 0:
        raise RuntimeError("Failed to rasterize UV atlas during xatlas baking.")

    orig_faces = mesh.faces[fid]
    raw_tri_uv = source_uv[orig_faces]
    src_uv = (raw_tri_uv * bary[:, :, None]).sum(axis=1)

    colors = _sample_texture_bilinear(src_img, src_uv)
    colors = np.nan_to_num(colors, nan=128.0)

    base_flat = np.zeros((target_res * target_res, channels), dtype=np.uint8)
    base_flat[sel] = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    base_img = base_flat.reshape(target_res, target_res, channels)

    covered = np.zeros(target_res * target_res, dtype=bool)
    covered[sel] = True
    covered = covered.reshape(target_res, target_res)

    # 6. Apply 16px dilation to prevent black edge bleeding during mipmapping
    dilated_img = dilate_texture(base_img, covered, padding=dilation_padding)
    dilated_pil = Image.fromarray(dilated_img, mode=out_mode)

    # 7. Material setup with doubleSided=double_sided (FrontSide rendering by default)
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
        # Recompute smooth vertex normals for smooth shading
        _ = recharted_mesh.vertex_normals

    recharted_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_recharted, material=mat)

    # Attach optimized fast_save texture to mesh material and update dilated_pil
    opt_img = optimize_mesh_texture_for_export(recharted_mesh)
    if opt_img is not None:
        dilated_pil = opt_img

    # 8. Compute Texel Density & UV Coverage metrics
    mesh_area = float(mesh.area)
    covered_pixels = int(len(sel))
    total_pixels = target_res * target_res
    coverage_percent = round(float(covered_pixels / total_pixels * 100.0), 2)
    texel_density_linear = round(float(np.sqrt(covered_pixels) / np.sqrt(max(mesh_area, 1e-6))), 2)
    texel_density_area = round(float(covered_pixels / max(mesh_area, 1e-6)), 2)

    result_stats = {
        "target_resolution": target_res,
        "canvas_pixels": total_pixels,
        "covered_pixels": covered_pixels,
        "uv_coverage_ratio_percent": coverage_percent,
        "texel_density_linear": texel_density_linear,
        "texel_density_area": texel_density_area,
        "mesh_surface_area": round(mesh_area, 4),
        "xatlas_chart_count": int(atlas.chart_count),
        "xatlas_atlas_count": int(atlas.atlas_count),
        "xatlas_utilization_percent": round(float(atlas.utilization * 100.0), 2),
        "dilation_padding": dilation_padding,
        "double_sided": double_sided
    }

    if stats is not None:
        stats.update(result_stats)
    if hasattr(recharted_mesh, "metadata"):
        recharted_mesh.metadata["uv_metrics"] = result_stats

    if return_stats:
        return recharted_mesh, dilated_pil, result_stats
    return recharted_mesh, dilated_pil


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


def direct_resample_texture(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    target_res: int = 1024,
    dilation_padding: int = 16,
    copy_mesh: bool = False
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """
    Direct mode: Keeps 100% original UVs, resamples texture with Lanczos and applies 16px dilation.
    Preserves vertex normals, alpha channel, and material properties.
    """
    # Defense-in-depth: Never upscale texture
    target_res = clamp_target_resolution(target_res, source_image.size)

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
        img = source_image.convert("RGBA") if source_image.mode != "RGBA" else source_image
        out_mode = "RGBA"
    else:
        img = source_image.convert("RGB") if source_image.mode != "RGB" else source_image
        out_mode = "RGB"

    if img.size == (target_res, target_res):
        resampled = img
    else:
        resampled = img.resize((target_res, target_res), Image.Resampling.LANCZOS)
    arr = np.array(resampled, dtype=np.uint8)

    # Detect covered pixels for dilation
    if has_alpha:
        alpha = arr[:, :, 3]
        if np.any(alpha == 0):
            is_covered = alpha > 0
        else:
            is_black = np.all(arr[:, :, :3] <= 2, axis=-1)
            is_covered = ~is_black
    else:
        is_black = np.all(arr <= 2, axis=-1)
        is_covered = ~is_black

    if np.any(~is_covered) and not np.all(~is_covered):
        arr = dilate_texture(arr, is_covered, padding=dilation_padding)

    clean_pil = Image.fromarray(arr, mode=out_mode)

    orig_mat = getattr(mesh.visual, "material", None) if hasattr(mesh, "visual") and mesh.visual is not None else None
    if isinstance(orig_mat, trimesh.visual.material.PBRMaterial):
        mat = orig_mat.copy()
        mat.baseColorTexture = clean_pil
        mat.doubleSided = True
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=clean_pil,
            metallicFactor=getattr(orig_mat, "metallicFactor", 0.0) if orig_mat else 0.0,
            roughnessFactor=getattr(orig_mat, "roughnessFactor", 0.8) if orig_mat else 0.8,
            doubleSided=True
        )
        if orig_mat and hasattr(orig_mat, "baseColorFactor") and orig_mat.baseColorFactor is not None:
            mat.baseColorFactor = orig_mat.baseColorFactor
        if orig_mat and hasattr(orig_mat, "alphaMode") and orig_mat.alphaMode is not None:
            mat.alphaMode = orig_mat.alphaMode
        if orig_mat and hasattr(orig_mat, "alphaCutoff") and orig_mat.alphaCutoff is not None:
            mat.alphaCutoff = orig_mat.alphaCutoff

    out_mesh = mesh.copy() if copy_mesh else mesh

    # Ensure vertex normals are preserved
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        if not hasattr(out_mesh, "vertex_normals") or out_mesh.vertex_normals is None:
            out_mesh.vertex_normals = mesh.vertex_normals.copy()

    if hasattr(out_mesh, "visual") and isinstance(out_mesh.visual, trimesh.visual.TextureVisuals):
        out_mesh.visual.material = mat
    else:
        out_mesh.visual = trimesh.visual.TextureVisuals(
            uv=getattr(mesh.visual, "uv", None),
            material=mat
        )

    opt_img = optimize_mesh_texture_for_export(out_mesh)
    if opt_img is not None:
        clean_pil = opt_img

    return out_mesh, clean_pil
