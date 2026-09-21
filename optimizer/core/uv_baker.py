"""
uv_baker.py

UV Atlas Re-charting, Direct Barycentric Baking, and 16px Boundary Dilation.
Ensures zero black edge bleeding during GPU texture mipmapping.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh
import xatlas


def _sample_texture_bilinear(image_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Samples RGB image at normalized UV coordinates [0, 1] using bilinear interpolation."""
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


def _rasterize_uv_atlas(faces: np.ndarray, uv: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterizes UV atlas into a (dim x dim) grid.
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
    total_samples = int(np.sum(reps))

    fids = np.repeat(idx, reps)
    offsets = np.concatenate([[0], np.cumsum(reps)])[:-1]
    seq = np.arange(total_samples) - np.repeat(offsets, reps)
    sx = np.repeat(span_x[idx], reps)

    grid_x = np.repeat(min_x[idx], reps) + (seq % sx)
    grid_y = np.repeat(min_y[idx], reps) + (seq // sx)

    p0x, p0y = px[fids, 0], py[fids, 0]
    p1x, p1y = px[fids, 1], py[fids, 1]
    p2x, p2y = px[fids, 2], py[fids, 2]

    det = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
    nonzero_det = np.abs(det) > 1e-10

    gx = grid_x[nonzero_det]
    gy = grid_y[nonzero_det]
    ff = fids[nonzero_det]
    d = det[nonzero_det]

    p0x_s, p0y_s = p0x[nonzero_det], p0y[nonzero_det]
    p1x_s, p1y_s = p1x[nonzero_det], p1y[nonzero_det]
    p2x_s, p2y_s = p2x[nonzero_det], p2y[nonzero_det]

    w0 = ((p1y_s - p2y_s) * (gx - p2x_s) + (p2x_s - p1x_s) * (gy - p2y_s)) / d
    w1 = ((p2y_s - p0y_s) * (gx - p2x_s) + (p0x_s - p2x_s) * (gy - p2y_s)) / d
    w2 = 1.0 - w0 - w1

    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not np.any(inside):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.empty((0, 3))

    gx_in = gx[inside]
    gy_in = gy[inside]
    flat_idx = gy_in * dim + gx_in
    final_fids = ff[inside]
    bary = np.column_stack([w0[inside], w1[inside], w2[inside]])

    # Deduplicate pixels in case multiple triangles cover the same pixel
    uniq_idx, first_idx = np.unique(flat_idx, return_index=True)
    return uniq_idx, final_fids[first_idx], bary[first_idx]


def dilate_texture(image_rgb: np.ndarray, mask_covered: np.ndarray, padding: int = 16) -> np.ndarray:
    """
    Dilates covered pixel colors into non-covered background by `padding` pixels.
    Eliminates dark / black edge artifacts when generating GPU mipmaps.
    """
    if mask_covered.all():
        return image_rgb

    out = image_rgb.copy()
    dist, near = ndimage.distance_transform_edt(~mask_covered, return_distances=True, return_indices=True)
    grow = ~mask_covered & (dist <= padding)
    rows, cols = near[0][grow], near[1][grow]
    out[grow] = out[rows, cols]
    return out


def rebake_texture_xatlas(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    source_uv: np.ndarray,
    target_res: int = 1024,
    dilation_padding: int = 16,
    target_coverage: float = 0.52
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """
    Repacks UV charts with xatlas and bakes new high-coverage texture.
    Preserves 100% triangles (Zero-Decimation).
    """
    area = mesh.area
    tpu = float(np.sqrt(target_coverage * target_res * target_res / max(area, 1e-4)))

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    )

    pack = xatlas.PackOptions()
    pack.resolution = target_res
    pack.padding = 4
    pack.texels_per_unit = tpu
    pack.bilinear = True
    pack.rotate_charts = True
    pack.rotate_charts_to_axis = True
    atlas.generate(pack_options=pack)

    vmapping, indices, new_uv = atlas[0]
    vertices_recharted = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(vmapping, dtype=np.int64)]
    faces_recharted = np.asarray(indices, dtype=np.int64)
    uv_recharted = np.asarray(new_uv, dtype=np.float64)

    src_img_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)

    # Rasterize new atlas and sample source texture
    sel, fid, bary = _rasterize_uv_atlas(faces_recharted, uv_recharted, target_res)
    if len(sel) == 0:
        raise RuntimeError("Failed to rasterize UV atlas during xatlas baking.")

    orig_faces = mesh.faces[fid]
    raw_tri_uv = source_uv[orig_faces]
    src_uv = (raw_tri_uv * bary[:, :, None]).sum(axis=1)

    colors = _sample_texture_bilinear(src_img_rgb, src_uv)
    colors = np.nan_to_num(colors, nan=128.0)

    base_flat = np.zeros((target_res * target_res, 3), dtype=np.uint8)
    base_flat[sel] = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    base_img = base_flat.reshape(target_res, target_res, 3)

    covered = np.zeros(target_res * target_res, dtype=bool)
    covered[sel] = True
    covered = covered.reshape(target_res, target_res)

    # Apply 16px dilation
    dilated_img = dilate_texture(base_img, covered, padding=dilation_padding)
    dilated_pil = Image.fromarray(dilated_img, mode="RGB")

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=dilated_pil,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        doubleSided=False
    )

    recharted_mesh = trimesh.Trimesh(vertices=vertices_recharted, faces=faces_recharted, process=False)
    recharted_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_recharted, material=mat)

    return recharted_mesh, dilated_pil


def direct_resample_texture(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    target_res: int = 1024,
    dilation_padding: int = 16
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """
    Direct mode: Keeps 100% original UVs, resamples texture with Lanczos and applies 16px dilation.
    """
    resampled = source_image.convert("RGB").resize((target_res, target_res), Image.Resampling.LANCZOS)
    arr = np.array(resampled, dtype=np.uint8)

    # Detect black/empty background pixels
    is_black = np.all(arr <= 2, axis=-1)
    is_covered = ~is_black

    if np.any(is_black) and not np.all(is_black):
        arr = dilate_texture(arr, is_covered, padding=dilation_padding)

    clean_pil = Image.fromarray(arr, mode="RGB")

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=clean_pil,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        doubleSided=False
    )

    out_mesh = mesh.copy()
    out_mesh.visual.material = mat
    return out_mesh, clean_pil
