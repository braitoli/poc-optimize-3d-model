"""
texture_utils.py

Utilities for preserving original texture formats and raw binary streams
across geometric transformations in 3D optimization pipelines.
Prevents redundant re-compression to heavy PNGs during Steps 1 & 2.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import io
import math
import sys
import numpy as np
from PIL import Image
import trimesh


def extract_original_texture_info(glb_path: Path, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extracts original texture format and raw binary bytes from a source GLB.
    Preserves original formats (JPEG, PNG, WebP) and binary bytes to eliminate
    costly PNG re-compression in geometric stages (Step 1 and Step 2).
    """
    orig_info: Dict[str, Any] = {
        "default_format": "JPEG",
        "slots": {},
        "base_image": None
    }

    try:
        raw_scene = trimesh.load(str(glb_path), process=False)
        geoms = raw_scene.geometry.values() if isinstance(raw_scene, trimesh.Scene) else [raw_scene]
        for g in geoms:
            mat = getattr(getattr(g, "visual", None), "material", None)
            if not mat:
                continue
            for attr in [
                "baseColorTexture",
                "image",
                "metallicRoughnessTexture",
                "normalTexture",
                "emissiveTexture",
                "occlusionTexture"
            ]:
                img = getattr(mat, attr, None)
                if img is not None and isinstance(img, Image.Image):
                    slot_data: Dict[str, Any] = {
                        "format": getattr(img, "format", None),
                        "raw_bytes": None,
                        "image": img
                    }
                    if hasattr(img, "fp") and img.fp is not None:
                        try:
                            img.fp.seek(0)
                            slot_data["raw_bytes"] = img.fp.read()
                        except Exception:
                            pass
                    if slot_data["format"]:
                        orig_info["default_format"] = slot_data["format"]
                    orig_info["slots"][attr] = slot_data
                    if attr in ("baseColorTexture", "image") and orig_info["base_image"] is None:
                        orig_info["base_image"] = img
    except Exception:
        pass

    if metrics and metrics.get("textures"):
        fmt_name = metrics["textures"][0].get("format")
        if fmt_name:
            fmt_upper = fmt_name.upper()
            if fmt_upper in ("JPEG", "JPG"):
                orig_info["default_format"] = "JPEG"
            elif fmt_upper == "PNG":
                orig_info["default_format"] = "PNG"
            elif fmt_upper == "WEBP":
                orig_info["default_format"] = "WEBP"

    return orig_info


def preserve_mesh_textures(mesh: trimesh.Trimesh, tex_info: Dict[str, Any]) -> None:
    """
    Re-attaches original format and fast-save raw binary bytes to a mesh's
    material textures so trimesh.exchange.gltf.export_glb will export original
    texture bytes without re-encoding to heavy PNG.
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return

    slots = tex_info.get("slots", {})
    default_format = tex_info.get("default_format", "JPEG")

    for attr in [
        "baseColorTexture",
        "image",
        "metallicRoughnessTexture",
        "normalTexture",
        "emissiveTexture",
        "occlusionTexture"
    ]:
        img = getattr(mat, attr, None)
        if img is not None and isinstance(img, Image.Image):
            slot_data = slots.get(attr)
            fmt = (slot_data and slot_data.get("format")) or default_format
            img.format = fmt

            raw_bytes = slot_data and slot_data.get("raw_bytes")
            if raw_bytes:
                def make_fast_save(data_bytes):
                    def fast_save(f, format=None, **kwargs):
                        f.write(data_bytes)
                    return fast_save
                img.save = make_fast_save(raw_bytes)
                img._fast_save_data = raw_bytes


def clamp_target_resolution(
    requested_res: Union[int, str],
    orig_size: Tuple[int, int],
    logger_fn: Optional[Any] = None
) -> int:
    """
    Core principle: NEVER UPSCALE TEXTURES.
    If requested_res == 'auto':
        Calculates the largest power of 2 <= max(orig_size).
    If requested_res > max(orig_size):
        Clamps to the largest power of 2 such that 2**k <= max(orig_size).
        (e.g., for 1536x1536, max POT <= 1536 is 1024; for 768x768, max POT <= 768 is 512).
    If requested_res <= max(orig_size):
        Retains requested_res unchanged.
    Logs clearly:
        [Step 3] Original texture: {w}x{h}, requested: {requested_res} -> clamped to {target_res} (NO-UPSCALE policy)
    """
    w, h = orig_size
    orig_max = max(w, h)
    if orig_max <= 0:
        return 1024

    max_pot = 1 << int(math.floor(math.log2(orig_max)))
    max_pot = max(1, max_pot)

    if isinstance(requested_res, str) and requested_res.lower() == "auto":
        return max_pot

    try:
        req = int(requested_res)
    except (ValueError, TypeError):
        return max_pot

    if req > orig_max:
        msg = f"[Step 3] Original texture: {w}x{h}, requested: {req} -> clamped to {max_pot} (NO-UPSCALE policy)"
        if logger_fn is not None:
            logger_fn(msg)
        else:
            print(f"   {msg}", file=sys.stderr, flush=True)
        return max_pot
    else:
        return req


def can_downscale_texture(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    uv: Optional[np.ndarray] = None,
    target_res: Union[int, str, None] = "auto",
    min_texel_density: float = 120.0,
    logger_fn: Optional[Any] = None
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Evaluates whether texture resolution can be safely downscaled based on:
    1. Organizing UV tightly into square canvas [0, 1] x [0, 1]
    2. Evaluating texel density at candidate downscaled resolution
    3. If texel density >= min_texel_density (and candidate_res >= 512):
       returns (True, candidate_res, details)
       otherwise:
       returns (False, base_res, details)
    """
    from optimizer.core.uv_baker import compute_uv_metrics

    w, h = source_image.size
    orig_max = max(w, h)
    if orig_max <= 0:
        return False, 1024, {"reason": "invalid_texture_size"}

    # Base power-of-2 resolution adhering to NO-UPSCALE policy
    base_res = 1 << int(math.floor(math.log2(orig_max)))
    base_res = max(256, base_res)

    if uv is None:
        uv = getattr(mesh.visual, "uv", None)

    # 1. Analyze UV bounding box and organization into square
    uv_details = {}
    uv_box_area = 1.0
    if uv is not None and len(uv) > 0:
        valid_mask = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        if np.any(valid_mask):
            valid_uv = uv[valid_mask]
            u_min, v_min = valid_uv.min(axis=0)
            u_max, v_max = valid_uv.max(axis=0)
            span_u = max(0.001, min(1.0, float(u_max)) - max(0.0, float(u_min)))
            span_v = max(0.001, min(1.0, float(v_max)) - max(0.0, float(v_min)))
            uv_box_area = span_u * span_v
            uv_details = {
                "u_min": round(float(u_min), 4),
                "u_max": round(float(u_max), 4),
                "v_min": round(float(v_min), 4),
                "v_max": round(float(v_max), 4),
                "span_u": round(span_u, 4),
                "span_v": round(span_v, 4),
                "box_area": round(uv_box_area, 4)
            }

    # 2. Compute current texel density at base_res
    base_metrics = compute_uv_metrics(mesh, target_res=base_res, uv=uv) if uv is not None else {}
    base_td = base_metrics.get("texel_density_linear", 0.0)

    # 3. Determine candidate downscaled resolution
    if base_res >= 4096:
        candidate_res = 2048
    elif base_res >= 2048:
        candidate_res = 1024
    elif base_res == 1024:
        candidate_res = 512
    else:
        candidate_res = base_res

    # 4. Compute candidate texel density with UV space maximization gain
    cand_metrics = compute_uv_metrics(mesh, target_res=candidate_res, uv=uv) if uv is not None else {}
    cand_td = cand_metrics.get("texel_density_linear", 0.0)
    space_gain = min(1.5, 1.0 / np.sqrt(max(0.25, uv_box_area)))
    effective_cand_td = cand_td * space_gain

    # 5. Decision: can we downscale?
    can_downscale = False
    if candidate_res < base_res and candidate_res >= 512:
        if base_res >= 4096:
            # 4K textures always downscale to 2K for GPU memory & web performance
            can_downscale = True
        elif effective_cand_td >= min_texel_density:
            can_downscale = True
        elif base_res >= 2048 and effective_cand_td >= (min_texel_density * 0.75):
            can_downscale = True

    optimal_res = candidate_res if can_downscale else base_res

    details = {
        "original_size": [w, h],
        "base_res": base_res,
        "candidate_res": candidate_res,
        "optimal_res": optimal_res,
        "can_downscale": can_downscale,
        "base_texel_density": round(base_td, 2),
        "candidate_texel_density": round(cand_td, 2),
        "effective_candidate_texel_density": round(effective_cand_td, 2),
        "min_texel_density_threshold": min_texel_density,
        "uv_space_gain": round(space_gain, 2),
        "uv_box": uv_details
    }

    if logger_fn:
        decision_str = f"Downscaled to {optimal_res}px" if can_downscale else f"Retained {optimal_res}px"
        logger_fn(
            f"[Auto-Resolution] {decision_str} (Orig: {w}x{h}, Base: {base_res}px TD={base_td:.1f} px/u, "
            f"Candidate: {candidate_res}px TD={cand_td:.1f} px/u, MinTD={min_texel_density:.1f} px/u, "
            f"UV Box Area={uv_box_area:.2f})"
        )

    return can_downscale, optimal_res, details


def maximize_uv_space(
    mesh: trimesh.Trimesh,
    source_image: Image.Image,
    uv: Optional[np.ndarray] = None,
    target_res: Optional[int] = None,
    margin_ratio: float = 0.005,
    logger_fn: Optional[Any] = None
) -> Tuple[trimesh.Trimesh, Image.Image, np.ndarray, Dict[str, Any]]:
    """
    Adjusts UV coordinates and texture canvas to maximize utilization of the square space.
    If UVs only occupy a sub-rectangle within [0, 1] x [0, 1] with substantial margins,
    crops the texture to the square-adjusted UV bounding box and normalizes UVs to [0, 1],
    ensuring maximum texel density on surface faces.
    """
    if uv is None:
        uv = getattr(mesh.visual, "uv", None)

    if uv is None or len(uv) == 0:
        return mesh, source_image, uv, {"adjusted": False, "reason": "no_uv"}

    valid_mask = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    if not np.any(valid_mask):
        return mesh, source_image, uv, {"adjusted": False, "reason": "invalid_uv"}

    valid_uv = uv[valid_mask]
    u_min, v_min = valid_uv.min(axis=0)
    u_max, v_max = valid_uv.max(axis=0)

    # Check if UVs occupy a sub-rectangle within [0, 1] with unused margin
    has_unused_margin = (
        (u_min > 0.05 or v_min > 0.05 or u_max < 0.95 or v_max < 0.95) and
        (u_min >= -0.05 and v_min >= -0.05 and u_max <= 1.05 and v_max <= 1.05)
    )

    if not has_unused_margin:
        return mesh, source_image, uv, {
            "adjusted": False,
            "reason": "uv_already_utilizes_canvas",
            "bounds": [float(u_min), float(v_min), float(u_max), float(v_max)]
        }

    # Calculate square bounding box to maintain 1:1 aspect ratio
    u_span = u_max - u_min
    v_span = v_max - v_min
    max_span = max(u_span, v_span)
    max_span_with_margin = min(1.0, max_span * (1.0 + margin_ratio * 2))

    u_center = (u_min + u_max) / 2.0
    v_center = (v_min + v_max) / 2.0

    sq_u_min = max(0.0, u_center - max_span_with_margin / 2.0)
    sq_u_max = min(1.0, sq_u_min + max_span_with_margin)
    sq_v_min = max(0.0, v_center - max_span_with_margin / 2.0)
    sq_v_max = min(1.0, sq_v_min + max_span_with_margin)

    # Recompute spans
    sq_span_u = sq_u_max - sq_u_min
    sq_span_v = sq_v_max - sq_v_min

    if sq_span_u < 0.05 or sq_span_v < 0.05:
        return mesh, source_image, uv, {"adjusted": False, "reason": "span_too_small"}

    # Crop image
    w, h = source_image.size
    crop_x0 = int(sq_u_min * w)
    crop_x1 = int(sq_u_max * w)
    crop_y0 = int((1.0 - sq_v_max) * h)
    crop_y1 = int((1.0 - sq_v_min) * h)

    crop_x0 = max(0, min(crop_x0, w - 1))
    crop_x1 = max(crop_x0 + 1, min(crop_x1, w))
    crop_y0 = max(0, min(crop_y0, h - 1))
    crop_y1 = max(crop_y0 + 1, min(crop_y1, h))

    cropped_img = source_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    if target_res:
        cropped_img = cropped_img.resize((target_res, target_res), Image.Resampling.LANCZOS)

    # Rescale UVs into [0, 1]
    new_uv = uv.copy()
    new_uv[:, 0] = np.clip((uv[:, 0] - sq_u_min) / sq_span_u, 0.0, 1.0)
    new_uv[:, 1] = np.clip((uv[:, 1] - sq_v_min) / sq_span_v, 0.0, 1.0)

    if hasattr(mesh.visual, "uv"):
        mesh.visual.uv = new_uv

    if logger_fn:
        logger_fn(
            f"[UV Maximizer] Adjusted UV to square canvas: U=[{sq_u_min:.3f}, {sq_u_max:.3f}], "
            f"V=[{sq_v_min:.3f}, {sq_v_max:.3f}] -> rescaled to [0, 1]"
        )

    return mesh, cropped_img, new_uv, {
        "adjusted": True,
        "old_bounds": [float(u_min), float(v_min), float(u_max), float(v_max)],
        "new_bounds": [float(sq_u_min), float(sq_v_min), float(sq_u_max), float(sq_v_max)],
        "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1]
    }



def optimize_mesh_texture_for_export(
    mesh: trimesh.Trimesh,
    orig_tex_info: Optional[Dict[str, Any]] = None,
    preferred_format: Optional[str] = None,
    jpeg_quality: int = 92
) -> Optional[Image.Image]:
    """
    Optimizes the mesh texture before trimesh.exchange.gltf.export_glb to prevent
    uncompressed PNG bloat.
    - If texture has no alpha channel (RGB) or original format was JPEG, saves as high quality
      JPEG (quality=92, optimize=True) and sets format='JPEG' with fast_save.
    - If texture has an active alpha channel, optimizes PNG compression (optimize=True, compress_level=9)
      and sets format='PNG' with fast_save.
    Returns the optimized PIL Image object.
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return None
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None

    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None or not isinstance(img, Image.Image):
        return None

    # Check if texture has an alpha channel
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )

    # Check original format
    orig_format = (orig_tex_info or {}).get("default_format", "JPEG").upper()

    # Determine target format:
    # 1. If preferred_format is specified, respect it.
    # 2. If texture has an alpha channel (RGBA/LA), it MUST be PNG to preserve transparency.
    # 3. If texture has NO alpha channel (RGB) or original format is JPEG, save as JPEG.
    if preferred_format:
        target_fmt = preferred_format.upper()
    elif has_alpha:
        target_fmt = "PNG"
    elif orig_format in ("JPEG", "JPG") or img.mode == "RGB":
        target_fmt = "JPEG"
    else:
        target_fmt = "PNG"

    def make_fast_save(data_bytes: bytes):
        def fast_save(f, format=None, **kwargs):
            f.write(data_bytes)
        return fast_save

    if target_fmt in ("JPEG", "JPG"):
        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        data = buf.getvalue()

        rgb_img.format = "JPEG"
        rgb_img.save = make_fast_save(data)
        rgb_img._fast_save_data = data

        if hasattr(mat, "baseColorTexture"):
            mat.baseColorTexture = rgb_img
        if hasattr(mat, "image"):
            mat.image = rgb_img
        return rgb_img
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=9)
        data = buf.getvalue()

        img.format = "PNG"
        img.save = make_fast_save(data)
        img._fast_save_data = data

        if hasattr(mat, "baseColorTexture"):
            mat.baseColorTexture = img
        if hasattr(mat, "image"):
            mat.image = img
        return img

