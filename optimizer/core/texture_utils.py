"""
texture_utils.py

Utilities for preserving original texture formats and raw binary streams
across geometric transformations in 3D optimization pipelines.
Prevents redundant re-compression to heavy PNGs during Steps 1 & 2.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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
    requested_res: int,
    orig_size: Tuple[int, int],
    logger_fn: Optional[Any] = None
) -> int:
    """
    Core principle: NEVER UPSCALE TEXTURES.
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
        return requested_res

    if requested_res > orig_max:
        # Largest power of 2 <= orig_max
        target_res = 1 << int(math.floor(math.log2(orig_max)))
        target_res = max(1, target_res)
        msg = f"[Step 3] Original texture: {w}x{h}, requested: {requested_res} -> clamped to {target_res} (NO-UPSCALE policy)"
        if logger_fn is not None:
            logger_fn(msg)
        else:
            print(f"   {msg}", file=sys.stderr, flush=True)
        return target_res
    else:
        return requested_res


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

