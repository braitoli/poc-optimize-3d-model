"""
texture_utils.py

Utilities for preserving original texture formats and raw binary streams
across geometric transformations in 3D optimization pipelines.
Prevents redundant re-compression to heavy PNGs during Steps 1 & 2.
"""

from pathlib import Path
from typing import Dict, Any, Optional
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
