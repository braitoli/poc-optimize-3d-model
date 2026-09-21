"""
texture_utils.py

Utilities for preserving original texture formats and raw binary streams
across geometric transformations in 3D optimization pipelines.
Prevents redundant re-compression to heavy PNGs during Steps 1 & 2.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import io
import numpy as np
from PIL import Image
import trimesh

from optimizer.core.errors import PipelineAbort


TEXTURE_SLOTS = (
    "baseColorTexture",
    "image",
    "metallicRoughnessTexture",
    "normalTexture",
    "emissiveTexture",
    "occlusionTexture"
)
SUPPORTED_TEXTURE_FORMATS = ("JPEG", "PNG", "WEBP")


def _encoded_bytes(img: Image.Image, slot: str) -> bytes:
    """The encoded bytes a texture was decoded from (read before anything decodes the pixels,
    which closes the stream). Without them the texture could only be re-encoded: raise instead."""
    fp = getattr(img, "fp", None)
    if fp is None:
        raise PipelineAbort(f"The encoded bytes of the {slot} image are not available (its stream is already closed)")
    try:
        fp.seek(0)
        data = fp.read()
    except Exception as err:
        raise PipelineAbort(f"Cannot read the encoded bytes of the {slot} image: {err}") from err
    if not data:
        raise PipelineAbort(f"The {slot} image has no encoded bytes")
    return data


def extract_original_texture_info(glb_path: Path) -> Dict[str, Any]:
    """
    Extracts original texture format and raw binary bytes from a source GLB.
    Preserves original formats (JPEG, PNG, WebP) and binary bytes to eliminate
    costly PNG re-compression in geometric stages (Step 1 and Step 2).
    Raises PipelineAbort if the GLB cannot be read, is not exactly one mesh primitive with one
    material placed once, has no decodable base colour texture, or a texture's format is not
    JPEG / PNG / WebP or its encoded bytes cannot be read.
    """
    name = Path(glb_path).name
    try:
        raw_scene = trimesh.load(str(glb_path), process=False)
    except Exception as err:
        raise PipelineAbort(f"Cannot read the textures of {name}: {err}") from err

    if isinstance(raw_scene, trimesh.Scene):
        geoms = list(raw_scene.geometry.values())
        placements = len(raw_scene.graph.nodes_geometry)
    else:
        geoms, placements = [raw_scene], 1
    if len(geoms) != 1 or placements != 1:
        raise PipelineAbort(
            f"{name} has {len(geoms)} mesh primitive(s) placed {placements} time(s): "
            f"only single-mesh, single-material models are supported"
        )

    mat = getattr(getattr(geoms[0], "visual", None), "material", None)
    slots: Dict[str, Dict[str, Any]] = {}
    for attr in TEXTURE_SLOTS:
        img = getattr(mat, attr, None)
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            raise PipelineAbort(f"The {attr} of {name} is a {type(img).__name__}, not an image")
        if img.format not in SUPPORTED_TEXTURE_FORMATS:
            raise PipelineAbort(
                f"The {attr} of {name} has image format {img.format!r}; supported: {', '.join(SUPPORTED_TEXTURE_FORMATS)}"
            )
        slots[attr] = {"format": img.format, "raw_bytes": _encoded_bytes(img, attr), "image": img}

    base = slots.get("baseColorTexture") or slots.get("image")
    if base is None:
        raise PipelineAbort(f"{name} has no decodable baseColorTexture: only textured models are supported")
    return {"default_format": base["format"], "slots": slots, "base_image": base["image"]}


def preserve_mesh_textures(mesh: trimesh.Trimesh, tex_info: Dict[str, Any]) -> None:
    """
    Re-attaches original format and fast-save raw binary bytes to a mesh's
    material textures so trimesh.exchange.gltf.export_glb will export original
    texture bytes without re-encoding to heavy PNG.
    Every texture on the mesh needs its source slot (format + raw bytes) in tex_info, else PipelineAbort.
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return

    slots = tex_info["slots"]
    for attr in TEXTURE_SLOTS:
        img = getattr(mat, attr, None)
        if img is not None and isinstance(img, Image.Image):
            slot_data = slots.get(attr)
            if not slot_data or slot_data.get("format") not in SUPPORTED_TEXTURE_FORMATS or not slot_data.get("raw_bytes"):
                raise PipelineAbort(
                    f"No source format / encoded bytes recorded for the mesh's {attr}: it cannot be passed through"
                )
            img.format = slot_data["format"]
            raw_bytes = slot_data["raw_bytes"]

            def make_fast_save(data_bytes):
                def fast_save(f, format=None, **kwargs):
                    f.write(data_bytes)
                return fast_save
            img.save = make_fast_save(raw_bytes)
            img._fast_save_data = raw_bytes
            img._is_bitstream_passthrough = True


def optimize_mesh_texture_for_export(
    mesh: trimesh.Trimesh,
    preferred_format: Optional[str] = None,
    jpeg_quality: int = 99
) -> Optional[Image.Image]:
    """
    Optimizes the mesh texture before trimesh.exchange.gltf.export_glb to prevent
    uncompressed PNG bloat.
    - 'ORIGINAL' / 'PASSTHROUGH' (or no preferred_format on a texture marked _is_bitstream_passthrough)
      keeps the original raw bytes untouched (Bit-for-Bit Lossless); without raw bytes: PipelineAbort.
    - 'JPEG' saves ultra-high quality JPEG (quality=99, subsampling=0 for 4:4:4 chroma, optimize=True)
      with fast_save; a texture with real transparency cannot be JPEG: PipelineAbort.
    - 'PNG' optimizes PNG compression (optimize=True, compress_level=9) with fast_save.
    - No preferred_format: PNG with an active alpha channel, JPEG otherwise.
    Any other format, or a mesh without a base colour texture: PipelineAbort.
    Returns the optimized PIL Image object.
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None or not isinstance(img, Image.Image):
        raise PipelineAbort("The mesh has no base colour texture to export")

    pref_upper = (preferred_format or "").upper()
    if pref_upper in ("ORIGINAL", "PASSTHROUGH") or (not pref_upper and getattr(img, "_is_bitstream_passthrough", False)):
        if not getattr(img, "_fast_save_data", None):
            raise PipelineAbort(
                f"Texture format {preferred_format or 'ORIGINAL'} requested, but the original encoded bytes "
                f"of the base colour texture are not available"
            )
        return img
    if pref_upper and pref_upper not in ("JPEG", "JPG", "PNG"):
        raise PipelineAbort(f"Unsupported texture export format {preferred_format!r} (expected ORIGINAL, JPEG or PNG)")

    # Check if texture has an active alpha channel with true transparency
    has_alpha = False
    if img.mode in ("RGBA", "LA"):
        alpha_channel = np.asarray(img)[..., -1]
        if np.any(alpha_channel < 255):
            has_alpha = True
    elif img.mode == "P" and "transparency" in img.info:
        has_alpha = True

    # Determine target format:
    # 1. If preferred_format is specified, respect it (JPEG cannot hold an active alpha channel).
    # 2. If texture has an active alpha channel (RGBA/LA with transparency), it MUST be PNG.
    # 3. If texture has NO active alpha channel (RGB or opaque RGBA), save as high-quality JPEG (quality=99, subsampling=0).
    if pref_upper:
        target_fmt = pref_upper
        if target_fmt in ("JPEG", "JPG") and has_alpha:
            raise PipelineAbort("JPEG export requested, but the base colour texture has an alpha channel with transparency")
    elif has_alpha:
        target_fmt = "PNG"
    else:
        target_fmt = "JPEG"

    def make_fast_save(data_bytes: bytes):
        def fast_save(f, format=None, **kwargs):
            if isinstance(f, (str, Path)) or hasattr(f, "__fspath__"):
                with open(f, "wb") as fp:
                    fp.write(data_bytes)
            else:
                f.write(data_bytes)
        return fast_save

    # Image.Image.save: a passthrough image's own save() only writes its original bytes
    if target_fmt in ("JPEG", "JPG"):
        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        buf = io.BytesIO()
        Image.Image.save(rgb_img, buf, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
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
        Image.Image.save(img, buf, format="PNG", optimize=True, compress_level=9)
        data = buf.getvalue()

        img.format = "PNG"
        img.save = make_fast_save(data)
        img._fast_save_data = data

        if hasattr(mat, "baseColorTexture"):
            mat.baseColorTexture = img
        if hasattr(mat, "image"):
            mat.image = img
        return img

