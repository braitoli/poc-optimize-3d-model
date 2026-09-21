"""
palette.py

Dominant color palette extraction and glTF extras embedding.
"""

from typing import Dict, Any, Optional, List
import json
import struct
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

from optimizer.core.errors import PipelineAbort


def extract_palette(
    image: Optional[Image.Image] = None,
    sample_pixels: Optional[np.ndarray] = None,
    n_colors: int = 10,
    min_alpha: int = 10
) -> Dict[str, Any]:
    """
    Extracts dominant color palette using K-Means clustering.
    Uses sample_pixels directly when given (to avoid empty atlas backgrounds), else the image.
    Raises PipelineAbort for wrongly shaped sample_pixels, no usable pixels or a K-Means failure.
    """
    n_colors = max(4, min(16, int(n_colors)))
    pixels = None

    if sample_pixels is not None:
        p = np.asarray(sample_pixels)
        if p.ndim != 2 or p.shape[1] not in (3, 4):
            raise PipelineAbort(f"Palette sample pixels must be an (N, 3) or (N, 4) array, got shape {p.shape}")
        if p.shape[1] == 4:
            mask = p[:, 3] >= min_alpha
            pixels = p[mask, :3].astype(np.float32)
        else:
            pixels = p[:, :3].astype(np.float32)
    elif image is not None:
        pil_img = image.copy()
        if pil_img.width > 256 or pil_img.height > 256:
            pil_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        arr = np.asarray(pil_img.convert("RGB"))
        pixels = arr.reshape(-1, 3).astype(np.float32)

    if pixels is None or len(pixels) == 0:
        raise PipelineAbort(
            "No usable pixels for the colour palette (no image or sample pixels given, "
            f"or every sample pixel has alpha < {min_alpha})"
        )

    # Filter out near-pure-black margins
    non_black = ~np.all(pixels <= 5, axis=1)
    if np.any(non_black):
        pixels = pixels[non_black]

    # Subsample if pixel count is large for fast, robust K-Means clustering
    max_samples = 8000
    if len(pixels) > max_samples:
        step = max(1, len(pixels) // max_samples)
        kmeans_pixels = pixels[::step]
    else:
        kmeans_pixels = pixels

    # K-Means clustering; a small model (one sample pixel per vertex) can have fewer pixels than
    # colours, and "points" init draws k distinct pixels: k is at most the pixel count
    k = min(n_colors, len(kmeans_pixels))
    try:
        centroids, labels = kmeans2(kmeans_pixels, k, minit="points", iter=15)
    except Exception as err:
        raise PipelineAbort(f"K-Means colour palette clustering failed on {len(kmeans_pixels)} pixels: {err}") from err
    counts = np.bincount(labels, minlength=len(centroids))
    order = np.argsort(counts)[::-1]

    palette = []
    details = []
    total_p = len(kmeans_pixels)

    for idx in order:
        c = np.clip(np.round(centroids[idx]), 0, 255).astype(int)
        hex_val = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        if hex_val not in palette:
            palette.append(hex_val)
            details.append({
                "hex": hex_val,
                "rgb": [int(c[0]), int(c[1]), int(c[2])],
                "weight": round(float(counts[idx]) / max(total_p, 1), 4)
            })

    return {
        "palette": palette,
        "primaryColor": palette[0],
        "paletteDetails": details
    }


def embed_gltf_extras(glb_bytes: bytes, extras: Dict[str, Any]) -> bytes:
    """Embeds dictionary payload into glTF root extras in a binary GLB file.
    Raises PipelineAbort if the bytes are not a GLB whose first chunk is JSON."""
    if len(glb_bytes) < 20:
        raise PipelineAbort(f"Cannot embed glTF extras: {len(glb_bytes)} bytes is too short for a GLB")

    magic, version, _ = struct.unpack("<III", glb_bytes[:12])
    if magic != 0x46546C67:
        raise PipelineAbort("Cannot embed glTF extras: the data is not a GLB (bad magic)")

    json_len, json_type = struct.unpack("<II", glb_bytes[12:20])
    if json_type != 0x4E4F534A:
        raise PipelineAbort("Cannot embed glTF extras: the GLB's first chunk is not JSON")

    try:
        gltf = json.loads(glb_bytes[20:20 + json_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise PipelineAbort(f"Cannot embed glTF extras: the GLB's JSON chunk is invalid ({err})") from err
    
    if "extras" not in gltf:
        gltf["extras"] = {}
    gltf["extras"].update(extras)

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + json_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    header = struct.pack("<III", magic, version, new_total_len)
    chunk0 = struct.pack("<II", len(new_json_bytes), json_type)
    return header + chunk0 + new_json_bytes + bin_chunk
