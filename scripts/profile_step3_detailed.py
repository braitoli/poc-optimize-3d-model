#!/usr/bin/env python3
"""
profile_step3_detailed.py

Profiles Step 3 of the 3D optimization pipeline on Dinoki and Gravilux.
Measures time and memory for each sub-step:
- Direct resample mode (rechart_uv=False)
- High-density xatlas rechart mode (rechart_uv=True)
  - xatlas mesh add & generate
  - Atlas extraction
  - Rasterization (_rasterize_uv_atlas)
  - Bilinear texture sampling
  - Dilation (ndimage.distance_transform_edt)
  - Trimesh reconstruction & export
"""

import os
import sys
import time
import tracemalloc
import resource
import gc
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import trimesh
import xatlas
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimizer.core.uv_baker import (
    _rasterize_uv_atlas,
    _sample_texture_bilinear,
    dilate_texture,
    direct_resample_texture,
    rechart_and_bake_high_density,
    compute_uv_metrics
)
from optimizer.core.texture_utils import clamp_target_resolution, optimize_mesh_texture_for_export
from optimizer.core.glb_utils import set_doublesided_material


def get_mem_mb() -> float:
    # On macOS ru_maxrss is bytes; on Linux it is KB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def load_model_mesh(path: Path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        # Pick the main geometry
        for geom in scene.geometry.values():
            if hasattr(geom, "faces") and len(geom.faces) > 0:
                return geom
        raise ValueError(f"No geometry found in {path}")
    return scene


def profile_direct_mode(mesh: trimesh.Trimesh, target_res: int, label: str):
    print(f"\n--- [DIRECT MODE] Profiling {label} (target_res={target_res}) ---")
    gc.collect()
    t0 = time.perf_counter()
    m0 = get_mem_mb()

    # Source texture
    orig_mat = getattr(mesh.visual, "material", None)
    source_img = getattr(orig_mat, "baseColorTexture", None) or getattr(orig_mat, "image", None)
    if source_img is None:
        source_img = Image.new("RGB", (target_res, target_res), (200, 200, 200))
    print(f"  Source image size: {source_img.size}, mode: {source_img.mode}")

    # Step 1: clamp res
    eff_res = clamp_target_resolution(target_res, source_img.size)

    # Step 2: resize
    t_resize_start = time.perf_counter()
    img_rgb = source_img.convert("RGB")
    resampled = img_rgb.resize((eff_res, eff_res), Image.Resampling.LANCZOS)
    arr = np.array(resampled, dtype=np.uint8)
    t_resize = time.perf_counter() - t_resize_start

    # Step 3: mask & dilate
    t_dilate_start = time.perf_counter()
    is_black = np.all(arr <= 2, axis=-1)
    is_covered = ~is_black
    dilated_arr = dilate_texture(arr, is_covered, padding=16)
    t_dilate = time.perf_counter() - t_dilate_start

    # Step 4: direct_resample_texture full call
    t_full_start = time.perf_counter()
    out_mesh, clean_pil = direct_resample_texture(mesh, source_image=source_img, target_res=target_res, dilation_padding=16)
    t_full = time.perf_counter() - t_full_start

    # Step 5: GLB Export
    t_export_start = time.perf_counter()
    s_scene = trimesh.Scene({"Model": out_mesh})
    s_bytes = trimesh.exchange.gltf.export_glb(s_scene, include_normals=True)
    s_bytes = set_doublesided_material(s_bytes)
    t_export = time.perf_counter() - t_export_start

    t_total = time.perf_counter() - t0
    m_peak = get_mem_mb()

    print(f"  Resize (Lanczos): {t_resize:.4f}s")
    print(f"  Dilate (16px):    {t_dilate:.4f}s")
    print(f"  Full direct_resample: {t_full:.4f}s")
    print(f"  GLB Export:       {t_export:.4f}s ({len(s_bytes)/(1024*1024):.2f} MB)")
    print(f"  Total Direct:     {t_total:.4f}s | Max RSS: {m_peak:.1f} MB (Delta: +{m_peak - m0:.1f} MB)")

    return {
        "resize_sec": t_resize,
        "dilate_sec": t_dilate,
        "full_direct_sec": t_full,
        "export_sec": t_export,
        "total_sec": t_total,
        "peak_rss_mb": m_peak,
        "delta_rss_mb": m_peak - m0,
        "glb_size_mb": len(s_bytes)/(1024*1024)
    }


def main():
    dinoki_path = REPO_ROOT / "examples" / "models" / "dinoki_raw.glb"
    gravilux_path = REPO_ROOT / "examples" / "models" / "gravilux_raw.glb"

    print("Loading meshes...")
    m_dino = load_model_mesh(dinoki_path)
    print(f"Dinoki loaded: {len(m_dino.vertices)} verts, {len(m_dino.faces)} faces")

    # Direct mode test for Dinoki
    profile_direct_mode(m_dino, 1024, "Dinoki (1024)")

    m_grav = load_model_mesh(gravilux_path)
    print(f"Gravilux loaded: {len(m_grav.vertices)} verts, {len(m_grav.faces)} faces")

    # Direct mode test for Gravilux
    profile_direct_mode(m_grav, 1024, "Gravilux (1024)")
    profile_direct_mode(m_grav, 2048, "Gravilux (2048)")


if __name__ == "__main__":
    main()
