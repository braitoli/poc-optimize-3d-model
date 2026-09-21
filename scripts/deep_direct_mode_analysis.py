#!/usr/bin/env python3
"""
deep_direct_mode_analysis.py

Rigorous Empirical Measurement and Comparison of Direct Master UV Mode (Direct Mode)
between Step 0 (Raw) and Step 3 (Texture Baked) for Flamibo and Dinoki models:
1. Texture dimensions, format, file size, VRAM footprint, PSNR/SSIM, eye/pattern crops.
2. PBR Material parameters inspection via pygltflib JSON structure.
3. Vertex normals analysis: count, length, angular deviation distribution, face flipping correlation.
4. Offscreen 3D rendered view evaluation: MAE, MSE, PSNR, SSIM, and side-by-side composite images.
"""

import os
import sys
import io
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
import pygltflib
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    """Computes Mean Structural Similarity Index (SSIM) between two images."""
    if im1.ndim == 3:
        return float(np.mean([compute_ssim(im1[..., c], im2[..., c]) for c in range(im1.shape[2])]))
    
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    sigma = 1.5

    mu1 = gaussian_filter(im1, sigma)
    mu2 = gaussian_filter(im2, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(im1 * im1, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(im2 * im2, sigma) - mu2_sq
    sigma12 = gaussian_filter(im1 * im2, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def compute_image_metrics(im1: np.ndarray, im2: np.ndarray) -> Dict[str, float]:
    """Computes MAE, MSE, PSNR, and SSIM between two RGB images of identical shape."""
    assert im1.shape == im2.shape, f"Shape mismatch: {im1.shape} vs {im2.shape}"
    arr1 = im1.astype(np.float64)
    arr2 = im2.astype(np.float64)
    diff = np.abs(arr1 - arr2)
    mae = float(np.mean(diff))
    mse = float(np.mean((arr1 - arr2) ** 2))
    psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse))) if mse > 1e-10 else 999.0
    ssim = compute_ssim(arr1, arr2)
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4)
    }


def extract_glb_texture_data(glb_path: str) -> Dict[str, Any]:
    """Extracts low-level image and bufferView info from GLB using pygltflib."""
    gltf = pygltflib.GLTF2().load(glb_path)
    if not gltf.images:
        return {}
    img_meta = gltf.images[0]
    bv = gltf.bufferViews[img_meta.bufferView] if img_meta.bufferView is not None else None
    
    # Extract binary bytes
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    tex_img = mesh.visual.material.baseColorTexture if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "baseColorTexture") else None
    
    return {
        "mime_type": img_meta.mimeType,
        "byte_length": bv.byteLength if bv else None,
        "pil_image": tex_img,
        "width": tex_img.width if tex_img else None,
        "height": tex_img.height if tex_img else None,
        "mode": tex_img.mode if tex_img else None
    }


def inspect_pbr_materials(glb_path: str) -> Dict[str, Any]:
    """Inspects all PBR material fields via pygltflib."""
    gltf = pygltflib.GLTF2().load(glb_path)
    mat_list = []
    for m in gltf.materials:
        pbr = m.pbrMetallicRoughness
        mat_info = {
            "name": m.name,
            "doubleSided": m.doubleSided,
            "alphaMode": m.alphaMode,
            "alphaCutoff": m.alphaCutoff,
            "baseColorFactor": pbr.baseColorFactor if pbr else None,
            "roughnessFactor": pbr.roughnessFactor if pbr else None,
            "metallicFactor": pbr.metallicFactor if pbr else None,
            "hasBaseColorTexture": (pbr.baseColorTexture is not None) if pbr else False,
            "hasMetallicRoughnessTexture": (pbr.metallicRoughnessTexture is not None) if pbr else False,
            "hasNormalTexture": (m.normalTexture is not None),
            "hasOcclusionTexture": (m.occlusionTexture is not None),
            "hasEmissiveTexture": (m.emissiveTexture is not None),
            "emissiveFactor": m.emissiveFactor
        }
        mat_list.append(mat_info)
    return {
        "materials_count": len(gltf.materials),
        "materials": mat_list
    }


def inspect_vertex_normals(step0_path: str, step3_path: str) -> Dict[str, Any]:
    """Compares vertex normals between Step 0 and Step 3."""
    m0 = trimesh.load(step0_path, force="mesh", process=False)
    m3 = trimesh.load(step3_path, force="mesh", process=False)
    
    n0 = getattr(m0, "vertex_normals", None)
    n3 = getattr(m3, "vertex_normals", None)
    
    count0 = len(n0) if n0 is not None else 0
    count3 = len(n3) if n3 is not None else 0
    
    if n0 is None or n3 is None or len(n0) != len(n3):
        return {
            "step0_count": count0,
            "step3_count": count3,
            "count_preserved": count0 == count3,
            "error": "Count mismatch or missing normals"
        }
    
    norm_len0 = np.linalg.norm(n0, axis=1, keepdims=True)
    norm_len3 = np.linalg.norm(n3, axis=1, keepdims=True)
    
    n0_unit = n0 / np.clip(norm_len0, 1e-9, None)
    n3_unit = n3 / np.clip(norm_len3, 1e-9, None)
    
    dots = np.sum(n0_unit * n3_unit, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    
    total = len(angles_deg)
    return {
        "step0_count": count0,
        "step3_count": count3,
        "count_preserved": count0 == count3,
        "mean_angular_deviation_deg": round(float(np.mean(angles_deg)), 4),
        "median_angular_deviation_deg": round(float(np.median(angles_deg)), 4),
        "percentile_95_deg": round(float(np.percentile(angles_deg, 95)), 4),
        "percentile_99_deg": round(float(np.percentile(angles_deg, 99)), 4),
        "max_angular_deviation_deg": round(float(np.max(angles_deg)), 4),
        "exact_zero_deviation_count": int(np.sum(angles_deg == 0.0)),
        "exact_zero_deviation_percent": round(float(np.sum(angles_deg == 0.0) / total * 100.0), 2),
        "deviation_under_0_01_deg_count": int(np.sum(angles_deg < 0.01)),
        "deviation_under_0_01_deg_percent": round(float(np.sum(angles_deg < 0.01) / total * 100.0), 2),
        "deviation_under_1_0_deg_count": int(np.sum(angles_deg < 1.0)),
        "deviation_under_1_0_deg_percent": round(float(np.sum(angles_deg < 1.0) / total * 100.0), 2),
        "flipped_normals_over_90_deg_count": int(np.sum(angles_deg > 90.0)),
        "flipped_normals_over_90_deg_percent": round(float(np.sum(angles_deg > 90.0) / total * 100.0), 2),
    }


def inspect_uv_coordinates(step0_path: str, step3_path: str) -> Dict[str, Any]:
    """Compares UV coordinates between Step 0 and Step 3."""
    m0 = trimesh.load(step0_path, force="mesh", process=False)
    m3 = trimesh.load(step3_path, force="mesh", process=False)
    uv0 = getattr(m0.visual, "uv", None)
    uv3 = getattr(m3.visual, "uv", None)
    if uv0 is None or uv3 is None or len(uv0) != len(uv3):
        return {"error": "UV coordinates mismatch"}
    diff = np.abs(uv0 - uv3)
    return {
        "uv_count": len(uv0),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "bitwise_identical": bool(np.all(uv0 == uv3))
    }


def render_views_open3d(glb_path: str, views: Dict[str, Tuple], output_dir: Path, prefix: str) -> Dict[str, str]:
    """Offscreen rendering of standardized views using Open3D with Apple Metal backend."""
    import open3d as o3d
    
    renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 1024)
    scene = renderer.scene
    scene.set_background([0.15, 0.15, 0.18, 1.0])
    
    mesh = o3d.io.read_triangle_mesh(glb_path, enable_post_processing=True)
    mesh.compute_vertex_normals()
    
    temp_tex_path = output_dir / f"{prefix}_temp_tex.png"
    if mesh.textures and len(mesh.textures) > 0:
        o3d.io.write_image(str(temp_tex_path), mesh.textures[0])
        tex = o3d.io.read_image(str(temp_tex_path))
        temp_tex_path.unlink(missing_ok=True)
    else:
        tex = None
        
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    if tex is not None:
        mat.albedo_img = tex
    mat.base_roughness = 0.5
    mat.base_metallic = 0.0
    
    scene.add_geometry("mesh", mesh, mat)
    
    rendered_files = {}
    for name, (center, eye, up, fov) in views.items():
        renderer.setup_camera(fov, center, eye, up)
        img = renderer.render_to_image()
        out_file = output_dir / f"{prefix}_{name}.png"
        o3d.io.write_image(str(out_file), img)
        rendered_files[name] = str(out_file)
        
    return rendered_files


def run_model_analysis(
    name: str,
    step0_path: str,
    step3_path: str,
    eye_box_raw: Tuple[int, int, int, int], # (x0, y0, x1, y1) in raw texture
    views_config: Dict[str, Tuple],
    scratch_dir: Path
) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"📊 EXECUTING DIRECT MODE FIDELITY ANALYSIS: {name}")
    print(f"   Step 0: {step0_path}")
    print(f"   Step 3: {step3_path}")
    print(f"================================================================================")
    
    out_dir = scratch_dir / name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Texture extraction
    tex0 = extract_glb_texture_data(step0_path)
    tex3 = extract_glb_texture_data(step3_path)
    
    pil0: Image.Image = tex0["pil_image"]
    pil3: Image.Image = tex3["pil_image"]
    
    w0, h0 = pil0.size
    w3, h3 = pil3.size
    scale_factor = float(w3) / float(w0)
    
    # Uncompressed VRAM footprint
    vram0_rgb_mb = (w0 * h0 * 3) / (1024 * 1024)
    vram3_rgb_mb = (w3 * h3 * 3) / (1024 * 1024)
    vram_saved_pct = (1.0 - (vram3_rgb_mb / vram0_rgb_mb)) * 100.0
    
    # Save extracted textures
    tex0_png_path = out_dir / f"{name.lower()}_tex_step0.png"
    tex3_png_path = out_dir / f"{name.lower()}_tex_step3.png"
    pil0.save(tex0_png_path)
    pil3.save(tex3_png_path)
    
    # Texture metrics: Resample pil3 back to pil0 size with Lanczos to compare full texture
    pil3_upsampled = pil3.resize((w0, h0), Image.Resampling.LANCZOS)
    arr0 = np.array(pil0.convert("RGB"))
    arr3_up = np.array(pil3_upsampled.convert("RGB"))
    tex_metrics_full = compute_image_metrics(arr0, arr3_up)
    
    # Texture crop on eye / pupil / retina
    rx0, ry0, rx1, ry1 = eye_box_raw
    crop0 = pil0.crop((rx0, ry0, rx1, ry1))
    
    sx0 = int(round(rx0 * scale_factor))
    sy0 = int(round(ry0 * scale_factor))
    sx1 = int(round(rx1 * scale_factor))
    sy1 = int(round(ry1 * scale_factor))
    crop3 = pil3.crop((sx0, sy0, sx1, sy1))
    
    # Upscale crop3 to match crop0 for direct pixel comparison
    crop3_up = crop3.resize(crop0.size, Image.Resampling.LANCZOS)
    crop0_arr = np.array(crop0.convert("RGB"))
    crop3_arr = np.array(crop3_up.convert("RGB"))
    eye_metrics = compute_image_metrics(crop0_arr, crop3_arr)
    
    # Save eye crop side-by-side composite
    cw, ch = crop0.size
    comp_eye = Image.new("RGB", (cw * 2 + 10, ch + 40), (24, 24, 28))
    comp_eye.paste(crop0, (0, 40))
    comp_eye.paste(crop3_up, (cw + 10, 40))
    draw_eye = ImageDraw.Draw(comp_eye)
    draw_eye.text((5, 10), f"Step 0 Raw ({cw}x{ch})", fill=(220, 220, 220))
    draw_eye.text((cw + 15, 10), f"Step 3 Direct Mode (Upscaled to {cw}x{ch})", fill=(100, 255, 150))
    eye_comp_path = out_dir / f"{name.lower()}_eye_comparison.png"
    comp_eye.save(eye_comp_path)
    
    # 2. Material PBR inspection
    pbr0 = inspect_pbr_materials(step0_path)
    pbr3 = inspect_pbr_materials(step3_path)
    
    # 3. Vertex Normals & UV
    normals_info = inspect_vertex_normals(step0_path, step3_path)
    uv_info = inspect_uv_coordinates(step0_path, step3_path)
    
    # 4. Offscreen 3D Render Views
    renders0 = render_views_open3d(step0_path, views_config, out_dir, f"{name.lower()}_s0")
    renders3 = render_views_open3d(step3_path, views_config, out_dir, f"{name.lower()}_s3")
    
    render_metrics = {}
    side_by_side_renders = {}
    for vname in views_config:
        im0_view = np.array(Image.open(renders0[vname]).convert("RGB"))
        im3_view = np.array(Image.open(renders3[vname]).convert("RGB"))
        metrics_v = compute_image_metrics(im0_view, im3_view)
        render_metrics[vname] = metrics_v
        
        # Create side-by-side view
        vw, vh = im0_view.shape[1], im0_view.shape[0]
        comp_v = Image.new("RGB", (vw * 2 + 10, vh + 50), (20, 22, 28))
        comp_v.paste(Image.fromarray(im0_view), (0, 50))
        comp_v.paste(Image.fromarray(im3_view), (vw + 10, 50))
        draw_v = ImageDraw.Draw(comp_v)
        title_text = f"{name} {vname.upper()} | Step 0 (Raw {w0}x{h0}) vs Step 3 (Direct {w3}x{h3}) | PSNR={metrics_v['psnr_db']}dB, SSIM={metrics_v['ssim']}"
        draw_v.text((15, 15), title_text, fill=(240, 240, 240))
        comp_v_path = out_dir / f"{name.lower()}_comparison_{vname}.png"
        comp_v.save(comp_v_path)
        side_by_side_renders[vname] = str(comp_v_path)
        
    result = {
        "model": name,
        "texture": {
            "step0": {
                "resolution": f"{w0}x{h0}",
                "width": w0,
                "height": h0,
                "format": tex0["mime_type"],
                "file_size_bytes": tex0["byte_length"],
                "file_size_formatted": f"{tex0['byte_length'] / 1024:.2f} KB" if tex0["byte_length"] else "N/A",
                "uncompressed_vram_mb": round(vram0_rgb_mb, 2)
            },
            "step3": {
                "resolution": f"{w3}x{h3}",
                "width": w3,
                "height": h3,
                "format": tex3["mime_type"],
                "file_size_bytes": tex3["byte_length"],
                "file_size_formatted": f"{tex3['byte_length'] / 1024:.2f} KB" if tex3["byte_length"] else "N/A",
                "uncompressed_vram_mb": round(vram3_rgb_mb, 2)
            },
            "scale_factor": round(scale_factor, 4),
            "vram_saved_percent": round(vram_saved_pct, 2),
            "full_texture_metrics": tex_metrics_full,
            "eye_region_crop": {
                "raw_box": list(eye_box_raw),
                "step3_box": [sx0, sy0, sx1, sy1],
                "metrics": eye_metrics,
                "composite_image": str(eye_comp_path)
            }
        },
        "material_pbr": {
            "step0": pbr0["materials"][0],
            "step3": pbr3["materials"][0],
            "roughness_diff": abs((pbr0["materials"][0]["roughnessFactor"] or 0) - (pbr3["materials"][0]["roughnessFactor"] or 0)),
            "metallic_diff": abs((pbr0["materials"][0]["metallicFactor"] or 0) - (pbr3["materials"][0]["metallicFactor"] or 0)),
            "double_sided_match": pbr0["materials"][0]["doubleSided"] == pbr3["materials"][0]["doubleSided"],
            "alpha_mode_match": pbr0["materials"][0]["alphaMode"] == pbr3["materials"][0]["alphaMode"]
        },
        "vertex_normals": normals_info,
        "uv_coordinates": uv_info,
        "render_views_metrics": render_metrics,
        "side_by_side_renders": side_by_side_renders
    }
    
    return result


def main():
    scratch_dir = REPO_ROOT / "scratch" / "direct_mode_experiment"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dinoki Configuration
    dinoki_views = {
        "front": ([0.0, 0.8, 0.0], [0.0, 0.8, 2.6], [0.0, 1.0, 0.0], 50.0),
        "head": ([0.0, 1.25, 0.15], [0.0, 1.25, 1.0], [0.0, 1.0, 0.0], 40.0),
        "perspective": ([0.0, 0.8, 0.0], [1.6, 1.1, 1.8], [0.0, 1.0, 0.0], 50.0)
    }
    dinoki_eye_box_raw = (800, 320, 900, 440) # (100x120) in 1536x1536
    
    dinoki_results = run_model_analysis(
        name="Dinoki",
        step0_path="workspaces/exp_direct_mode/dinoki/step_00_raw.glb",
        step3_path="workspaces/exp_direct_mode/dinoki/step_03_texture_baked.glb",
        eye_box_raw=dinoki_eye_box_raw,
        views_config=dinoki_views,
        scratch_dir=scratch_dir
    )
    
    # 2. Flamibo Configuration
    flamibo_views = {
        "front": ([0.0, 0.8, 0.0], [0.0, 0.8, 2.6], [0.0, 1.0, 0.0], 50.0),
        "helmet": ([0.0, 1.35, 0.05], [0.0, 1.35, 0.75], [0.0, 1.0, 0.0], 40.0),
        "chest": ([0.0, 0.95, 0.15], [0.0, 0.95, 1.0], [0.0, 1.0, 0.0], 45.0),
        "perspective": ([0.0, 0.8, 0.0], [1.7, 1.1, 1.9], [0.0, 1.0, 0.0], 50.0)
    }
    flamibo_eye_box_raw = (2443, 3423, 2578, 3538) # (135x115) in 4096x4096
    
    flamibo_results = run_model_analysis(
        name="Flamibo",
        step0_path="workspaces/exp_direct_mode/flamibo/step_00_raw.glb",
        step3_path="workspaces/exp_direct_mode/flamibo/step_03_texture_baked.glb",
        eye_box_raw=flamibo_eye_box_raw,
        views_config=flamibo_views,
        scratch_dir=scratch_dir
    )
    
    full_report = {
        "timestamp": "2026-09-22T00:46:00",
        "dinoki": dinoki_results,
        "flamibo": flamibo_results
    }
    
    report_path = scratch_dir / "direct_mode_fidelity_report.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(f"\n✅ Full Fidelity Report successfully generated and saved to: {report_path}")


if __name__ == "__main__":
    main()
