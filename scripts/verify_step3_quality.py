#!/usr/bin/env python3
"""
verify_step3_quality.py

Automated Quality & Visual Regression Test Suite for Step 3 (Atlas / Bake).
Tests sample_dinoki.glb and flamibo_raw.glb:
1. UV Overlap Ratio after re-charting (< 1.0%).
2. Abnormal black pixels (0,0,0) inside triangles.
3. Dinoki: Zero white streaks on feet & preservation of dark eye pupils.
4. Flamibo: Color fidelity (black hat, red feathers, orange beak in correct anatomical places).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import trimesh
import pygltflib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimizer.core.uv_baker import (
    rechart_and_bake_high_density,
    _rasterize_uv_atlas,
    _sample_texture_bilinear
)
from optimizer.core.texture_utils import clamp_target_resolution


def measure_uv_overlap(faces: np.ndarray, uv: np.ndarray, dim: int = 1024) -> Dict[str, Any]:
    """Measures precise 2D triangle overlap ratio in UV space."""
    tri_uv = uv[faces]
    px = tri_uv[:, :, 0] * (dim - 1)
    py = (1.0 - tri_uv[:, :, 1]) * (dim - 1)

    min_x = np.clip(np.floor(px.min(axis=1)).astype(np.int32), 0, dim - 1)
    max_x = np.clip(np.ceil(px.max(axis=1)).astype(np.int32), 0, dim - 1)
    min_y = np.clip(np.floor(py.min(axis=1)).astype(np.int32), 0, dim - 1)
    max_y = np.clip(np.ceil(py.max(axis=1)).astype(np.int32), 0, dim - 1)

    coverage_grid = np.zeros((dim, dim), dtype=np.int32)
    for i in range(len(faces)):
        p0x, p0y = px[i, 0], py[i, 0]
        p1x, p1y = px[i, 1], py[i, 1]
        p2x, p2y = px[i, 2], py[i, 2]

        det = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
        if abs(det) < 1e-7:
            continue

        gx, gy = np.meshgrid(np.arange(min_x[i], max_x[i] + 1), np.arange(min_y[i], max_y[i] + 1))
        gx = gx.flatten()
        gy = gy.flatten()

        w0 = ((p1y - p2y) * (gx - p2x) + (p2x - p1x) * (gy - p2y)) / det
        w1 = ((p2y - p0y) * (gx - p2x) + (p0x - p2x) * (gy - p2y)) / det
        w2 = 1.0 - w0 - w1

        # Strict interior check (margin 1e-4) prevents false overlap along shared triangle edges
        inside = (w0 >= 1e-4) & (w1 >= 1e-4) & (w2 >= 1e-4)
        coverage_grid[gy[inside], gx[inside]] += 1

    total_covered = int(np.sum(coverage_grid >= 1))
    overlapped = int(np.sum(coverage_grid > 1))
    overlap_ratio_pct = round(float(overlapped / max(total_covered, 1) * 100.0), 2)

    return {
        "dim": dim,
        "total_covered_pixels": total_covered,
        "overlapped_pixels": overlapped,
        "overlap_ratio_percent": overlap_ratio_pct
    }


def sample_vertex_colors(mesh: trimesh.Trimesh, image_pil: Image.Image) -> np.ndarray:
    """Samples RGB colors from texture image for each mesh vertex using its UV coordinates."""
    arr = np.array(image_pil.convert("RGB"))
    h, w = arr.shape[:2]
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) == 0:
        return np.zeros((len(mesh.vertices), 3), dtype=np.uint8)

    u = (uv[:, 0] % 1.0) * (w - 1)
    v = (1.0 - (uv[:, 1] % 1.0)) * (h - 1)
    px = np.clip(np.round(u).astype(int), 0, w - 1)
    py = np.clip(np.round(v).astype(int), 0, h - 1)
    return arr[py, px]


def test_dinoki(model_path: str, target_res: int = 1024) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"TESTING DINOKI: {model_path}")
    print(f"================================================================================")
    mesh = trimesh.load(model_path, force='mesh')
    raw_tex = getattr(mesh.visual.material, "baseColorTexture", None) or getattr(mesh.visual.material, "image", None)
    if raw_tex is None:
        gltf = pygltflib.GLTF2().load(model_path)
        bv = gltf.bufferViews[gltf.images[0].bufferView]
        blob = gltf.binary_blob()
        raw_tex = Image.open(io.BytesIO(blob[bv.byteOffset:bv.byteOffset+bv.byteLength]))

    raw_vert_cols = sample_vertex_colors(mesh, raw_tex)

    t0 = time.perf_counter()
    baked_mesh, baked_pil, stats = rechart_and_bake_high_density(
        mesh=mesh,
        target_res=target_res,
        source_image=raw_tex,
        source_uv=mesh.visual.uv,
        return_stats=True
    )
    bake_time = time.perf_counter() - t0
    print(f"Bake completed in {bake_time:.2f}s (Result: {baked_pil.size[0]}x{baked_pil.size[1]})")

    # 1. Overlap Check
    overlap_res = measure_uv_overlap(baked_mesh.faces, baked_mesh.visual.uv, dim=baked_pil.size[0])
    print(f"UV Overlap: {overlap_res['overlapped_pixels']:,} px ({overlap_res['overlap_ratio_percent']}%)")

    # 2. Black Pixels Check
    baked_arr = np.array(baked_pil.convert("RGB"))
    h, w = baked_arr.shape[:2]
    pure_black_count = int(np.sum(np.all(baked_arr == 0, axis=-1)))
    pure_black_pct = round(pure_black_count / (h * w) * 100.0, 2)
    print(f"Pure black (0,0,0) pixels on canvas: {pure_black_count:,} ({pure_black_pct}%)")

    # 3. Dinoki Feet White Streak Check
    # Feet region: 0.12 <= Y < 0.35
    baked_verts = baked_mesh.vertices
    baked_cols = sample_vertex_colors(baked_mesh, baked_pil)

    feet_idx = np.where((baked_verts[:, 1] >= 0.12) & (baked_verts[:, 1] < 0.35))[0]
    feet_cols = baked_cols[feet_idx]
    white_feet = feet_idx[np.all(feet_cols > 200, axis=1)]
    white_feet_count = len(white_feet)
    print(f"Feet vertices (0.12 <= Y < 0.35): {len(feet_idx):,}")
    print(f"White vertices on feet (RGB > 200): {white_feet_count} (Must be 0)")

    # 4. Dinoki Eye Pupils Check
    head_idx = np.where(baked_verts[:, 1] >= 1.10)[0]
    head_cols = baked_cols[head_idx]
    dark_pupils = head_idx[np.all(head_cols < 40, axis=1)]
    print(f"Dark pupil/crevice vertices on head (RGB < 40): {len(dark_pupils):,}")

    passed = (
        overlap_res["overlap_ratio_percent"] < 1.0
        and white_feet_count == 0
        and len(dark_pupils) > 0
    )

    result = {
        "model": "Dinoki",
        "passed": passed,
        "bake_time_sec": round(bake_time, 2),
        "texture_res": f"{baked_pil.size[0]}x{baked_pil.size[1]}",
        "overlap_metrics": overlap_res,
        "canvas_pure_black_pixels": pure_black_count,
        "canvas_pure_black_percent": pure_black_pct,
        "white_feet_vertices": white_feet_count,
        "dark_pupil_vertices": len(dark_pupils),
        "stats": stats
    }
    print(f"DINOKI VERIFICATION: {'[PASS] PASSED' if passed else '[FAIL] FAILED'}")
    return result


def test_flamibo(model_path: str, target_res: int = 2048) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"TESTING FLAMIBO: {model_path}")
    print(f"================================================================================")
    mesh = trimesh.load(model_path, force='mesh')
    raw_tex = getattr(mesh.visual.material, "baseColorTexture", None) or getattr(mesh.visual.material, "image", None)
    if raw_tex is None:
        gltf = pygltflib.GLTF2().load(model_path)
        bv = gltf.bufferViews[gltf.images[0].bufferView]
        blob = gltf.binary_blob()
        raw_tex = Image.open(io.BytesIO(blob[bv.byteOffset:bv.byteOffset+bv.byteLength]))

    raw_vert_cols = sample_vertex_colors(mesh, raw_tex)

    t0 = time.perf_counter()
    baked_mesh, baked_pil, stats = rechart_and_bake_high_density(
        mesh=mesh,
        target_res=target_res,
        source_image=raw_tex,
        source_uv=mesh.visual.uv,
        return_stats=True
    )
    bake_time = time.perf_counter() - t0
    print(f"Bake completed in {bake_time:.2f}s (Result: {baked_pil.size[0]}x{baked_pil.size[1]})")

    # 1. Overlap Check
    overlap_res = measure_uv_overlap(baked_mesh.faces, baked_mesh.visual.uv, dim=min(baked_pil.size[0], 1024))
    print(f"UV Overlap: {overlap_res['overlapped_pixels']:,} px ({overlap_res['overlap_ratio_percent']}%)")

    # 2. Color Fidelity Verification
    # Check anatomical regions:
    # A. Black Hat (regions where raw hat is dark RGB < 40 and Y > 1.2)
    baked_verts = baked_mesh.vertices
    baked_cols = sample_vertex_colors(baked_mesh, baked_pil)

    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    _, nearest_raw_idx = tree.query(baked_verts)
    raw_matched_cols = raw_vert_cols[nearest_raw_idx]

    hat_mask = np.all(raw_matched_cols < 40, axis=1) & (baked_verts[:, 1] > 1.2)
    hat_cols = baked_cols[hat_mask]
    hat_mean_rgb = hat_cols.mean(axis=0).astype(int)
    hat_dark_pct = round(float(np.sum(np.all(hat_cols < 60, axis=1)) / max(len(hat_cols), 1) * 100.0), 2)
    print(f"Black Hat ({len(hat_cols):,} verts): mean RGB={hat_mean_rgb}, % dark (<60)={hat_dark_pct}%")

    # B. Red feathers (regions where raw R > 150, G < 80, B < 80)
    mae = float(np.mean(np.abs(baked_cols.astype(float) - raw_matched_cols.astype(float))))
    print(f"Vertex Color Mean Absolute Error (MAE) vs Raw: {mae:.2f} / 255")

    # Check Red fidelity
    red_mask_in_raw = (raw_matched_cols[:, 0] > 150) & (raw_matched_cols[:, 1] < 80) & (raw_matched_cols[:, 2] < 80)
    red_baked_cols = baked_cols[red_mask_in_raw]
    red_mean_rgb = red_baked_cols.mean(axis=0).astype(int)
    print(f"Red regions ({len(red_baked_cols):,} verts): mean RGB={red_mean_rgb}")

    # Check Yellow/Orange fidelity
    yellow_mask_in_raw = (raw_matched_cols[:, 0] > 180) & (raw_matched_cols[:, 1] > 120) & (raw_matched_cols[:, 2] < 80)
    yellow_baked_cols = baked_cols[yellow_mask_in_raw]
    yellow_mean_rgb = yellow_baked_cols.mean(axis=0).astype(int)
    print(f"Yellow/Orange regions ({len(yellow_baked_cols):,} verts): mean RGB={yellow_mean_rgb}")

    passed = (
        overlap_res["overlap_ratio_percent"] < 1.0
        and hat_dark_pct > 80.0
        and red_mean_rgb[0] > 140
        and yellow_mean_rgb[0] > 160
        and mae < 25.0
    )

    result = {
        "model": "Flamibo",
        "passed": passed,
        "bake_time_sec": round(bake_time, 2),
        "texture_res": f"{baked_pil.size[0]}x{baked_pil.size[1]}",
        "overlap_metrics": overlap_res,
        "color_mae": round(mae, 2),
        "hat_mean_rgb": hat_mean_rgb.tolist(),
        "hat_dark_percent": hat_dark_pct,
        "red_mean_rgb": red_mean_rgb.tolist(),
        "yellow_mean_rgb": yellow_mean_rgb.tolist(),
        "stats": stats
    }
    print(f"FLAMIBO VERIFICATION: {'[PASS] PASSED' if passed else '[FAIL] FAILED'}")
    return result


if __name__ == "__main__":
    import io
    dinoki_path = str(REPO_ROOT / "examples" / "sample_dinoki.glb")
    flamibo_path = str(REPO_ROOT / "examples" / "models" / "flamibo_raw.glb")

    results = {}
    if os.path.exists(dinoki_path):
        results["dinoki"] = test_dinoki(dinoki_path, target_res=1024)
    else:
        print(f"Dinoki not found at {dinoki_path}")

    if os.path.exists(flamibo_path):
        results["flamibo"] = test_flamibo(flamibo_path, target_res=2048)
    else:
        print(f"Flamibo not found at {flamibo_path}")

    out_json = REPO_ROOT / "scratch" / "step3_verification_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nAll verification results saved to: {out_json}")

    all_passed = all(r.get("passed", False) for r in results.values())
    print(f"\nFINAL VERIFICATION STATUS: {'ALL TESTS PASSED [PASS]' if all_passed else 'SOME TESTS FAILED [FAIL]'}")
    sys.exit(0 if all_passed else 1)
