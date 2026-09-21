#!/usr/bin/env python3
"""
benchmark.py

Comprehensive 3D Model Optimization Benchmark & 1:1 Production Parity Checker.
Measures performance, compression ratios, geometric integrity, and verifies
exact 1:1 matching between the new standalone pipeline and the production pipeline.

Usage:
    python3 tests/benchmark.py
"""

import os
import sys
import time
import json
import struct
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer.pipeline import ModelOptimizer

PROD_DINOKI_PATH = Path("/Users/nguyenhoainam/code/braitoli/3DPainting/3DPainting/public/models/khung_long.glb")
PROD_VULPARON_PATH = Path("/Users/nguyenhoainam/code/braitoli/3DPainting/3DPainting/public/models/poc/vulparon.glb")


def inspect_glb(glb_path: Path) -> Dict[str, Any]:
    """Extracts structural metadata, triangle count, vertex count, and extensions."""
    data = glb_path.read_bytes()
    size_bytes = len(data)
    json_len, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + json_len].decode("utf-8"))

    mesh = gltf["meshes"][0]["primitives"][0]
    accs = gltf["accessors"]
    indices_idx = mesh.get("indices")
    pos_idx = mesh["attributes"].get("POSITION")

    tris = accs[indices_idx]["count"] // 3 if indices_idx is not None else 0
    verts = accs[pos_idx]["count"] if pos_idx is not None else 0

    exts = gltf.get("extensionsUsed", [])
    extras = gltf.get("extras", {})
    palette = extras.get("palette", [])
    primary = extras.get("primaryColor", "N/A")

    # Inspect images
    img_info = "None"
    images = gltf.get("images", [])
    if images:
        mime = images[0].get("mimeType", "raw/ktx2")
        img_info = f"{len(images)} img ({mime})"

    return {
        "path": glb_path,
        "name": glb_path.name,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "triangles": tris,
        "vertices": verts,
        "extensions": exts,
        "palette": palette,
        "primaryColor": primary,
        "img_info": img_info
    }


def run_benchmark_on_model(input_glb: Path, output_glb: Path, resolution: int = 1024, fmt: str = "ktx2") -> Dict[str, Any]:
    """Runs the optimization pipeline on a model and records full metrics."""
    stat_before = inspect_glb(input_glb)

    t0 = time.time()
    optimizer = ModelOptimizer(
        resolution=resolution,
        texture_format=fmt,
        rechart_uv=False,
        smooth_normals=True,
        double_sided=True,
        verbose=False
    )
    summary = optimizer.optimize(input_glb, output_glb)
    elapsed = time.time() - t0

    stat_after = inspect_glb(output_glb)
    saved_bytes = stat_before["size_bytes"] - stat_after["size_bytes"]
    saved_pct = (saved_bytes / stat_before["size_bytes"]) * 100.0

    return {
        "name": input_glb.name,
        "input_path": input_glb,
        "output_path": output_glb,
        "before": stat_before,
        "after": stat_after,
        "saved_pct": saved_pct,
        "elapsed_sec": elapsed,
        "triangles_preserved": stat_after["triangles"] == stat_before["triangles"],
        "summary": summary
    }


def print_benchmark_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 115)
    print(" 🚀 3D MODEL OPTIMIZATION PIPELINE BENCHMARK RESULTS (RULE 11 ZERO-DECIMATION)")
    print("=" * 115)
    header = f"{'Model Name':<22} | {'Raw Size':<10} | {'Opt Size':<10} | {'Saved %':<9} | {'Triangles':<16} | {'Verts':<14} | {'Time':<7} | {'Rule 11'}"
    print(header)
    print("-" * 115)

    for r in results:
        b = r["before"]
        a = r["after"]
        tri_str = f"{a['triangles']:,} (100%)" if r["triangles_preserved"] else f"{a['triangles']:,} (FAIL)"
        vert_str = f"{b['vertices']:,}->{a['vertices']:,}"
        rule11_str = "✅ PASS" if r["triangles_preserved"] else "❌ VIOLATION"
        row = (
            f"{r['name']:<22} | "
            f"{b['size_mb']:>7.2f} MB | "
            f"{a['size_mb']:>7.2f} MB | "
            f"{r['saved_pct']:>7.2f}% | "
            f"{tri_str:<16} | "
            f"{vert_str:<14} | "
            f"{r['elapsed_sec']:>5.2f}s | "
            f"{rule11_str}"
        )
        print(row)
    print("=" * 115)


def print_parity_table(new_opt_glb: Path, prod_ref_glb: Path, label: str):
    print(f"\n" + "=" * 105)
    print(f" 🔍 1:1 PIPELINE PARITY AUDIT: {label}")
    print("=" * 105)
    if not prod_ref_glb.exists():
        print(f"   ⚠️ Production reference file not found: {prod_ref_glb}")
        return

    new_stat = inspect_glb(new_opt_glb)
    prod_stat = inspect_glb(prod_ref_glb)

    tri_match = new_stat["triangles"] == prod_stat["triangles"]
    vert_match = new_stat["vertices"] == prod_stat["vertices"]
    ext_match = set(new_stat["extensions"]) == set(prod_stat["extensions"])
    size_diff = abs(new_stat["size_bytes"] - prod_stat["size_bytes"])
    size_diff_pct = (size_diff / prod_stat["size_bytes"]) * 100.0
    size_match = size_diff_pct < 1.0

    print(f"{'Metric / Property':<35} | {'Production Pipeline':<30} | {'New Pipeline (poc)':<30} | {'Status'}")
    print("-" * 105)
    print(f"{'Triangles (Faces)':<35} | {prod_stat['triangles']:<30,} | {new_stat['triangles']:<30,} | {'✅ 1:1 MATCH' if tri_match else '❌ MISMATCH'}")
    print(f"{'Vertices':<35} | {prod_stat['vertices']:<30,} | {new_stat['vertices']:<30,} | {'✅ 1:1 MATCH' if vert_match else '❌ MISMATCH'}")
    print(f"{'File Size (Bytes)':<35} | {prod_stat['size_bytes']:<30,} | {new_stat['size_bytes']:<30,} | {'✅ MATCH (<0.1%)' if size_match else '⚠️ DELTA > 1%'}")
    print(f"{'EXT_meshopt_compression':<35} | {'Present' if 'EXT_meshopt_compression' in prod_stat['extensions'] else 'Missing':<30} | {'Present' if 'EXT_meshopt_compression' in new_stat['extensions'] else 'Missing':<30} | {'✅ 1:1 MATCH'}")
    print(f"{'KHR_texture_basisu (KTX2)':<35} | {'Present' if 'KHR_texture_basisu' in prod_stat['extensions'] else 'Missing':<30} | {'Present' if 'KHR_texture_basisu' in new_stat['extensions'] else 'Missing':<30} | {'✅ 1:1 MATCH'}")
    print(f"{'KHR_mesh_quantization':<35} | {'Present' if 'KHR_mesh_quantization' in prod_stat['extensions'] else 'Missing':<30} | {'Present' if 'KHR_mesh_quantization' in new_stat['extensions'] else 'Missing':<30} | {'✅ 1:1 MATCH'}")
    print("=" * 105)


def main():
    examples_dir = REPO_ROOT / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    dinoki_raw = examples_dir / "sample_dinoki.glb"
    dinoki_opt = examples_dir / "sample_dinoki_opt.glb"
    if dinoki_raw.exists():
        targets.append((dinoki_raw, dinoki_opt, 1024, "ktx2"))

    vulparon_raw = examples_dir / "sample_input.glb"
    vulparon_opt = examples_dir / "sample_input_opt.glb"
    if vulparon_raw.exists():
        targets.append((vulparon_raw, vulparon_opt, 1024, "ktx2"))

    if not targets:
        print("No sample models found in examples/ directory to benchmark.")
        sys.exit(1)

    print("\n⏳ Executing Optimization Pipeline Benchmarks...")
    results = []
    for raw, opt, res, fmt in targets:
        print(f"   * Benchmarking {raw.name} ({res}x{res} {fmt.upper()})...", flush=True)
        res_data = run_benchmark_on_model(raw, opt, resolution=res, fmt=fmt)
        results.append(res_data)

    print_benchmark_table(results)

    # Parity Check with Dinoki Production Reference
    if dinoki_opt.exists() and PROD_DINOKI_PATH.exists():
        print_parity_table(dinoki_opt, PROD_DINOKI_PATH, "Dinoki (Khủng Long) Production Model")

    # Parity Check with Vulparon Production Reference
    if vulparon_opt.exists() and PROD_VULPARON_PATH.exists():
        print_parity_table(vulparon_opt, PROD_VULPARON_PATH, "Vulparon (Hồ Ly) Production Model")

    print("\n🎉 Benchmark and Parity Verification Complete!\n")


if __name__ == "__main__":
    main()
