#!/usr/bin/env python3
"""
scripts/benchmark_runner.py

High-Precision 3D Model Optimization Pipeline Profiler & Benchmark Runner.
Measures performance, microsecond step durations via time.perf_counter(),
intermediate stage file sizes, Rule 11 Zero-Decimation geometric integrity,
texture resolutions, and GPU VRAM footprint before and after optimization.

Supports:
- Model 1: koidrax (examples/models/koidrax_raw.glb)
- Model 2: dinoki (examples/sample_dinoki.glb)
- Model 3: vulparon (examples/models/vulparon_raw.glb)
- Benchmark modes:
  * --mode optimized: Runs with texture preservation fix (preserve_mesh_textures)
  * --mode baseline: Runs baseline without texture preservation (trimesh PNG export)
  * --mode both: Runs both and produces comparative speedup & size reduction analytics

Usage:
  # Quick test on fast dinoki model (both baseline & optimized comparison)
  .venv/bin/python scripts/benchmark_runner.py --model dinoki --mode both

  # Benchmark single model in optimized mode
  .venv/bin/python scripts/benchmark_runner.py --model dinoki -o dinoki_benchmark.json

  # Benchmark all 3 baseline models
  .venv/bin/python scripts/benchmark_runner.py --all --mode both -o all_models_benchmark.json
"""

import os
import sys
import time
import json
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer.step_pipeline import StepPipeline, inspect_glb_metrics

# Standard target model catalog
BASELINE_MODELS: Dict[str, Dict[str, Any]] = {
    "dinoki": {
        "key": "dinoki",
        "name": "Dinoki (Khủng Long)",
        "path": REPO_ROOT / "examples" / "sample_dinoki.glb",
        "description": "Fast baseline model (~45k triangles, 1.93 MB, 1536x1536 JPEG)"
    },
    "koidrax": {
        "key": "koidrax",
        "name": "Koidrax (Dragon)",
        "path": REPO_ROOT / "examples" / "models" / "koidrax_raw.glb",
        "description": "High-poly baseline model (~289k triangles, 11.75 MB, 4096x4096 JPEG)"
    },
    "vulparon": {
        "key": "vulparon",
        "name": "Vulparon (Fox)",
        "path": REPO_ROOT / "examples" / "models" / "vulparon_raw.glb",
        "description": "High-poly baseline model (~282k triangles, 10.33 MB, 4096x4096 JPEG)"
    }
}


def format_bytes(num_bytes: int, decimals: int = 2) -> str:
    """Formats raw byte count into human-readable string (B, KB, MB, GB)."""
    if not num_bytes or num_bytes <= 0:
        return "0 B"
    k = 1024.0
    sizes = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(num_bytes) / math.log(k)))
    i = max(0, min(i, len(sizes) - 1))
    val = num_bytes / (k ** i)
    return f"{val:.{decimals}f} {sizes[i]}"


def calculate_geometry_vram(faces: int, vertices: int) -> int:
    """
    Estimates geometry GPU VRAM in bytes:
    - Index buffer: faces * 3 * (4 if vertices > 65535 else 2) bytes
    - Vertex buffer: vertices * 32 bytes (Position: 12B, Normal: 12B, UV0: 8B)
    """
    index_elem_size = 4 if vertices > 65535 else 2
    index_bytes = faces * 3 * index_elem_size
    vertex_bytes = vertices * 32
    return index_bytes + vertex_bytes


def calculate_texture_vram(width: int, height: int, is_ktx2: bool) -> int:
    """
    Estimates texture GPU VRAM with standard 1.333x mipmap overhead:
    - Uncompressed RGBA8888: W * H * 4 * (4/3) bytes
    - KTX2 (UASTC / BC7 / ASTC): W * H * 1 * (4/3) bytes
    """
    bytes_per_pixel = 1.0 if is_ktx2 else 4.0
    return int(round(width * height * bytes_per_pixel * (4.0 / 3.0)))


def run_model_benchmark(
    model_key: str,
    model_path: Path,
    resolution: int = 1024,
    texture_format: str = "ktx2",
    preserve_textures: bool = True,
    workdir: Optional[Path] = None,
    clean_workdir: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes full 7-step pipeline on target model with high-precision time.perf_counter().
    Captures step durations, file sizes, triangle integrity (Rule 11), and VRAM metrics.
    """
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mode_label = "optimized" if preserve_textures else "baseline"

    temp_dir_ctx = None
    if workdir is None:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix=f"benchmark_{model_key}_{mode_label}_")
        effective_workdir = Path(temp_dir_ctx.name)
    else:
        effective_workdir = Path(workdir).resolve()
        effective_workdir.mkdir(parents=True, exist_ok=True)

    try:
        t_pipeline_start = time.perf_counter()

        pipeline = StepPipeline(
            resolution=resolution,
            texture_format=texture_format,
            smooth_normals=True,
            double_sided=False,
            preserve_textures=preserve_textures,
            verbose=verbose,
            stream_events=False
        )

        pipeline_result = pipeline.run(model_path, effective_workdir)
        t_pipeline_end = time.perf_counter()
        total_duration_sec = t_pipeline_end - t_pipeline_start

        steps_record = pipeline_result.get("steps", [])
        steps_map = {step["step"]: step for step in steps_record}

        # Step 0 (Raw baseline)
        step0 = steps_map.get(0, {})
        step0_metrics = step0.get("metrics", {})
        step0_duration_sec = step0.get("durationSeconds", 0.0)
        raw_bytes = step0_metrics.get("fileSizeBytes", model_path.stat().st_size)
        raw_faces = step0_metrics.get("faces", 0)
        raw_verts = step0_metrics.get("vertices", 0)

        # Step 1 (Cleaned & Grounded)
        step1 = steps_map.get(1, {})
        step1_metrics = step1.get("metrics", {})
        step1_duration_sec = step1.get("durationSeconds", 0.0)
        step1_file = effective_workdir / "step_01_cleaned_grounded.glb"
        step1_bytes = step1_file.stat().st_size if step1_file.exists() else step1_metrics.get("fileSizeBytes", 0)
        step1_faces = step1_metrics.get("faces", 0)
        step1_verts = step1_metrics.get("vertices", 0)

        # Step 2 (Visibility Orient)
        step2 = steps_map.get(2, {})
        step2_metrics = step2.get("metrics", {})
        step2_duration_sec = step2.get("durationSeconds", 0.0)
        step2_file = effective_workdir / "step_02_oriented.glb"
        step2_bytes = step2_file.stat().st_size if step2_file.exists() else step2_metrics.get("fileSizeBytes", 0)
        step2_faces = step2_metrics.get("faces", 0)
        step2_verts = step2_metrics.get("vertices", 0)

        # Step 3 (Texture Baked)
        step3 = steps_map.get(3, {})
        step3_duration_sec = step3.get("durationSeconds", 0.0)

        # Step 4 (Palette Tagged)
        step4 = steps_map.get(4, {})
        step4_duration_sec = step4.get("durationSeconds", 0.0)

        # Step 5 (Meshopt Geometry)
        step5 = steps_map.get(5, {})
        step5_duration_sec = step5.get("durationSeconds", 0.0)

        # Step 6 (Final Output)
        step6 = steps_map.get(6, {})
        step6_metrics = step6.get("metrics", {})
        step6_duration_sec = step6.get("durationSeconds", 0.0)
        step6_file = effective_workdir / "step_06_final.glb"
        final_bytes = step6_file.stat().st_size if step6_file.exists() else step6_metrics.get("fileSizeBytes", 0)
        final_faces = step6_metrics.get("faces", 0)
        final_verts = step6_metrics.get("vertices", 0)

        # Compression calculations
        saved_bytes = raw_bytes - final_bytes
        saved_pct = round((saved_bytes / raw_bytes) * 100.0, 2) if raw_bytes > 0 else 0.0

        # Rule 11 Zero-Decimation verification
        rule11_verified = (raw_faces > 0 and
                           step1_faces == raw_faces and
                           step2_faces == raw_faces and
                           final_faces == raw_faces)
        tri_preserved_pct = round((final_faces / raw_faces) * 100.0, 2) if raw_faces > 0 else 0.0

        # Textures & VRAM calculation
        raw_textures = step0_metrics.get("textures", [])
        final_textures = step6_metrics.get("textures", [])

        raw_tex_res = raw_textures[0].get("resolutionFormatted", "N/A") if raw_textures else "N/A"
        final_tex_res = final_textures[0].get("resolutionFormatted", f"{resolution}x{resolution}") if final_textures else f"{resolution}x{resolution}"

        raw_tex_format = raw_textures[0].get("format", "JPEG") if raw_textures else "None"
        final_tex_format = final_textures[0].get("format", texture_format.upper()) if final_textures else texture_format.upper()

        # GPU VRAM before (Raw)
        raw_tex_vram = step0_metrics.get("totalGpuVramBytes", 0)
        if raw_tex_vram == 0 and raw_textures:
            w, h = raw_textures[0].get("resolution", [0, 0])
            raw_tex_vram = calculate_texture_vram(w, h, is_ktx2=False)
        raw_geo_vram = calculate_geometry_vram(raw_faces, raw_verts)
        raw_total_vram = raw_tex_vram + raw_geo_vram

        # GPU VRAM after (Final)
        final_tex_vram = step6_metrics.get("totalGpuVramBytes", 0)
        if final_tex_vram == 0 and final_textures:
            w, h = final_textures[0].get("resolution", [resolution, resolution])
            final_tex_vram = calculate_texture_vram(w, h, is_ktx2=(texture_format.lower() == "ktx2"))
        final_geo_vram = calculate_geometry_vram(final_faces, final_verts)
        final_total_vram = final_tex_vram + final_geo_vram

        vram_saved_bytes = raw_total_vram - final_total_vram
        vram_saved_pct = round((vram_saved_bytes / raw_total_vram) * 100.0, 2) if raw_total_vram > 0 else 0.0
        tex_vram_saved_bytes = raw_tex_vram - final_tex_vram
        tex_vram_saved_pct = round((tex_vram_saved_bytes / raw_tex_vram) * 100.0, 2) if raw_tex_vram > 0 else 0.0

        # Detailed step list
        step_items = []
        for s in steps_record:
            step_file_path = effective_workdir / s["file"]
            size_on_disk = step_file_path.stat().st_size if step_file_path.exists() else s["metrics"].get("fileSizeBytes", 0)
            step_items.append({
                "step": s["step"],
                "name": s["stepName"],
                "file": s["file"],
                "description": s["description"],
                "duration_seconds": round(s.get("durationSeconds", 0.0), 4),
                "duration_ms": round(s.get("durationMs", s.get("durationSeconds", 0.0) * 1000.0), 2),
                "file_size_bytes": size_on_disk,
                "file_size_formatted": format_bytes(size_on_disk),
                "triangles": s["metrics"].get("faces", 0),
                "vertices": s["metrics"].get("vertices", 0)
            })

        result = {
            "model_key": model_key,
            "model_name": model_path.name,
            "mode": mode_label,
            "preserve_textures": preserve_textures,
            "input_path": str(model_path),
            "output_dir": str(effective_workdir),
            "pipeline_config": {
                "target_resolution": f"{resolution}x{resolution}",
                "texture_format": texture_format.upper(),
                "smooth_normals": True,
                "preserve_textures": preserve_textures,
                "rule_11_zero_decimation": True
            },
            "timing": {
                "total_pipeline_execution_seconds": round(total_duration_sec, 4),
                "total_pipeline_execution_ms": round(total_duration_sec * 1000.0, 2),
                "step_01_elapsed_seconds": round(step1_duration_sec, 4),
                "step_01_elapsed_ms": round(step1_duration_sec * 1000.0, 2),
                "step_02_elapsed_seconds": round(step2_duration_sec, 4),
                "step_02_elapsed_ms": round(step2_duration_sec * 1000.0, 2),
                "remaining_step_durations": {
                    "step_00_raw": {
                        "seconds": round(step0_duration_sec, 4),
                        "ms": round(step0_duration_sec * 1000.0, 2)
                    },
                    "step_03_texture_baked": {
                        "seconds": round(step3_duration_sec, 4),
                        "ms": round(step3_duration_sec * 1000.0, 2)
                    },
                    "step_04_palette_tagged": {
                        "seconds": round(step4_duration_sec, 4),
                        "ms": round(step4_duration_sec * 1000.0, 2)
                    },
                    "step_05_meshopt": {
                        "seconds": round(step5_duration_sec, 4),
                        "ms": round(step5_duration_sec * 1000.0, 2)
                    },
                    "step_06_final": {
                        "seconds": round(step6_duration_sec, 4),
                        "ms": round(step6_duration_sec * 1000.0, 2)
                    }
                },
                "all_steps": step_items
            },
            "file_size": {
                "raw_input_bytes": raw_bytes,
                "raw_input_mb": round(raw_bytes / (1024.0 * 1024.0), 3),
                "raw_input_formatted": format_bytes(raw_bytes),
                "step_01_intermediate_bytes": step1_bytes,
                "step_01_intermediate_mb": round(step1_bytes / (1024.0 * 1024.0), 3),
                "step_01_intermediate_formatted": format_bytes(step1_bytes),
                "step_01_intermediate_file": "step_01_cleaned_grounded.glb",
                "step_02_intermediate_bytes": step2_bytes,
                "step_02_intermediate_mb": round(step2_bytes / (1024.0 * 1024.0), 3),
                "step_02_intermediate_formatted": format_bytes(step2_bytes),
                "step_02_intermediate_file": "step_02_oriented.glb",
                "final_output_bytes": final_bytes,
                "final_output_mb": round(final_bytes / (1024.0 * 1024.0), 3),
                "final_output_formatted": format_bytes(final_bytes),
                "saved_bytes": saved_bytes,
                "saved_mb": round(saved_bytes / (1024.0 * 1024.0), 3),
                "saved_percent": saved_pct
            },
            "geometry_rule11": {
                "raw_triangles": raw_faces,
                "step_01_triangles": step1_faces,
                "step_02_triangles": step2_faces,
                "final_triangles": final_faces,
                "zero_decimation_verified": rule11_verified,
                "triangles_preserved_percent": tri_preserved_pct,
                "raw_vertices": raw_verts,
                "step_01_vertices": step1_verts,
                "step_02_vertices": step2_verts,
                "final_vertices": final_verts
            },
            "textures_and_vram": {
                "texture_resolution_before": raw_tex_res,
                "texture_resolution_after": final_tex_res,
                "texture_format_before": raw_tex_format,
                "texture_format_after": final_tex_format,
                "texture_vram_before_bytes": raw_tex_vram,
                "texture_vram_before_mb": round(raw_tex_vram / (1024.0 * 1024.0), 2),
                "texture_vram_before_formatted": format_bytes(raw_tex_vram),
                "texture_vram_after_bytes": final_tex_vram,
                "texture_vram_after_mb": round(final_tex_vram / (1024.0 * 1024.0), 2),
                "texture_vram_after_formatted": format_bytes(final_tex_vram),
                "texture_vram_saved_bytes": tex_vram_saved_bytes,
                "texture_vram_saved_percent": tex_vram_saved_pct,
                "geometry_vram_before_bytes": raw_geo_vram,
                "geometry_vram_after_bytes": final_geo_vram,
                "total_gpu_vram_before_bytes": raw_total_vram,
                "total_gpu_vram_before_mb": round(raw_total_vram / (1024.0 * 1024.0), 2),
                "total_gpu_vram_before_formatted": format_bytes(raw_total_vram),
                "total_gpu_vram_after_bytes": final_total_vram,
                "total_gpu_vram_after_mb": round(final_total_vram / (1024.0 * 1024.0), 2),
                "total_gpu_vram_after_formatted": format_bytes(final_total_vram),
                "total_gpu_vram_saved_bytes": vram_saved_bytes,
                "total_gpu_vram_saved_percent": vram_saved_pct
            },
            "intermediate_files": {
                s["file"]: {
                    "step": s["step"],
                    "path": str(effective_workdir / s["file"]),
                    "file_size_bytes": s["file_size_bytes"],
                    "file_size_formatted": s["file_size_formatted"],
                    "triangles": s["triangles"],
                    "vertices": s["vertices"]
                }
                for s in step_items
            }
        }

        return result

    finally:
        if clean_workdir and temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


def run_model_comparison(
    model_key: str,
    model_path: Path,
    resolution: int = 1024,
    texture_format: str = "ktx2",
    workdir: Optional[Path] = None,
    clean_workdir: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs both Baseline (without preserve_mesh_textures) and Optimized (with preserve_mesh_textures)
    and computes speedup, intermediate size reduction, and parity checks.
    """
    base_workdir = (Path(workdir) / f"{model_key}_baseline") if workdir else None
    opt_workdir = (Path(workdir) / f"{model_key}_optimized") if workdir else None

    if verbose:
        print(f"\n   [1/2] Running BASELINE (without preserve_mesh_textures)...", flush=True)
    baseline_res = run_model_benchmark(
        model_key=model_key,
        model_path=model_path,
        resolution=resolution,
        texture_format=texture_format,
        preserve_textures=False,
        workdir=base_workdir,
        clean_workdir=clean_workdir,
        verbose=verbose
    )

    if verbose:
        print(f"   [2/2] Running OPTIMIZED (with preserve_mesh_textures)...", flush=True)
    optimized_res = run_model_benchmark(
        model_key=model_key,
        model_path=model_path,
        resolution=resolution,
        texture_format=texture_format,
        preserve_textures=True,
        workdir=opt_workdir,
        clean_workdir=clean_workdir,
        verbose=verbose
    )

    # Calculate comparative gains
    b_t = baseline_res["timing"]
    o_t = optimized_res["timing"]
    b_fs = baseline_res["file_size"]
    o_fs = optimized_res["file_size"]

    # Step 1 time saving
    s1_base_t = b_t["step_01_elapsed_seconds"]
    s1_opt_t = o_t["step_01_elapsed_seconds"]
    s1_time_saved_pct = round((1.0 - (s1_opt_t / s1_base_t)) * 100.0, 2) if s1_base_t > 0 else 0.0

    # Step 2 time saving
    s2_base_t = b_t["step_02_elapsed_seconds"]
    s2_opt_t = o_t["step_02_elapsed_seconds"]
    s2_time_saved_pct = round((1.0 - (s2_opt_t / s2_base_t)) * 100.0, 2) if s2_base_t > 0 else 0.0

    # Total time saving
    total_base_t = b_t["total_pipeline_execution_seconds"]
    total_opt_t = o_t["total_pipeline_execution_seconds"]
    total_time_saved_pct = round((1.0 - (total_opt_t / total_base_t)) * 100.0, 2) if total_base_t > 0 else 0.0

    # Step 1 intermediate size reduction (bloat eliminated)
    s1_base_b = b_fs["step_01_intermediate_bytes"]
    s1_opt_b = o_fs["step_01_intermediate_bytes"]
    s1_bytes_saved = s1_base_b - s1_opt_b
    s1_size_saved_pct = round((s1_bytes_saved / s1_base_b) * 100.0, 2) if s1_base_b > 0 else 0.0

    comparison = {
        "model_key": model_key,
        "model_name": model_path.name,
        "step_01_time_baseline_seconds": s1_base_t,
        "step_01_time_optimized_seconds": s1_opt_t,
        "step_01_time_saved_percent": s1_time_saved_pct,
        "step_02_time_baseline_seconds": s2_base_t,
        "step_02_time_optimized_seconds": s2_opt_t,
        "step_02_time_saved_percent": s2_time_saved_pct,
        "total_time_baseline_seconds": total_base_t,
        "total_time_optimized_seconds": total_opt_t,
        "total_time_saved_percent": total_time_saved_pct,
        "step_01_size_baseline_bytes": s1_base_b,
        "step_01_size_baseline_formatted": b_fs["step_01_intermediate_formatted"],
        "step_01_size_optimized_bytes": s1_opt_b,
        "step_01_size_optimized_formatted": o_fs["step_01_intermediate_formatted"],
        "step_01_bloat_eliminated_bytes": s1_bytes_saved,
        "step_01_bloat_eliminated_formatted": format_bytes(s1_bytes_saved),
        "step_01_size_saved_percent": s1_size_saved_pct,
        "rule11_verified": optimized_res["geometry_rule11"]["zero_decimation_verified"] and baseline_res["geometry_rule11"]["zero_decimation_verified"]
    }

    return {
        "model_key": model_key,
        "model_name": model_path.name,
        "comparison": comparison,
        "baseline": baseline_res,
        "optimized": optimized_res
    }


def print_comparison_table(comparisons: List[Dict[str, Any]]) -> None:
    """Prints a comparative table showing speedup and intermediate size reduction."""
    print("\n" + "=" * 130)
    print(" 🚀 TEXTURE PRESERVATION OPTIMIZATION BENCHMARK: BASELINE vs OPTIMIZED (Steps 1 & 2)")
    print("=" * 130)
    header = (
        f"{'Model':<10} | "
        f"{'Step 1 Size (Base -> Opt)':<27} | "
        f"{'Bloat Saved':<12} | "
        f"{'Step 1 Time (s)':<18} | "
        f"{'Step 1 %':<9} | "
        f"{'Total Time (s)':<18} | "
        f"{'Total %':<8} | "
        f"{'Rule 11'}"
    )
    print(header)
    print("-" * 130)

    for c in comparisons:
        comp = c["comparison"]
        size_str = f"{comp['step_01_size_baseline_formatted']} -> {comp['step_01_size_optimized_formatted']}"
        s1_time_str = f"{comp['step_01_time_baseline_seconds']:>6.3f}s -> {comp['step_01_time_optimized_seconds']:>6.3f}s"
        total_time_str = f"{comp['total_time_baseline_seconds']:>6.2f}s -> {comp['total_time_optimized_seconds']:>6.2f}s"
        rule11_str = "✅ PASS" if comp["rule11_verified"] else "❌ FAIL"

        row = (
            f"{comp['model_key']:<10} | "
            f"{size_str:<27} | "
            f"{comp['step_01_bloat_eliminated_formatted']:>12} | "
            f"{s1_time_str:<18} | "
            f"{comp['step_01_time_saved_percent']:>7.1f}% | "
            f"{total_time_str:<18} | "
            f"{comp['total_time_saved_percent']:>6.1f}% | "
            f"{rule11_str}"
        )
        print(row)
    print("=" * 130)


def print_single_mode_summary(results: List[Dict[str, Any]]) -> None:
    """Prints a formatted summary table for a single mode run."""
    print("\n" + "=" * 135)
    print(" 🚀 3D MODEL OPTIMIZATION PIPELINE PROFILER SUMMARY (RULE 11 ZERO-DECIMATION)")
    print("=" * 135)
    header = (
        f"{'Model':<10} | "
        f"{'Raw Size':<9} | "
        f"{'Opt Size':<9} | "
        f"{'Saved %':<8} | "
        f"{'Triangles (Raw->Opt)':<22} | "
        f"{'Rule 11':<10} | "
        f"{'GPU VRAM':<18} | "
        f"{'VRAM Saved':<10} | "
        f"{'Step 1 (s)':<10} | "
        f"{'Step 2 (s)':<10} | "
        f"{'Total (s)':<9}"
    )
    print(header)
    print("-" * 135)

    for r in results:
        fs = r["file_size"]
        g = r["geometry_rule11"]
        tv = r["textures_and_vram"]
        tm = r["timing"]

        rule11_label = "✅ PASS" if g["zero_decimation_verified"] else "❌ FAIL"
        tri_str = f"{g['raw_triangles']:,} -> {g['final_triangles']:,}"
        vram_str = f"{tv['total_gpu_vram_before_formatted']} -> {tv['total_gpu_vram_after_formatted']}"

        row = (
            f"{r['model_key']:<10} | "
            f"{fs['raw_input_formatted']:>9} | "
            f"{fs['final_output_formatted']:>9} | "
            f"{fs['saved_percent']:>7.2f}% | "
            f"{tri_str:<22} | "
            f"{rule11_label:<10} | "
            f"{vram_str:<18} | "
            f"{tv['total_gpu_vram_saved_percent']:>8.2f}% | "
            f"{tm['step_01_elapsed_seconds']:>9.3f}s | "
            f"{tm['step_02_elapsed_seconds']:>9.3f}s | "
            f"{tm['total_pipeline_execution_seconds']:>8.3f}s"
        )
        print(row)
    print("=" * 135)


def print_step_breakdown(result: Dict[str, Any]) -> None:
    """Prints a granular breakdown table of all 7 pipeline steps for a single model."""
    print(f"\n📊 Step-by-Step Breakdown: {result['model_key'].upper()} [{result.get('mode', 'optimized').upper()}] ({result['model_name']})")
    print("-" * 105)
    header = f"{'Step':<6} | {'Step Name':<20} | {'Intermediate File':<28} | {'Duration (s)':<12} | {'Duration (ms)':<14} | {'Size':<10} | {'Triangles'}"
    print(header)
    print("-" * 105)
    for s in result["timing"]["all_steps"]:
        row = (
            f"Step {s['step']:<1} | "
            f"{s['name']:<20} | "
            f"{s['file']:<28} | "
            f"{s['duration_seconds']:>11.4f}s | "
            f"{s['duration_ms']:>12.2f}ms | "
            f"{s['file_size_formatted']:>10} | "
            f"{s['triangles']:>9,}"
        )
        print(row)
    print("-" * 105)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-Precision 3D Model Optimization Profiler & Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Target model: 'dinoki', 'koidrax', 'vulparon', or path to .glb file."
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Profile all 3 baseline models (koidrax, dinoki, vulparon)."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "optimized", "baseline"],
        default="both",
        help="Benchmark mode: 'both' (compare baseline vs optimized), 'optimized', or 'baseline'."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save structured benchmark JSON results."
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=1024,
        help="Target texture resolution (e.g. 512, 1024, 2048)."
    )
    parser.add_argument(
        "--format", "-f",
        choices=["ktx2", "webp"],
        default="ktx2",
        help="GPU texture compression format."
    )
    parser.add_argument(
        "--workdir", "-w",
        default=None,
        help="Custom output workspace directory to preserve intermediate step GLBs."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress intermediate pipeline step logs during execution."
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON to stdout without human-readable tables."
    )

    args = parser.parse_args()

    # Determine target models
    targets: List[Dict[str, Any]] = []

    if args.all:
        targets = [
            BASELINE_MODELS["dinoki"],
            BASELINE_MODELS["koidrax"],
            BASELINE_MODELS["vulparon"]
        ]
    elif args.model:
        model_input = args.model.strip().lower()
        if model_input in BASELINE_MODELS:
            targets = [BASELINE_MODELS[model_input]]
        else:
            custom_path = Path(args.model).resolve()
            if custom_path.exists():
                targets = [{
                    "key": custom_path.stem.lower(),
                    "name": custom_path.stem,
                    "path": custom_path,
                    "description": f"Custom user model ({custom_path.name})"
                }]
            else:
                available = ", ".join(BASELINE_MODELS.keys())
                print(f"Error: Unknown model '{args.model}'. Available preset models: {available}", file=sys.stderr)
                sys.exit(1)
    else:
        # Default to fast baseline model (dinoki)
        targets = [BASELINE_MODELS["dinoki"]]

    if not args.json_only:
        print("\n" + "=" * 80)
        print(" 🎯 3D MODEL OPTIMIZATION PIPELINE PROFILER INITIALIZED")
        print("=" * 80)
        print(f" Targets: {', '.join(t['key'] for t in targets)}")
        print(f" Mode: {args.mode.upper()}")
        print(f" Resolution: {args.resolution}x{args.resolution} | Format: {args.format.upper()}")
        print(f" Rule 11 Zero-Decimation Policy: ENFORCED (100% face count preservation)")
        print("=" * 80)

    comparison_results: List[Dict[str, Any]] = []
    single_results: List[Dict[str, Any]] = []

    for target in targets:
        model_key = target["key"]
        model_path = target["path"]

        if not model_path.exists():
            print(f"⚠️ Model file not found on disk: {model_path}", file=sys.stderr)
            continue

        if not args.json_only:
            print(f"\n▶️ Processing model '{model_key}' ({model_path.name})...", flush=True)

        if args.mode == "both":
            comp_data = run_model_comparison(
                model_key=model_key,
                model_path=model_path,
                resolution=args.resolution,
                texture_format=args.format,
                workdir=args.workdir,
                clean_workdir=(args.workdir is None),
                verbose=not args.quiet and not args.json_only
            )
            comparison_results.append(comp_data)
        else:
            preserve = (args.mode == "optimized")
            res_data = run_model_benchmark(
                model_key=model_key,
                model_path=model_path,
                resolution=args.resolution,
                texture_format=args.format,
                preserve_textures=preserve,
                workdir=args.workdir,
                clean_workdir=(args.workdir is None),
                verbose=not args.quiet and not args.json_only
            )
            single_results.append(res_data)

    if not comparison_results and not single_results:
        print("Error: No models were successfully benchmarked.", file=sys.stderr)
        sys.exit(1)

    # Prepare structured JSON payload
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "poc-optimize-3d-model benchmark_runner",
        "benchmark_mode": args.mode,
        "models_profiled_count": len(comparison_results) if args.mode == "both" else len(single_results)
    }

    if args.mode == "both":
        payload["results"] = [c["optimized"] for c in comparison_results] if len(comparison_results) > 1 else comparison_results[0]["optimized"]
        payload["comparisons"] = comparison_results if len(comparison_results) > 1 else comparison_results[0]
    else:
        payload["results"] = single_results if len(single_results) > 1 else single_results[0]

    # Save to disk if requested
    output_path = Path(args.output).resolve() if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        if not args.json_only:
            print(f"\n💾 Benchmark metrics JSON successfully saved to: {output_path}")

    # Output tables and JSON
    if not args.json_only:
        if args.mode == "both":
            print_comparison_table(comparison_results)
            for c in comparison_results:
                print_step_breakdown(c["optimized"])
        else:
            print_single_mode_summary(single_results)
            for r in single_results:
                print_step_breakdown(r)

        print("\n📋 Structured JSON Results:")
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
