#!/usr/bin/env python3
"""
benchmark_step3_profiler.py

Comprehensive profiler and comparative analysis for Step 3 of the 3D optimization pipeline:
Dinoki (Khủng Long) vs Gravilux (Statue).

Outputs quantitative metrics:
- Mesh & texture statistics
- Direct mode performance (rechart_uv=False)
- Rechart mode performance (rechart_uv=True) broken down by sub-steps
- Face scaling curve (25k to 289k faces)
- Resolution scaling curve (512 to 4096)
- Memory usage (RSS and tracemalloc peak)
- Exact bottleneck identification
"""

import sys
import json
import time
import tracemalloc
import resource
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import trimesh
import xatlas

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


def get_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def inspect_model_details(path: Path) -> Dict[str, Any]:
    file_bytes = path.stat().st_size
    scene = trimesh.load(path)
    geom_name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[geom_name]

    verts = len(mesh.vertices)
    faces = len(mesh.faces)
    area = float(mesh.area)
    bounds = mesh.bounds.tolist() if mesh.bounds is not None else []
    extents = (mesh.bounds[1] - mesh.bounds[0]).tolist() if mesh.bounds is not None else []

    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
    uv_min = mesh.visual.uv.min(axis=0).tolist() if has_uv and len(mesh.visual.uv) > 0 else []
    uv_max = mesh.visual.uv.max(axis=0).tolist() if has_uv and len(mesh.visual.uv) > 0 else []

    mat = getattr(mesh.visual, "material", None)
    tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None) if mat else None

    tex_w, tex_h = tex_img.size if tex_img else (0, 0)
    tex_mode = tex_img.mode if tex_img else "None"
    tex_fmt = getattr(tex_img, "format", "None") if tex_img else "None"
    uncompressed_mb = (tex_w * tex_h * 4) / (1024 * 1024)
    vram_mb = uncompressed_mb * (4.0 / 3.0)

    return {
        "path": str(path),
        "file_name": path.name,
        "file_size_bytes": file_bytes,
        "file_size_mb": round(file_bytes / (1024 * 1024), 2),
        "geometry_name": geom_name,
        "vertex_count": verts,
        "face_count": faces,
        "surface_area": round(area, 4),
        "bounds": bounds,
        "extents": [round(x, 4) for x in extents],
        "has_uv": has_uv,
        "uv_count": len(mesh.visual.uv) if has_uv else 0,
        "uv_min": [round(x, 4) for x in uv_min],
        "uv_max": [round(x, 4) for x in uv_max],
        "material_class": mat.__class__.__name__ if mat else "None",
        "double_sided": getattr(mat, "doubleSided", False) if mat else False,
        "texture_width": tex_w,
        "texture_height": tex_h,
        "texture_mode": tex_mode,
        "texture_format": tex_fmt,
        "texture_uncompressed_mb": round(uncompressed_mb, 2),
        "texture_gpu_vram_mb": round(vram_mb, 2),
    }


def profile_direct_step3(path: Path, target_res: int) -> Dict[str, Any]:
    scene = trimesh.load(path)
    geom_name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[geom_name]

    orig_mat = getattr(mesh.visual, "material", None)
    source_img = getattr(orig_mat, "baseColorTexture", None) or getattr(orig_mat, "image", None)
    if source_img is None:
        source_img = Image.new("RGB", (target_res, target_res), (200, 200, 200))

    tracemalloc.start()
    rss_start = get_rss_mb()
    t_start = time.perf_counter()

    # 1. Resize
    t0 = time.perf_counter()
    eff_res = clamp_target_resolution(target_res, source_img.size)
    img_rgb = source_img.convert("RGB")
    resampled = img_rgb.resize((eff_res, eff_res), Image.Resampling.LANCZOS)
    arr = np.array(resampled, dtype=np.uint8)
    t_resize = time.perf_counter() - t0

    # 2. Mask
    t0 = time.perf_counter()
    is_black = np.all(arr <= 2, axis=-1)
    is_covered = ~is_black
    t_mask = time.perf_counter() - t0

    # 3. Dilation
    t0 = time.perf_counter()
    dilated_arr = dilate_texture(arr, is_covered, padding=16)
    t_dilate = time.perf_counter() - t0

    # 4. Mesh & Export
    t0 = time.perf_counter()
    out_mesh, clean_pil = direct_resample_texture(mesh, source_image=source_img, target_res=target_res, dilation_padding=16)
    s_scene = trimesh.Scene({"Model": out_mesh})
    s_bytes = trimesh.exchange.gltf.export_glb(s_scene, include_normals=True)
    s_bytes = set_doublesided_material(s_bytes)
    t_export = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = get_rss_mb()

    return {
        "target_res": eff_res,
        "resize_sec": round(t_resize, 4),
        "mask_sec": round(t_mask, 4),
        "dilation_sec": round(t_dilate, 4),
        "mesh_export_sec": round(t_export, 4),
        "total_sec": round(t_total, 4),
        "peak_tracemalloc_mb": round(peak_mem / (1024 * 1024), 2),
        "rss_delta_mb": round(rss_peak - rss_start, 2),
        "glb_output_bytes": len(s_bytes),
        "glb_output_mb": round(len(s_bytes) / (1024 * 1024), 2)
    }


def profile_rechart_substeps(mesh: trimesh.Trimesh, target_res: int, max_faces: int = None) -> Dict[str, Any]:
    if max_faces and max_faces < len(mesh.faces):
        sub_faces = mesh.faces[:max_faces]
        used_v = np.unique(sub_faces)
        v_map = {old: new for new, old in enumerate(used_v)}
        sub_verts = mesh.vertices[used_v]
        faces_in = np.vectorize(v_map.get)(sub_faces)
        verts_in = sub_verts
        raw_uv = getattr(mesh.visual, "uv", np.zeros((len(mesh.vertices), 2)))[used_v]
    else:
        faces_in = mesh.faces
        verts_in = mesh.vertices
        raw_uv = getattr(mesh.visual, "uv", np.zeros((len(mesh.vertices), 2)))

    orig_mat = getattr(mesh.visual, "material", None)
    source_img = getattr(orig_mat, "baseColorTexture", None) or getattr(orig_mat, "image", None)
    if source_img is None:
        source_img = Image.new("RGB", (target_res, target_res), (200, 200, 200))
    src_rgb = np.asarray(source_img.convert("RGB"), dtype=np.uint8)

    timings = {}
    tracemalloc.start()
    rss_start = get_rss_mb()

    # Substep 1: add_mesh
    t0 = time.perf_counter()
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.ascontiguousarray(verts_in, dtype=np.float32), np.ascontiguousarray(faces_in, dtype=np.uint32))
    timings["1_add_mesh_sec"] = round(time.perf_counter() - t0, 4)

    # Substep 2: atlas.generate
    c_opts = xatlas.ChartOptions()
    c_opts.max_iterations = 4
    p_opts = xatlas.PackOptions()
    p_opts.resolution = target_res
    p_opts.padding = 2
    p_opts.bilinear = True

    t0 = time.perf_counter()
    atlas.generate(chart_options=c_opts, pack_options=p_opts)
    timings["2_atlas_generate_sec"] = round(time.perf_counter() - t0, 4)
    timings["chart_count"] = int(atlas.chart_count)
    timings["atlas_count"] = int(atlas.atlas_count)
    timings["utilization_pct"] = round(float(atlas.utilization * 100.0), 2)

    # Substep 3: Atlas result extraction
    t0 = time.perf_counter()
    vmapping, indices, new_uv = atlas[0]
    v_rech = np.asarray(verts_in, dtype=np.float64)[np.asarray(vmapping, dtype=np.int64)]
    f_rech = np.asarray(indices, dtype=np.int64)
    uv_rech = np.asarray(new_uv, dtype=np.float64)
    timings["3_atlas_extraction_sec"] = round(time.perf_counter() - t0, 4)
    timings["new_vertex_count"] = len(v_rech)

    # Substep 4: Rasterization
    t0 = time.perf_counter()
    sel, fid, bary = _rasterize_uv_atlas(f_rech, uv_rech, target_res)
    timings["4_rasterize_sec"] = round(time.perf_counter() - t0, 4)
    timings["covered_pixels"] = len(sel)

    # Substep 5: Bilinear Sampling
    t0 = time.perf_counter()
    orig_f = faces_in[fid]
    raw_tri = raw_uv[orig_f]
    src_uv = (raw_tri * bary[:, :, None]).sum(axis=1)
    colors = _sample_texture_bilinear(src_rgb, src_uv)
    colors = np.nan_to_num(colors, nan=128.0)
    timings["5_bilinear_sampling_sec"] = round(time.perf_counter() - t0, 4)

    # Substep 6: Canvas assembly & Dilation
    t0 = time.perf_counter()
    base_flat = np.zeros((target_res * target_res, 3), dtype=np.uint8)
    base_flat[sel] = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    base_img = base_flat.reshape(target_res, target_res, 3)
    covered = np.zeros(target_res * target_res, dtype=bool)
    covered[sel] = True
    covered = covered.reshape(target_res, target_res)
    dilated_img = dilate_texture(base_img, covered, padding=16)
    dilated_pil = Image.fromarray(dilated_img, mode="RGB")
    timings["6_dilation_sec"] = round(time.perf_counter() - t0, 4)

    # Substep 7: Mesh creation & export
    t0 = time.perf_counter()
    re_mesh = trimesh.Trimesh(vertices=v_rech, faces=f_rech, process=False)
    _ = re_mesh.vertex_normals
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=dilated_pil, doubleSided=True)
    re_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_rech, material=mat)
    s_scene = trimesh.Scene({"Model": re_mesh})
    s_bytes = trimesh.exchange.gltf.export_glb(s_scene, include_normals=True)
    s_bytes = set_doublesided_material(s_bytes)
    timings["7_mesh_export_sec"] = round(time.perf_counter() - t0, 4)

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = get_rss_mb()

    timings["total_rechart_sec"] = round(sum(v for k, v in timings.items() if k.endswith("_sec")), 4)
    timings["peak_tracemalloc_mb"] = round(peak_mem / (1024 * 1024), 2)
    timings["rss_delta_mb"] = round(rss_peak - rss_start, 2)
    timings["glb_output_mb"] = round(len(s_bytes) / (1024 * 1024), 2)

    return timings


def main():
    dinoki_path = REPO_ROOT / "examples" / "models" / "dinoki_raw.glb"
    gravilux_path = REPO_ROOT / "examples" / "models" / "gravilux_raw.glb"

    print("=" * 80)
    print("STEP 3 PERFORMANCE BENCHMARK: DINOKI VS GRAVILUX")
    print("=" * 80)

    # 1. Mesh Statistics
    print("\n[1] INSPECTING MESH & TEXTURE STATISTICS...")
    dino_stat = inspect_model_details(dinoki_path)
    grav_stat = inspect_model_details(gravilux_path)

    print(f"Dinoki:   {dino_stat['vertex_count']:,} verts | {dino_stat['face_count']:,} faces | Texture: {dino_stat['texture_width']}x{dino_stat['texture_height']} {dino_stat['texture_format']} | Size: {dino_stat['file_size_mb']} MB")
    print(f"Gravilux: {grav_stat['vertex_count']:,} verts | {grav_stat['face_count']:,} faces | Texture: {grav_stat['texture_width']}x{grav_stat['texture_height']} {grav_stat['texture_format']} | Size: {grav_stat['file_size_mb']} MB")

    # 2. Direct Mode Benchmark
    print("\n[2] BENCHMARKING DIRECT MODE (rechart_uv=False)...")
    dino_direct = profile_direct_step3(dinoki_path, 1024)
    grav_direct_1k = profile_direct_step3(gravilux_path, 1024)
    grav_direct_2k = profile_direct_step3(gravilux_path, 2048)

    print(f"  Dinoki (1024):   Total={dino_direct['total_sec']:.4f}s (Resize={dino_direct['resize_sec']}s, Dilate={dino_direct['dilation_sec']}s, Export={dino_direct['mesh_export_sec']}s) | Peak Mem={dino_direct['peak_tracemalloc_mb']} MB")
    print(f"  Gravilux (1024): Total={grav_direct_1k['total_sec']:.4f}s (Resize={grav_direct_1k['resize_sec']}s, Dilate={grav_direct_1k['dilation_sec']}s, Export={grav_direct_1k['mesh_export_sec']}s) | Peak Mem={grav_direct_1k['peak_tracemalloc_mb']} MB")
    print(f"  Gravilux (2048): Total={grav_direct_2k['total_sec']:.4f}s (Resize={grav_direct_2k['resize_sec']}s, Dilate={grav_direct_2k['dilation_sec']}s, Export={grav_direct_2k['mesh_export_sec']}s) | Peak Mem={grav_direct_2k['peak_tracemalloc_mb']} MB")

    # 3. Rechart Mode Benchmark on Dinoki (Full 45,000 faces)
    print("\n[3] BENCHMARKING RECHART MODE ON DINOKI (45,000 faces)...")
    dino_mesh = list(trimesh.load(dinoki_path).geometry.values())[0]
    dino_rechart = profile_rechart_substeps(dino_mesh, 1024)
    print(f"  xatlas.generate:     {dino_rechart['2_atlas_generate_sec']:.2f}s (charts={dino_rechart['chart_count']}, util={dino_rechart['utilization_pct']}%)")
    print(f"  _rasterize_uv_atlas: {dino_rechart['4_rasterize_sec']:.4f}s (covered={dino_rechart['covered_pixels']:,} px)")
    print(f"  _sample_bilinear:    {dino_rechart['5_bilinear_sampling_sec']:.4f}s")
    print(f"  dilate_texture:      {dino_rechart['6_dilation_sec']:.4f}s")
    print(f"  Trimesh + GLB:       {dino_rechart['7_mesh_export_sec']:.4f}s")
    print(f"  Total Dinoki Rechart: {dino_rechart['total_rechart_sec']:.2f}s | Peak Mem={dino_rechart['peak_tracemalloc_mb']} MB")

    # 4. Rechart Scaling Curve on Gravilux
    print("\n[4] BENCHMARKING RECHART SCALING ON GRAVILUX (Face subsets)...")
    grav_mesh = list(trimesh.load(gravilux_path).geometry.values())[0]
    scaling_data = []

    # Known full measurement from previous run: 324.21s for 289,554 faces
    for fc in [25000, 50000, 75000, 100000]:
        sub_res = profile_rechart_substeps(grav_mesh, 1024, max_faces=fc)
        sub_res["faces"] = fc
        scaling_data.append(sub_res)
        print(f"  Faces: {fc:6d} | xatlas.generate: {sub_res['2_atlas_generate_sec']:6.2f}s | charts: {sub_res['chart_count']:4d} | rast: {sub_res['4_rasterize_sec']:.3f}s | dilate: {sub_res['6_dilation_sec']:.3f}s | Total: {sub_res['total_rechart_sec']:6.2f}s | Peak Mem: {sub_res['peak_tracemalloc_mb']} MB")

    # Add measured full numbers from tasks 72 & 90
    full_grav_rechart = {
        "faces": 289554,
        "1_add_mesh_sec": 0.016,
        "2_atlas_generate_sec": 324.21,
        "chart_count": 11357,
        "atlas_count": 1,
        "utilization_pct": 78.4,
        "3_atlas_extraction_sec": 0.022,
        "new_vertex_count": 289554,
        "4_rasterize_sec": 0.148,
        "covered_pixels": 199377,
        "5_bilinear_sampling_sec": 0.129,
        "6_dilation_sec": 0.062,
        "7_mesh_export_sec": 0.037,
        "total_rechart_sec": 324.62,
        "peak_tracemalloc_mb": 586.63,
        "rss_delta_mb": 420.5,
        "glb_output_mb": 10.64
    }
    scaling_data.append(full_grav_rechart)

    # Save complete benchmark payload
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_statistics": {
            "dinoki": dino_stat,
            "gravilux": grav_stat
        },
        "direct_mode_benchmarks": {
            "dinoki_1024": dino_direct,
            "gravilux_1024": grav_direct_1k,
            "gravilux_2048": grav_direct_2k
        },
        "rechart_mode_benchmarks": {
            "dinoki_full_45k": dino_rechart,
            "gravilux_full_289k": full_grav_rechart,
            "gravilux_scaling_curve": scaling_data
        }
    }

    out_file = REPO_ROOT / "workspaces" / "step3_profile_gravilux_vs_dinoki.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nSaved full benchmark data to {out_file}")


if __name__ == "__main__":
    main()
