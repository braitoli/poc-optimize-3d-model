"""
step_pipeline.py

Step-by-Step 3D Model Optimization Pipeline with granular GLB stage exports & metrics.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Step 0: step_00_raw.glb (Raw input mesh ingested & inspected)
- Step 1: step_01_cleaned_grounded.glb (Cleaner & auto_ground_and_center Y=0)
- Step 2: step_02_oriented.glb (Shell orient z-buffer visibility outward CCW FrontSide)
- Step 3: step_03_texture_baked.glb (uv_baker Lanczos resample & 16px dilation)
- Step 4: step_04_palette_tagged.glb (Palette k-means 10 dominant colors embedded)
- Step 5: step_05_meshopt.glb (Node smooth_normals, weld, quantize, meshopt geometry)
- Step 6: step_06_final.glb (Basisu KTX2/WebP GPU compression, frontSide, extras)

Emits real-time NDJSON events to stdout:
{"event": "step_complete", "step": X, "stepName": "...", "file": "step_XX_....glb", "metrics": {...}}
then {"event": "pipeline_complete", "summary": {...}}, and continuously updates <output_dir>/metrics.json.
On failure the CLI prints {"event": "pipeline_error", "error": "<reason>", "errorType": "<class>", "step": <n|null>}
as its last stdout line (full traceback on stderr) and exits 1.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import trimesh

from optimizer.core.cleaner import clean_and_repair_mesh, auto_ground_and_center
from optimizer.core.shell_orient import orient_faces_by_visibility, DEFAULT_VIEWS, DEFAULT_RESOLUTION
from optimizer.core.uv_baker import SIZE_MODES, plan_uv_canvas, bake_uv_plan
from optimizer.core.palette import extract_palette, embed_gltf_extras
from optimizer.core.texture_utils import (
    extract_original_texture_info,
    preserve_mesh_textures,
    optimize_mesh_texture_for_export
)
from optimizer.core.glb_utils import set_frontside_material, set_doublesided_material, check_glb_double_sided
from optimizer.core.errors import describe_failure

MODULE_ROOT = Path(__file__).resolve().parent
INSPECT_SCRIPT = MODULE_ROOT / "inspect_metrics.mjs"
NODE_OPT_SCRIPT = MODULE_ROOT / "node" / "optimize_meshopt.mjs"


def inspect_glb_metrics(glb_path: Path) -> Dict[str, Any]:
    """Invokes inspect_metrics.mjs to extract comprehensive 3D metrics from a GLB."""
    cmd = ["node", str(INSPECT_SCRIPT), str(glb_path), "--compact"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to inspect GLB metrics for {glb_path.name}: {proc.stderr or proc.stdout}")

    stdout = proc.stdout.strip()
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    return json.loads(stdout)


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a friendly human-readable string (e.g. '54ms', '0.24s', '1.53s')."""
    if seconds < 0.1:
        ms = round(seconds * 1000)
        return f"{ms}ms" if ms > 0 else "<1ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


class StepPipeline:
    """
    Orchestrates the 7-step optimization pipeline, exporting intermediate GLBs
    and recording 3D metrics at each discrete step.
    """

    STEP_DEFINITIONS = [
        {"step": 0, "name": "raw", "file": "step_00_raw.glb", "desc": "Raw input model ingested & analyzed"},
        {"step": 1, "name": "cleaned_grounded", "file": "step_01_cleaned_grounded.glb", "desc": "Cleaned geometry & grounded at Y=0"},
        {"step": 2, "name": "oriented", "file": "step_02_oriented.glb", "desc": "Visibility-based shell orientation (outward CCW)"},
        {"step": 3, "name": "texture_baked", "file": "step_03_texture_baked.glb", "desc": "Texture resampled with Lanczos + 16px dilation"},
        {"step": 4, "name": "palette_tagged", "file": "step_04_palette_tagged.glb", "desc": "10-color dominant palette extracted & embedded"},
        {"step": 5, "name": "meshopt", "file": "step_05_meshopt.glb", "desc": "Smooth normals, weld, quantize, and EXT_meshopt_compression"},
        {"step": 6, "name": "final", "file": "step_06_final.glb", "desc": "GPU texture compression (KTX2/WebP), frontSide, final extras"}
    ]

    def __init__(
        self,
        texture_format: str = "ktx2",
        uv_mode: str = "xatlas",
        downscale: bool = True,
        size_mode: str = "exact",
        smooth_normals: Optional[bool] = None,
        double_sided: bool = False,
        preserve_textures: bool = True,
        verbose: bool = True,
        stream_events: bool = True,
        ktx2_min_vram_mb: float = 20.0
    ):
        self.texture_format = texture_format.lower()
        self.uv_mode = uv_mode.lower()
        if self.uv_mode not in ("xatlas", "uvatlas"):
            raise ValueError(f"Unsupported uv_mode '{uv_mode}' (expected 'xatlas' or 'uvatlas')")
        if not isinstance(downscale, bool):
            raise TypeError(f"downscale must be a bool, got {downscale!r}")
        self.downscale = downscale
        if size_mode not in SIZE_MODES:
            raise ValueError(f"Unsupported size_mode '{size_mode}' (expected one of {', '.join(SIZE_MODES)})")
        self.size_mode = size_mode
        # Step 6 KTX2 runs only when the Step 5 texture VRAM estimate reaches this many MB (0 = always)
        if isinstance(ktx2_min_vram_mb, bool) or not isinstance(ktx2_min_vram_mb, (int, float)):
            raise TypeError(f"ktx2_min_vram_mb must be a number, got {ktx2_min_vram_mb!r}")
        if not ktx2_min_vram_mb >= 0:
            raise ValueError(f"ktx2_min_vram_mb must be a non-negative number, got {ktx2_min_vram_mb!r}")
        self.ktx2_min_vram_mb = float(ktx2_min_vram_mb)

        if smooth_normals is None:
            self.smooth_normals = True
        else:
            self.smooth_normals = smooth_normals

        self.double_sided = double_sided
        self.preserve_textures = preserve_textures
        self.verbose = verbose
        self.stream_events = stream_events

    def log(self, msg: str):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", file=sys.stderr, flush=True)

    def _emit_step_event(
        self,
        step_idx: int,
        step_name: str,
        filename: str,
        metrics: Dict[str, Any],
        extra_data: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        total_duration_seconds: Optional[float] = None
    ):
        event_payload = {
            "event": "step_complete",
            "step": step_idx,
            "stepName": step_name,
            "file": filename,
            "metrics": metrics
        }
        if duration_seconds is not None:
            event_payload["durationSeconds"] = round(duration_seconds, 3)
            event_payload["durationFormatted"] = format_duration(duration_seconds)
        if total_duration_seconds is not None:
            event_payload["totalDurationSeconds"] = round(total_duration_seconds, 3)
        if extra_data:
            event_payload.update(extra_data)

        if self.stream_events:
            print(json.dumps(event_payload), flush=True)

    def run(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Runs the 7 steps. Any exception that escapes carries the failing step index as `.step`
        (kept if the raiser already set it, e.g. PipelineAbort(reason, step=...))."""
        self._current_step = 0
        try:
            return self._run_steps(input_path, output_dir)
        except Exception as e:
            if getattr(e, "step", None) is None:
                e.step = self._current_step
            raise

    def _run_steps(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        t0 = time.time()
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = output_dir / "metrics.json"

        steps_record: List[Dict[str, Any]] = []
        texture_resolution: Optional[str] = None  # "WxH" of the Step 3 texture, once known
        texture_format_label = self.texture_format.upper()  # real final format once Step 6 skips KTX2

        def save_and_record_metrics(
            step_idx: int,
            glb_file: Path,
            extra_info: Optional[Dict[str, Any]] = None,
            t_step_start: Optional[float] = None,
            explicit_duration: Optional[float] = None,
            duration_seconds: Optional[float] = None,
            **kwargs
        ) -> Dict[str, Any]:
            step_def = self.STEP_DEFINITIONS[step_idx]
            metrics = inspect_glb_metrics(glb_file)
            
            if duration_seconds is not None:
                step_duration = duration_seconds
            elif explicit_duration is not None:
                step_duration = explicit_duration
            elif t_step_start is not None:
                step_duration = time.perf_counter() - t_step_start
            else:
                step_duration = 0.0
            total_duration = time.perf_counter() - t_total_start

            dur_sec = round(step_duration, 3)
            dur_fmt = format_duration(step_duration)
            tot_sec = round(total_duration, 3)

            # Enrich metrics dictionary with precise timing indicators
            metrics["durationSeconds"] = dur_sec
            metrics["durationFormatted"] = dur_fmt
            metrics["totalDurationSeconds"] = tot_sec
            metrics["durationMs"] = round(step_duration * 1000, 1)

            if extra_info:
                metrics.update(extra_info)

            step_entry = {
                "step": step_idx,
                "stepName": step_def["name"],
                "file": step_def["file"],
                "description": step_def["desc"],
                "durationSeconds": dur_sec,
                "durationFormatted": dur_fmt,
                "totalDurationSeconds": tot_sec,
                "durationMs": round(step_duration * 1000, 1),
                "metrics": metrics
            }
            if extra_info:
                step_entry["details"] = extra_info
            steps_record.append(step_entry)

            # Write updated metrics.json
            payload = {
                "success": True,
                "model": input_path.name,
                "outputDir": str(output_dir),
                "resolution": texture_resolution,
                "downscale": self.downscale,
                "sizeMode": self.size_mode,
                "uvMode": self.uv_mode,
                "textureFormat": texture_format_label,
                "steps": steps_record,
                "lastCompletedStep": step_idx
            }
            metrics_json_path.write_text(json.dumps(payload, indent=2))

            # Emit streaming NDJSON event
            self._emit_step_event(
                step_idx,
                step_def["name"],
                step_def["file"],
                metrics,
                extra_data=extra_info,
                duration_seconds=dur_sec,
                total_duration_seconds=tot_sec
            )
            return metrics

        self.log("=" * 68)
        self.log(f"🚀 STEP-BY-STEP 3D OPTIMIZATION PIPELINE: {input_path.name}")
        self.log(f"   Target Directory: {output_dir}")
        self.log(
            f"   Downscale: {'ON' if self.downscale else 'OFF'} | Size Mode: {self.size_mode} | "
            f"Format: {self.texture_format.upper()} | UV Mode: {self.uv_mode.upper()}"
        )
        self.log("=" * 68)

        # =====================================================================
        # STEP 0: Raw Model Ingestion & Baseline Metrics
        # =====================================================================
        self.log("▶️ [Step 0/6] Ingesting Raw GLB Model...")
        t_s0 = time.perf_counter()
        step0_file = output_dir / "step_00_raw.glb"
        if input_path.resolve() != step0_file.resolve():
            shutil.copy2(input_path, step0_file)
        m0 = save_and_record_metrics(0, step0_file, t_step_start=t_s0)
        initial_faces = m0["faces"]
        initial_verts = m0["vertices"]
        initial_bytes = m0["fileSizeBytes"]
        orig_tex_info = extract_original_texture_info(step0_file, metrics=m0)
        self.log(f"   ✓ Step 0 complete ({m0['durationFormatted']}): {initial_faces:,} faces, {initial_verts:,} verts, {m0['fileSizeFormatted']} (texture: {orig_tex_info.get('default_format')})")

        # Auto-detect doubleSided from input materials (or CLI flag)
        if check_glb_double_sided(input_path) and not self.double_sided:
            self.log("   ℹ️ Auto-detected doubleSided=True from input materials (preserving thin shells & armor)")
            self.double_sided = True

        # =====================================================================
        # STEP 1: Cleaner & Auto Grounding (Y=0, X/Z Centered)
        # =====================================================================
        self._current_step = 1
        self.log("▶️ [Step 1/6] Cleaning Geometry & Auto-Grounding at Y=0...")
        t_s1 = time.perf_counter()
        raw_mesh = trimesh.load(str(input_path), force="mesh", process=False)
        cleaned_mesh = clean_and_repair_mesh(raw_mesh)
        grounded_mesh, translation = auto_ground_and_center(cleaned_mesh)
        if self.preserve_textures:
            preserve_mesh_textures(grounded_mesh, orig_tex_info)

        step1_scene = trimesh.Scene({"Model": grounded_mesh})
        step1_bytes = trimesh.exchange.gltf.export_glb(step1_scene, include_normals=True)
        step1_file = output_dir / "step_01_cleaned_grounded.glb"
        step1_file.write_bytes(step1_bytes)
        m1 = save_and_record_metrics(1, step1_file, {"translationApplied": translation.tolist()}, t_step_start=t_s1)
        self.log(f"   ✓ Step 1 complete ({m1['durationFormatted']}): Grounded at Y=0 (shift: {np.round(translation, 3).tolist()})")

        # =====================================================================
        # STEP 2: Visibility Z-Buffer Shell Orient (Outward CCW Winding)
        # =====================================================================
        self._current_step = 2
        self.log("▶️ [Step 2/6] Orienting Shells via Visibility Z-Buffer (CCW)...")
        t_s2 = time.perf_counter()
        orient_stats: Dict[str, Any] = {}
        oriented_faces = orient_faces_by_visibility(
            grounded_mesh.vertices,
            grounded_mesh.faces,
            views=DEFAULT_VIEWS,
            resolution=DEFAULT_RESOLUTION,
            stats=orient_stats
        )
        grounded_mesh.faces = oriented_faces
        if self.preserve_textures:
            preserve_mesh_textures(grounded_mesh, orig_tex_info)

        step2_scene = trimesh.Scene({"Model": grounded_mesh})
        step2_bytes = trimesh.exchange.gltf.export_glb(step2_scene, include_normals=True)
        step2_file = output_dir / "step_02_oriented.glb"
        step2_file.write_bytes(step2_bytes)
        m2 = save_and_record_metrics(2, step2_file, orient_stats, t_step_start=t_s2)
        self.log(f"   ✓ Step 2 complete ({m2['durationFormatted']}): Flipped {orient_stats.get('faces_flipped', 0)} faces to outward CCW")

        # =====================================================================
        # STEP 3: Texture Baking / Resampling & 16px Dilation
        # =====================================================================
        self._current_step = 3
        t_s3 = time.perf_counter()
        raw_uv = getattr(raw_mesh.visual, "uv", None)
        if raw_uv is None:
            raw_uv = getattr(grounded_mesh.visual, "uv", np.zeros((len(grounded_mesh.vertices), 2)))

        raw_tex_img = orig_tex_info.get("base_image")
        if raw_tex_img is None and hasattr(grounded_mesh.visual, "material") and hasattr(grounded_mesh.visual.material, "baseColorTexture"):
            raw_tex_img = grounded_mesh.visual.material.baseColorTexture
        if raw_tex_img is None:
            raw_tex_img = Image.new("RGB", (1024, 1024), (200, 200, 200))

        orig_w, orig_h = raw_tex_img.size
        original_resolution = f"{orig_w}x{orig_h}"

        # Downscale on: size the re-chart canvas at the source's 1:1 texel density (size_mode), but keep
        # the original UVs & texture when that canvas is not smaller than the original texture.
        plan: Optional[Dict[str, Any]] = None
        if self.downscale:
            self.log(
                f"▶️ [Step 3/6] Sizing {self.uv_mode.upper()} re-chart canvas at 1:1 texel density "
                f"(size mode: {self.size_mode})..."
            )
            plan = plan_uv_canvas(
                grounded_mesh,
                source_image=raw_tex_img,
                source_uv=raw_uv,
                size_mode=self.size_mode,
                unwrap_method=self.uv_mode
            )
            canvas = plan["final_resolution"]
            rechart = canvas * canvas < orig_w * orig_h
            decision = (
                f"{'rechart' if rechart else 'kept_original'}: {self.size_mode} {canvas}x{canvas} "
                f"{'<' if rechart else '>='} original {original_resolution} (fit {plan['fit_resolution']})"
            )
        else:
            self.log("▶️ [Step 3/6] Downscale off: keeping original UVs & texture...")
            rechart = False
            decision = "kept_original: downscale off"
        self.log(f"   {decision}")

        if rechart:
            baked_mesh, dilated_pil, uv_stats = bake_uv_plan(
                grounded_mesh,
                plan,
                source_image=raw_tex_img,
                source_uv=raw_uv,
                dilation_padding=16,
                double_sided=self.double_sided
            )
            pref_fmt = "PNG"  # PNG Lossless export for Step 3
            texel_density_ratio = uv_stats["texel_density_ratio"]
        else:
            # Step 2 mesh as is (original UVs & faces) with the original texture bitstream
            baked_mesh = grounded_mesh
            preserve_mesh_textures(baked_mesh, orig_tex_info)
            dilated_pil = raw_tex_img
            pref_fmt = "ORIGINAL"
            uv_stats = {}
            texel_density_ratio = 1.0

        # FrontSide rendering (doubleSided=False by default)
        if hasattr(baked_mesh, "visual") and hasattr(baked_mesh.visual, "material") and baked_mesh.visual.material is not None:
            baked_mesh.visual.material.doubleSided = self.double_sided

        # Optimize texture before export (PNG Lossless or Original Bitstream Passthrough)
        opt_pil = optimize_mesh_texture_for_export(
            baked_mesh,
            orig_tex_info=orig_tex_info,
            preferred_format=pref_fmt
        )
        if opt_pil is not None:
            dilated_pil = opt_pil

        step3_scene = trimesh.Scene({"Model": baked_mesh})
        step3_bytes = trimesh.exchange.gltf.export_glb(step3_scene, include_normals=True)
        if self.double_sided:
            step3_bytes = set_doublesided_material(step3_bytes)
        else:
            step3_bytes = set_frontside_material(step3_bytes)
        step3_file = output_dir / "step_03_texture_baked.glb"
        step3_file.write_bytes(step3_bytes)
        tex_fmt = getattr(dilated_pil, "format", "PNG")
        texture_resolution = f"{dilated_pil.size[0]}x{dilated_pil.size[1]}"
        # Steps 5-6 never resize: the max dimension is the Step 3 texture's own
        texture_max_dim = max(dilated_pil.size)

        step3_extra = {
            "uvMode": self.uv_mode,
            "downscale": self.downscale,
            "sizeMode": self.size_mode,
            "fitResolution": plan["fit_resolution"] if plan is not None else None,
            "finalResolution": texture_resolution,
            "originalResolution": original_resolution,
            "downscaled": rechart,
            "decision": decision,
            "texelDensityRatio": texel_density_ratio,
            "textureResolution": texture_resolution,
            "textureFormat": tex_fmt,
            "dilationPadding": uv_stats["dilation_padding"] if rechart else None,
            "uvCoverageRatio": uv_stats["uvCoverageRatio"] if rechart else None,
            "doubleSided": self.double_sided,
            "uvMetrics": uv_stats
        }
        m3 = save_and_record_metrics(3, step3_file, step3_extra, t_step_start=t_s3)
        self.log(
            f"   ✓ Step 3 complete ({m3['durationFormatted']}): Texture {texture_resolution} ({tex_fmt}) | "
            f"Downscaled: {rechart} ({original_resolution} -> {texture_resolution}) | "
            f"Texel density ratio: {texel_density_ratio:.3f}"
        )

        # =====================================================================
        # STEP 4: Palette Tagging (10 Dominant Colors via K-Means)
        # =====================================================================
        self._current_step = 4
        self.log("▶️ [Step 4/6] Extracting 10 Dominant Colors & Embedding Extras...")
        t_s4 = time.perf_counter()
        img_rgb = np.asarray(dilated_pil)
        v_uv = baked_mesh.visual.uv % 1.0
        px = np.clip((v_uv[:, 0] * (img_rgb.shape[1] - 1)).astype(int), 0, img_rgb.shape[1] - 1)
        py = np.clip(((1.0 - v_uv[:, 1]) * (img_rgb.shape[0] - 1)).astype(int), 0, img_rgb.shape[0] - 1)
        surface_pixels = img_rgb[py, px]

        palette_data = extract_palette(image=dilated_pil, sample_pixels=surface_pixels, n_colors=10)
        step4_bytes = embed_gltf_extras(step3_bytes, {
            "palette": palette_data["palette"],
            "primaryColor": palette_data["primaryColor"],
            "paletteDetails": palette_data["paletteDetails"]
        })
        step4_file = output_dir / "step_04_palette_tagged.glb"
        step4_file.write_bytes(step4_bytes)
        m4 = save_and_record_metrics(4, step4_file, {
            "primaryColor": palette_data["primaryColor"],
            "paletteCount": len(palette_data["palette"])
        }, t_step_start=t_s4)
        self.log(f"   ✓ Step 4 complete ({m4['durationFormatted']}): Primary color: {palette_data['primaryColor']} | Palette: {palette_data['palette']}")

        # =====================================================================
        # STEP 5: Smooth Normals & EXT_meshopt_compression Geometry
        # =====================================================================
        self._current_step = 5
        self.log("▶️ [Step 5/6] Node.js Smooth Normals + Weld + Quantize + Meshopt...")
        t_s5 = time.perf_counter()
        step5_file = output_dir / "step_05_meshopt.glb"
        node_cmd_step5 = [
            "node", str(NODE_OPT_SCRIPT),
            str(step4_file),
            str(step5_file),
            "--no-ktx2",
            "--pos-bits", "14",
            "--weld", "0.0001",
            "--reorder",
            "--meshopt",
            "--texture-max-dim", str(texture_max_dim),
            "--json"
        ]
        if self.smooth_normals:
            node_cmd_step5.append("--smooth-normals")
        else:
            node_cmd_step5.append("--no-smooth-normals")

        if not self.double_sided:
            node_cmd_step5.append("--single-sided")
        else:
            node_cmd_step5.append("--keep-double-sided")

        proc5 = subprocess.run(node_cmd_step5, capture_output=True, text=True)
        if proc5.returncode != 0 or not step5_file.exists():
            raise RuntimeError(f"Step 5 Meshopt geometry compression failed: {proc5.stderr or proc5.stdout}")

        m5 = save_and_record_metrics(5, step5_file, t_step_start=t_s5)
        self.log(f"   ✓ Step 5 complete ({m5['durationFormatted']}): Geometry compressed ({m5['faces']:,} faces preserved 100%, {m5['fileSizeFormatted']})")

        # Small textures: KTX2 UASTC grows the file while the uncompressed VRAM is already small,
        # so Step 6 keeps the Step 5 textures when the texture VRAM estimate is below the threshold.
        texture_vram_bytes = m5["totalGpuVramBytes"]
        texture_vram_mb = texture_vram_bytes / (1024 * 1024)
        skip_ktx2 = self.texture_format == "ktx2" and texture_vram_mb < self.ktx2_min_vram_mb
        vram_vs_threshold = f"{texture_vram_mb:.2f} MB {'<' if skip_ktx2 else '>='} {self.ktx2_min_vram_mb:g} MB"
        ktx2_reason = f"texture VRAM {vram_vs_threshold}"
        if self.texture_format == "ktx2":
            self.log(
                f"   Texture VRAM estimate: {vram_vs_threshold} -> "
                f"{'Step 6 KTX2 will be skipped' if skip_ktx2 else 'Step 6 KTX2 UASTC'}"
            )

        # =====================================================================
        # STEP 6: KTX2 / WebP / Original GPU Compression, FrontSide Material, Extras
        # =====================================================================
        self._current_step = 6
        t_s6 = time.perf_counter()
        is_original_format = self.texture_format in ("original", "passthrough", "raw")
        if is_original_format:
            self.log("▶️ [Step 6/6] Finalizing Model (Original Texture Bitstream Pass-through 100% Lossless)...")
            final_bytes = step5_file.read_bytes()
        elif skip_ktx2:
            self.log(
                f"▶️ [Step 6/6] KTX2 GPU Texture Compression SKIPPED ({ktx2_reason}): "
                f"final model keeps the Step 5 textures..."
            )
            final_bytes = step5_file.read_bytes()
        else:
            self.log(f"▶️ [Step 6/6] Basis Universal GPU Texture Compression ({self.texture_format.upper()})...")
            step6_temp_file = output_dir / "step_06_temp.glb"
            node_cmd_step6 = [
                "node", str(NODE_OPT_SCRIPT),
                str(step5_file),
                str(step6_temp_file),
                "--textures-only",
                "--meshopt",
                "--texture-max-dim", str(texture_max_dim),
                "--json"
            ]

            if self.texture_format == "webp":
                node_cmd_step6.extend(["--webp", "--webp-quality", "85"])
            else:
                cpu_threads = str(os.cpu_count() or 4)
                ktx2_rdo = "1.0"
                ktx2_level = "2"
                node_cmd_step6.extend([
                    "--ktx2",
                    "--ktx2-mode", "uastc",
                    "--ktx2-level", ktx2_level,
                    "--ktx2-rdo", ktx2_rdo,
                    "--ktx2-rdo-d", "2048",
                    "--ktx2-threads", cpu_threads
                ])

            if not self.double_sided:
                node_cmd_step6.append("--single-sided")
            else:
                node_cmd_step6.append("--keep-double-sided")

            proc6 = subprocess.run(node_cmd_step6, capture_output=True, text=True)
            if proc6.returncode != 0 or not step6_temp_file.exists():
                raise RuntimeError(f"Step 6 Texture compression failed: {proc6.stderr or proc6.stdout}")

            final_bytes = step6_temp_file.read_bytes()
            step6_temp_file.unlink(missing_ok=True)

        if not self.double_sided:
            final_bytes = set_frontside_material(final_bytes)

        final_extras = {
            "palette": palette_data["palette"],
            "primaryColor": palette_data["primaryColor"],
            "paletteDetails": palette_data["paletteDetails"],
            "resolution": texture_resolution,
            "texture_format": self.texture_format.upper(),
            "policy": "STRICT 0-DECIMATION (--ratio 1.0)",
            "tool": "poc-optimize-3d-model v1.0.0"
        }
        step6_extra = None
        if self.texture_format == "ktx2":
            if skip_ktx2:
                # The final file keeps the Step 5 textures: record their real format(s), never "KTX2"
                final_extras["texture_format"] = "+".join(dict.fromkeys(t["format"] for t in m5["textures"]))
                final_extras["gpu_compression"] = f"skipped: {ktx2_reason}"
                texture_format_label = final_extras["texture_format"]
            else:
                final_extras["gpu_compression"] = "ktx2 uastc"
            step6_extra = {
                "gpuCompressionSkipped": skip_ktx2,
                "gpuCompressionReason": ktx2_reason,
                "textureVramEstimateBytes": texture_vram_bytes
            }
        final_bytes = embed_gltf_extras(final_bytes, final_extras)
        step6_file = output_dir / "step_06_final.glb"
        step6_file.write_bytes(final_bytes)

        m6 = save_and_record_metrics(6, step6_file, step6_extra, t_step_start=t_s6)
        self.log(f"   ✓ Step 6 complete ({m6['durationFormatted']}): Final GLB ready ({m6['fileSizeFormatted']}, GPU VRAM: {m6['totalGpuVramFormatted']})")

        # =====================================================================
        # Pipeline Summary & Finalization
        # =====================================================================
        total_perf_elapsed = time.perf_counter() - t_total_start
        elapsed = time.time() - t0
        final_bytes_count = m6["fileSizeBytes"]
        saved_bytes = initial_bytes - final_bytes_count
        saved_pct = round((saved_bytes / initial_bytes) * 100, 2)
        initial_vram = m0.get("totalGpuVramBytes", 0)
        final_vram = m6.get("totalGpuVramBytes", 0)
        vram_saved_pct = round((1 - final_vram / initial_vram) * 100, 2) if initial_vram > 0 else 0.0

        summary = {
            "model": input_path.name,
            "outputDir": str(output_dir),
            "elapsedSeconds": round(total_perf_elapsed, 3),
            "elapsedFormatted": format_duration(total_perf_elapsed),
            "elapsedMs": round(total_perf_elapsed * 1000, 1),
            "totalDurationSeconds": round(total_perf_elapsed, 3),
            "totalDurationFormatted": format_duration(total_perf_elapsed),
            "stepDurations": {
                s["stepName"]: {
                    "step": s["step"],
                    "file": s["file"],
                    "durationSeconds": s.get("durationSeconds", 0.0),
                    "durationFormatted": s.get("durationFormatted", "0s"),
                    "totalDurationSeconds": s.get("totalDurationSeconds", 0.0),
                    "seconds": s.get("durationSeconds", 0.0),
                    "ms": s.get("durationMs", 0.0)
                }
                for s in steps_record
            },
            "initialSizeBytes": initial_bytes,
            "finalSizeBytes": final_bytes_count,
            "savedBytes": saved_bytes,
            "savedPercent": saved_pct,
            "initialFaces": initial_faces,
            "finalFaces": m6["faces"],
            "facesPreservedPercent": round((m6["faces"] / initial_faces) * 100, 2),
            "zeroDecimationVerified": m6["faces"] == initial_faces,
            "initialVertices": initial_verts,
            "finalVertices": m6["vertices"],
            "initialGpuVramBytes": initial_vram,
            "finalGpuVramBytes": final_vram,
            "gpuVramSavedPercent": vram_saved_pct,
            "primaryColor": palette_data["primaryColor"],
            "palette": palette_data["palette"],
            "files": [step["file"] for step in self.STEP_DEFINITIONS]
        }

        final_payload = {
            "success": True,
            "summary": summary,
            "steps": steps_record
        }
        metrics_json_path.write_text(json.dumps(final_payload, indent=2))

        # Emit pipeline_completed event
        if self.stream_events:
            print(json.dumps({"event": "pipeline_complete", "summary": summary}), flush=True)

        self.log("=" * 68)
        self.log(f"🎉 PIPELINE COMPLETED IN {total_perf_elapsed:.2f}s!")
        self.log(f"   Size: {m0['fileSizeFormatted']} -> {m6['fileSizeFormatted']} (Saved {saved_pct}%)")
        self.log(f"   GPU VRAM: {m0['totalGpuVramFormatted']} -> {m6['totalGpuVramFormatted']} (Saved {vram_saved_pct}%)")
        self.log(f"   Triangles: {m6['faces']:,} / {initial_faces:,} (100% Zero-Decimation Verified: {summary['zeroDecimationVerified']})")
        self.log(f"   Metrics saved to: {metrics_json_path}")
        self.log("=" * 68)

        return final_payload


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Decimation Step-by-Step 3D Model Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Path to raw source .glb model")
    parser.add_argument("--output-dir", "-o", required=True, help="Destination directory for 7 GLB step files and metrics.json")
    parser.add_argument("--format", "-f", choices=["ktx2", "webp", "original", "passthrough"], default="ktx2", help="GPU texture compression format ('original' for 100%% bit-for-bit lossless pass-through)")
    parser.add_argument(
        "--uv-mode",
        choices=["xatlas", "uvatlas"],
        default="xatlas",
        help="UV unwrapping and layout mode: 'xatlas' (re-chart with xatlas), 'uvatlas' (Microsoft UVAtlas isochart unwrap)"
    )
    parser.add_argument(
        "--downscale",
        choices=["on", "off"],
        default="on",
        help="'on': re-chart & pack UV islands on a canvas sized by --size-mode (original kept when that canvas "
             "is not smaller than the original texture); 'off': keep original UVs, texture and resolution"
    )
    parser.add_argument(
        "--size-mode",
        choices=list(SIZE_MODES),
        default="exact",
        help="Re-chart canvas at the source's 1:1 texel density: 'exact' (smallest square, multiple of 4), "
             "'pot-up' (power of two, 1:1 or better), 'pot-down' (power of two, islands scaled down)"
    )
    parser.add_argument("--smooth-normals", dest="smooth_normals", action="store_true", default=None, help="Force angle-weighted normal smoothing across seams")
    parser.add_argument("--no-smooth-normals", dest="smooth_normals", action="store_false", default=None, help="Disable angle-weighted normal smoothing across seams")
    parser.add_argument("--double-sided", action="store_true", help="Keep double-sided materials instead of forcing single-sided FrontSide")
    parser.add_argument("--no-preserve-textures", action="store_true", help="Disable texture preservation in Steps 1 & 2")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress stderr logs and only stream NDJSON events")

    args = parser.parse_args()

    try:
        pipeline = StepPipeline(
            texture_format=args.format,
            uv_mode=args.uv_mode,
            downscale=(args.downscale == "on"),
            size_mode=args.size_mode,
            smooth_normals=args.smooth_normals,
            double_sided=args.double_sided,
            preserve_textures=not args.no_preserve_textures,
            verbose=not args.quiet,
            stream_events=True
        )
        pipeline.run(Path(args.input), Path(args.output_dir))
    except Exception as e:
        # Full traceback for debugging on stderr; one parseable event (reason, type, step) on stdout
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"event": "pipeline_error", **describe_failure(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
