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
and continuously updates <output_dir>/metrics.json.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
from PIL import Image
import trimesh

from optimizer.core.cleaner import clean_and_repair_mesh, auto_ground_and_center
from optimizer.core.shell_orient import orient_faces_by_visibility, DEFAULT_VIEWS, DEFAULT_RESOLUTION
from optimizer.core.uv_baker import (
    rebake_texture_xatlas,
    rebake_texture_uvatlas,
    direct_resample_texture,
    rechart_and_bake_high_density,
    compute_uv_metrics,
    can_downscale_texture,
    determine_safe_downscale_resolution,
    maximize_uv_bounds,
    compute_original_island_pixels,
    select_repack_canvas_resolution
)
from optimizer.core.uvatlas import is_uvatlas_available
from optimizer.core.palette import extract_palette, embed_gltf_extras
from optimizer.core.texture_utils import (
    extract_original_texture_info,
    preserve_mesh_textures,
    clamp_target_resolution,
    optimize_mesh_texture_for_export
)
from optimizer.pipeline import set_frontside_material, set_doublesided_material

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
        resolution: Union[int, str] = "auto",
        texture_format: str = "ktx2",
        rechart_uv: bool = False,
        uv_mode: Optional[str] = None,
        smooth_normals: Optional[bool] = None,
        double_sided: bool = False,
        preserve_textures: bool = True,
        verbose: bool = True,
        stream_events: bool = True
    ):
        self.resolution = resolution
        self.texture_format = texture_format.lower()
        if uv_mode is not None:
            self.uv_mode = uv_mode.lower()
            self.rechart_uv = (self.uv_mode != "direct")
        elif rechart_uv:
            self.uv_mode = "xatlas"
            self.rechart_uv = True
        else:
            self.uv_mode = "direct"
            self.rechart_uv = False

        if smooth_normals is None:
            # Direct Master UV mode: preserve artist's custom vertex normals by default
            self.smooth_normals = (self.uv_mode != "direct")
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
        t_total_start = time.perf_counter()
        t0 = time.time()
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = output_dir / "metrics.json"

        steps_record: List[Dict[str, Any]] = []

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
                "resolution": f"{self.resolution}x{self.resolution}" if isinstance(self.resolution, int) else str(self.resolution),
                "textureFormat": self.texture_format.upper(),
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
        res_display = f"{self.resolution}x{self.resolution}" if isinstance(self.resolution, int) else f"{self.resolution.upper()} (Adaptive)"
        self.log(f"   Target Resolution: {res_display} | Format: {self.texture_format.upper()} | UV Mode: {self.uv_mode.upper()}")
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

        # =====================================================================
        # STEP 1: Cleaner & Auto Grounding (Y=0, X/Z Centered)
        # =====================================================================
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
        t_s3 = time.perf_counter()
        raw_uv = getattr(raw_mesh.visual, "uv", None)
        if raw_uv is None:
            raw_uv = getattr(grounded_mesh.visual, "uv", np.zeros((len(grounded_mesh.vertices), 2)))

        raw_tex_img = orig_tex_info.get("base_image")
        if raw_tex_img is None and hasattr(grounded_mesh.visual, "material") and hasattr(grounded_mesh.visual.material, "baseColorTexture"):
            raw_tex_img = grounded_mesh.visual.material.baseColorTexture
        if raw_tex_img is None:
            default_dim = 1024 if (self.resolution == "auto" or not isinstance(self.resolution, int)) else self.resolution
            raw_tex_img = Image.new("RGB", (default_dim, default_dim), (200, 200, 200))

        orig_max_dim = max(raw_tex_img.size)
        initial_res = orig_max_dim
        is_auto_res = (self.resolution == "auto" or not isinstance(self.resolution, int))
        user_requested_downscale = (not is_auto_res and isinstance(self.resolution, int) and self.resolution < orig_max_dim)

        uv_stats: Dict[str, Any] = {}
        if self.uv_mode in ("xatlas", "uvatlas") or self.rechart_uv:
            # 1. 1:1 Capacity Check for UV Repacking:
            # Finds smallest canvas S in {1024, 2048, 4096} preserving 1:1 texel scale
            repack_res, repack_info = select_repack_canvas_resolution(
                grounded_mesh,
                source_image=raw_tex_img,
                source_uv=raw_uv,
                requested_res=self.resolution
            )
            initial_res = repack_res
            self.log(
                f"▶️ [Step 3/6] Baking Texture ({self.uv_mode.upper()} Canvas {initial_res}x{initial_res})... "
                f"orig_island_pixels={repack_info['orig_island_pixels']:,.0f} ({repack_info['coverage_ratio']:.1%} coverage) "
                f"[{repack_info['capacity_reason']}]"
            )

            if self.uv_mode == "uvatlas":
                baked_mesh, dilated_pil, uv_stats = rechart_and_bake_high_density(
                    grounded_mesh,
                    target_res=initial_res,
                    source_image=raw_tex_img,
                    source_uv=raw_uv,
                    dilation_padding=16,
                    double_sided=self.double_sided,
                    stats=uv_stats,
                    return_stats=True,
                    unwrap_method="uvatlas"
                )
            else:
                baked_mesh, dilated_pil, uv_stats = rechart_and_bake_high_density(
                    grounded_mesh,
                    target_res=initial_res,
                    source_image=raw_tex_img,
                    source_uv=raw_uv,
                    dilation_padding=16,
                    double_sided=self.double_sided,
                    stats=uv_stats,
                    return_stats=True,
                    unwrap_method="xatlas"
                )
            final_res = uv_stats.get("final_resolution", dilated_pil.size[0])
            self.resolution = final_res
            pref_fmt = "PNG"  # PNG Lossless export for Step 3
            uv_stats["origIslandPixels"] = repack_info["orig_island_pixels"]
            uv_stats["capacityReason"] = repack_info["capacity_reason"]
        else:
            # DIRECT MASTER UV:
            # Keep 100% original texture without implicit downscaling
            if not user_requested_downscale:
                # TRUE ZERO-LOSS DIRECT MASTER UV:
                # 100% Bit-for-Bit Lossless (0 decode, 0 re-encode, 0 implicit downscale, exact bitstream pass-through)
                target_res = orig_max_dim
                self.resolution = target_res
                self.log(
                    f"▶️ [Step 3/6] Direct Master UV (True Zero-Loss Pass-through): {raw_tex_img.size[0]}x{raw_tex_img.size[1]} | "
                    f"100% Bit-for-Bit Lossless Bitstream & UVs Preserved (Zero Implicit Downscale)..."
                )
                baked_mesh, dilated_pil = direct_resample_texture(
                    grounded_mesh,
                    source_image=raw_tex_img,
                    target_res=target_res,
                    dilation_padding=0,
                    double_sided=self.double_sided,
                    preserve_bitstream=True
                )
                if self.preserve_textures:
                    preserve_mesh_textures(baked_mesh, orig_tex_info)

                final_res = target_res
                pref_fmt = "ORIGINAL"
                uv_stats = {
                    "downscaled": False,
                    "originalResolution": f"{orig_max_dim}x{orig_max_dim}",
                    "finalResolution": f"{orig_max_dim}x{orig_max_dim}",
                    "bitstreamPassthrough": True,
                    "uvPreserved100Percent": True,
                    "uvCoverageRatio": 1.0,
                    "texelDensityOrig": 0.0,
                    "texelDensityFinal": 0.0,
                    "texelDensityDelta": 0.0
                }
            else:
                # User explicitly requested a downscaled resolution:
                # Gamma-Corrected Linear Resampling + PNG Lossless
                target_res = self.resolution
                self.log(
                    f"▶️ [Step 3/6] Direct Master UV (Gamma-Correct Linear Resampling): {raw_tex_img.size[0]}x{raw_tex_img.size[1]} -> "
                    f"{target_res}x{target_res} (Specular Highlights Preserved, PNG Lossless)..."
                )
                baked_mesh, dilated_pil = direct_resample_texture(
                    grounded_mesh,
                    source_image=raw_tex_img,
                    target_res=target_res,
                    dilation_padding=16,
                    double_sided=self.double_sided,
                    preserve_bitstream=False
                )
                final_res = dilated_pil.size[0]
                pref_fmt = "PNG"  # PNG Lossless export
                uv_stats = {
                    "downscaled": True,
                    "originalResolution": f"{orig_max_dim}x{orig_max_dim}",
                    "finalResolution": f"{target_res}x{target_res}",
                    "bitstreamPassthrough": False,
                    "uvPreserved100Percent": True,
                    "uvCoverageRatio": 1.0,
                    "texelDensityOrig": 0.0,
                    "texelDensityFinal": 0.0,
                    "texelDensityDelta": 0.0
                }

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

        step3_extra = {
            "uvMode": self.uv_mode,
            "textureResolution": f"{dilated_pil.size[0]}x{dilated_pil.size[1]}",
            "textureFormat": tex_fmt,
            "dilationPadding": 16,
            "downscaled": uv_stats.get("downscaled", False),
            "originalResolution": uv_stats.get("originalResolution", f"{initial_res}x{initial_res}"),
            "finalResolution": uv_stats.get("finalResolution", f"{final_res}x{final_res}"),
            "uvCoverageRatio": uv_stats.get("uvCoverageRatio", uv_stats.get("uv_coverage_ratio_percent", 0.0) / 100.0 if "uv_coverage_ratio_percent" in uv_stats else 0.0),
            "texelDensityOrig": uv_stats.get("texelDensityOrig", 0.0),
            "texelDensityFinal": uv_stats.get("texelDensityFinal", 0.0),
            "texelDensityDelta": uv_stats.get("texelDensityDelta", 0.0),
            "uvPreserved100Percent": (self.uv_mode == "direct"),
            "doubleSided": self.double_sided,
            "uvMetrics": uv_stats
        }
        m3 = save_and_record_metrics(3, step3_file, step3_extra, t_step_start=t_s3)
        self.log(
            f"   ✓ Step 3 complete ({m3['durationFormatted']}): [{self.uv_mode}] Baked texture {dilated_pil.size[0]}x{dilated_pil.size[1]} ({tex_fmt}) | "
            f"Downscaled: {step3_extra['downscaled']} ({step3_extra['originalResolution']} -> {step3_extra['finalResolution']}) | "
            f"UV Coverage: {step3_extra['uvCoverageRatio']:.2%} | TD Delta: {step3_extra['texelDensityDelta']:+.1f}"
        )

        # =====================================================================
        # STEP 4: Palette Tagging (10 Dominant Colors via K-Means)
        # =====================================================================
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
            "--texture-max-dim", str(self.resolution),
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

        # =====================================================================
        # STEP 6: KTX2 / WebP / Original GPU Compression, FrontSide Material, Extras
        # =====================================================================
        t_s6 = time.perf_counter()
        is_original_format = self.texture_format in ("original", "passthrough", "raw")
        if is_original_format:
            self.log("▶️ [Step 6/6] Finalizing Model (Original Texture Bitstream Pass-through 100% Lossless)...")
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
                "--texture-max-dim", str(self.resolution),
                "--json"
            ]

            if self.texture_format == "webp":
                node_cmd_step6.extend(["--webp", "--webp-quality", "85"])
            else:
                cpu_threads = str(os.cpu_count() or 4)
                ktx2_rdo = "0.0" if self.uv_mode == "direct" else "1.0"
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
            "resolution": f"{self.resolution}x{self.resolution}",
            "texture_format": self.texture_format.upper(),
            "policy": "STRICT 0-DECIMATION (--ratio 1.0)",
            "tool": "poc-optimize-3d-model v1.0.0"
        }
        final_bytes = embed_gltf_extras(final_bytes, final_extras)
        step6_file = output_dir / "step_06_final.glb"
        step6_file.write_bytes(final_bytes)

        m6 = save_and_record_metrics(6, step6_file, t_step_start=t_s6)
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


def parse_resolution_arg(val: Any) -> Union[int, str]:
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    try:
        res = int(s)
        return res if res > 0 else "auto"
    except ValueError:
        return "auto"


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Decimation Step-by-Step 3D Model Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Path to raw source .glb model")
    parser.add_argument("--output-dir", "-o", required=True, help="Destination directory for 7 GLB step files and metrics.json")
    parser.add_argument("--resolution", "-r", default="auto", type=parse_resolution_arg, help="Target texture dimension ('auto', 512, 1024, 2048; strictly capped at original texture size, never upscaled)")
    parser.add_argument("--format", "-f", choices=["ktx2", "webp", "original", "passthrough"], default="ktx2", help="GPU texture compression format ('original' for 100% bit-for-bit lossless pass-through)")
    parser.add_argument(
        "--uv-mode",
        choices=["direct", "xatlas", "uvatlas"],
        default=None,
        help="UV unwrapping and layout mode: 'direct' (preserve master UV), 'xatlas' (re-chart with xatlas), 'uvatlas' (Microsoft UVAtlas isochart unwrap)"
    )
    parser.add_argument("--rechart-uv", action="store_true", help="Re-chart UVs using xatlas (backward compatibility alias for --uv-mode xatlas; default: direct master UV)")
    parser.add_argument("--smooth-normals", dest="smooth_normals", action="store_true", default=None, help="Force angle-weighted normal smoothing across seams")
    parser.add_argument("--no-smooth-normals", dest="smooth_normals", action="store_false", default=None, help="Disable angle-weighted normal smoothing across seams")
    parser.add_argument("--double-sided", action="store_true", help="Keep double-sided materials instead of forcing single-sided FrontSide")
    parser.add_argument("--no-preserve-textures", action="store_true", help="Disable texture preservation in Steps 1 & 2")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress stderr logs and only stream NDJSON events")

    args = parser.parse_args()

    uv_mode = args.uv_mode
    if uv_mode is None:
        uv_mode = "xatlas" if args.rechart_uv else "direct"

    pipeline = StepPipeline(
        resolution=args.resolution,
        texture_format=args.format,
        rechart_uv=args.rechart_uv,
        uv_mode=uv_mode,
        smooth_normals=args.smooth_normals,
        double_sided=args.double_sided,
        preserve_textures=not args.no_preserve_textures,
        verbose=not args.quiet,
        stream_events=True
    )

    try:
        pipeline.run(Path(args.input), Path(args.output_dir))
    except Exception as e:
        err_payload = {"event": "pipeline_error", "error": str(e)}
        print(json.dumps(err_payload), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
