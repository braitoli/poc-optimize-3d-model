"""
pipeline.py

Master 3D Model Optimization Pipeline.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles (ratio = 1.0).
- Visibility z-buffer shell orienting (FrontSide CCW).
- UV Atlas Re-charting / Direct Lanczos + 16px boundary dilation.
- Angle-weighted smooth vertex normals across UV seams (spatial hashing).
- EXT_meshopt_compression: 14-bit position, 12-bit normal, Float32 UV, GPU cache reorder.
- Basis Universal KTX2 UASTC Level 2 Mipmaps (or WebP).
"""

import os
import sys
import time
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image
import trimesh

from optimizer.core.cleaner import clean_and_repair_mesh, auto_ground_and_center
from optimizer.core.shell_orient import orient_faces_by_visibility, DEFAULT_VIEWS, DEFAULT_RESOLUTION
from optimizer.core.uv_baker import (
    rebake_texture_xatlas,
    direct_resample_texture,
    rechart_and_bake_high_density,
    compute_uv_metrics
)
from optimizer.core.palette import extract_palette, embed_gltf_extras
from optimizer.core.texture_utils import (
    extract_original_texture_info,
    preserve_mesh_textures,
    clamp_target_resolution,
    optimize_mesh_texture_for_export,
    can_downscale_texture,
    maximize_uv_space
)

MODULE_ROOT = Path(__file__).resolve().parent
NODE_SCRIPT = MODULE_ROOT / "node" / "optimize_meshopt.mjs"


def set_frontside_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is false for all materials in a binary GLB."""
    import struct
    if len(glb_bytes) < 20:
        return glb_bytes

    magic, ver, length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF":
        return glb_bytes

    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        return glb_bytes

    json_bytes = glb_bytes[20:20 + chunk_len]
    gltf = json.loads(json_bytes.decode("utf-8"))

    if "materials" in gltf:
        for mat in gltf["materials"]:
            mat["doubleSided"] = False

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + chunk_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    out = bytearray()
    out.extend(struct.pack("<4sII", magic, ver, new_total_len))
    out.extend(struct.pack("<I4s", len(new_json_bytes), b"JSON"))
    out.extend(new_json_bytes)
    out.extend(bin_chunk)
    return bytes(out)


def _decompress_meshopt_if_needed(input_path: Path, tmp_dir: Path) -> Path:
    """If input GLB has EXT_meshopt_compression, decompress it using Node.js for trimesh compatibility."""
    import struct
    try:
        with open(input_path, "rb") as f:
            header = f.read(12)
            if len(header) == 12:
                magic, ver, length = struct.unpack("<4sII", header)
                if magic == b"glTF":
                    chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
                    if chunk_type == b"JSON":
                        gltf = json.loads(f.read(chunk_len))
                        exts = gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", [])
                        if "EXT_meshopt_compression" in exts:
                            uncompressed_path = tmp_dir / f"unpacked_{input_path.name}"
                            node_script = f"""
import {{ NodeIO }} from '@gltf-transform/core';
import {{ ALL_EXTENSIONS }} from '@gltf-transform/extensions';
import {{ MeshoptDecoder }} from 'meshoptimizer';
import fs from 'fs';

async function decompress() {{
    await MeshoptDecoder.ready;
    const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({{ 'meshopt.decoder': MeshoptDecoder }});
    const doc = await io.read({json.dumps(str(input_path.resolve()))});
    const ext = doc.getRoot().listExtensionsUsed().find(e => e.extensionName === 'EXT_meshopt_compression');
    if (ext) ext.dispose();
    const glb = await io.writeBinary(doc);
    fs.writeFileSync({json.dumps(str(uncompressed_path.resolve()))}, glb);
}}
decompress();
"""
                            subprocess.run(
                                ["node", "-e", node_script],
                                check=True,
                                cwd=str(MODULE_ROOT.parent),
                                capture_output=True
                            )
                            if uncompressed_path.exists():
                                return uncompressed_path
    except Exception:
        pass
    return input_path


def set_doublesided_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is true for all materials in a binary GLB."""
    import struct
    if len(glb_bytes) < 20:
        return glb_bytes
    magic, version, _ = struct.unpack("<III", glb_bytes[:12])
    if magic != 0x46546C67:
        return glb_bytes
    json_len, json_type = struct.unpack("<II", glb_bytes[12:20])
    if json_type != 0x4E4F534A:
        return glb_bytes

    gltf = json.loads(glb_bytes[20:20 + json_len].decode("utf-8"))
    modified = False
    for mat in gltf.get("materials", []):
        if mat.get("doubleSided") is not True:
            mat["doubleSided"] = True
            modified = True

    if not modified:
        return glb_bytes

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + json_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    header = struct.pack("<III", magic, version, new_total_len)
    chunk0 = struct.pack("<II", len(new_json_bytes), json_type)
    return header + chunk0 + new_json_bytes + bin_chunk


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


class ModelOptimizer:
    def __init__(
        self,
        resolution: Union[int, str] = "auto",
        texture_format: str = "ktx2",
        rechart_uv: bool = False,
        smooth_normals: bool = True,
        double_sided: bool = False,
        verbose: bool = True,
        export_steps_dir: Optional[Path] = None,
        step_callback: Optional[Any] = None
    ):
        self.resolution = resolution
        self.texture_format = texture_format.lower()
        self.rechart_uv = rechart_uv
        self.smooth_normals = smooth_normals
        self.double_sided = double_sided
        self.verbose = verbose
        self.export_steps_dir = Path(export_steps_dir).resolve() if export_steps_dir else None
        self.step_callback = step_callback

    def log(self, msg: str):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)

    def optimize(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        t0 = time.time()
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        raw_size = input_path.stat().st_size
        self.log("=" * 65)
        self.log(f"🚀 STARTING 3D MODEL OPTIMIZATION: {input_path.name}")
        self.log(f"   Input size: {raw_size / 1024 / 1024:.2f} MB ({raw_size:,} bytes)")
        res_display = f"{self.resolution}x{self.resolution}" if isinstance(self.resolution, int) else f"{self.resolution.upper()} (Adaptive)"
        self.log(f"   Resolution: {res_display}")
        self.log(f"   Texture format: {self.texture_format.upper()}")
        self.log(f"   UV Re-charting: {'ENABLED (xatlas)' if self.rechart_uv else 'DIRECT MASTER UV'}")
        self.log(f"   Rule 11 Compliance: STRICT ZERO-DECIMATION (100% faces preserved)")
        self.log("=" * 65)

        tmp_dir = Path(tempfile.mkdtemp(prefix="poc_opt_3d_"))
        steps_list = []

        def format_file_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.2f} MB"

        def emit_step(step_data: Dict[str, Any], t_step_start: Optional[float] = None, explicit_duration: Optional[float] = None):
            dur_sec = explicit_duration if explicit_duration is not None else (
                (time.perf_counter() - t_step_start) if t_step_start is not None else 0.0
            )
            total_sec = time.perf_counter() - t_total_start
            step_data["durationSeconds"] = round(dur_sec, 3)
            step_data["durationFormatted"] = format_duration(dur_sec)
            step_data["totalDurationSeconds"] = round(total_sec, 3)
            step_data["durationMs"] = round(dur_sec * 1000, 1)
            steps_list.append(step_data)

            if self.export_steps_dir:
                try:
                    payload = {
                        "success": True,
                        "model": input_path.name,
                        "outputDir": str(self.export_steps_dir),
                        "steps": steps_list,
                        "lastCompletedStep": step_data.get("step", 0)
                    }
                    (self.export_steps_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
                except Exception:
                    pass

            if self.step_callback:
                try:
                    self.step_callback(step_data)
                except Exception as cb_err:
                    self.log(f"   [StepCallback Error] {cb_err}")

        try:
            # 1. Load Raw Mesh
            t_s0 = time.perf_counter()
            self.log("▶️ [Phase 1/5] Loading & Geometric Cleaning...")
            load_path = _decompress_meshopt_if_needed(input_path, tmp_dir)
            raw_mesh = trimesh.load(str(load_path), force="mesh", process=False)
            initial_faces = len(raw_mesh.faces)
            initial_verts = len(raw_mesh.vertices)
            self.log(f"   Raw mesh: {initial_faces:,} faces, {initial_verts:,} vertices")

            if self.export_steps_dir:
                self.export_steps_dir.mkdir(parents=True, exist_ok=True)
                s0_path = self.export_steps_dir / "step0_raw.glb"
                shutil.copyfile(str(input_path), str(s0_path))
                s0_bbox = [round(float(x), 3) for x in (raw_mesh.bounds[1] - raw_mesh.bounds[0]).tolist()] if hasattr(raw_mesh, 'bounds') and raw_mesh.bounds is not None else [1.0, 1.0, 1.0]
                emit_step({
                    "step": 0,
                    "name": "Raw Input",
                    "status": "completed",
                    "description": "Original raw unoptimized 3D asset",
                    "fileSize": raw_size,
                    "fileSizeFormatted": format_file_size(raw_size),
                    "faces": initial_faces,
                    "vertices": initial_verts,
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s0_bbox,
                    "textureFormat": "PNG/JPEG",
                    "textureRes": "Native",
                    "gpuVramMb": round((raw_size * 2.5) / (1024 * 1024), 2),
                    "modelFile": "step0_raw.glb"
                }, t_step_start=t_s0)

            t_s1 = time.perf_counter()
            cleaned_mesh = clean_and_repair_mesh(raw_mesh)
            grounded_mesh, translation = auto_ground_and_center(cleaned_mesh)
            self.log(f"   Grounded base at Y=0, translation applied: {np.round(translation, 4)}")

            orig_tex_info = extract_original_texture_info(input_path)
            preserve_mesh_textures(grounded_mesh, orig_tex_info)

            if self.export_steps_dir:
                s1_bytes = trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": grounded_mesh}), include_normals=True)
                s1_path = self.export_steps_dir / "step1_clean_ground.glb"
                s1_path.write_bytes(s1_bytes)
                s1_bbox = [round(float(x), 3) for x in (grounded_mesh.bounds[1] - grounded_mesh.bounds[0]).tolist()] if hasattr(grounded_mesh, 'bounds') and grounded_mesh.bounds is not None else s0_bbox
                emit_step({
                    "step": 1,
                    "name": "Clean & Auto-Ground",
                    "status": "completed",
                    "description": "Base grounded at Y=0, degenerate geometry repaired, 100% faces preserved",
                    "fileSize": len(s1_bytes),
                    "fileSizeFormatted": format_file_size(len(s1_bytes)),
                    "faces": len(grounded_mesh.faces),
                    "vertices": len(grounded_mesh.vertices),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s1_bbox,
                    "textureFormat": orig_tex_info.get("default_format", "PNG/JPEG"),
                    "textureRes": "Native",
                    "gpuVramMb": round((len(s1_bytes) * 2.2) / (1024 * 1024), 2),
                    "modelFile": "step1_clean_ground.glb"
                }, t_step_start=t_s1)

            # 2. Shell Orient (Visibility Z-Buffer Raycast)
            t_s2 = time.perf_counter()
            self.log("▶️ [Phase 2/5] Shell Orienting (Visibility Z-Buffer CCW Winding)...")
            orient_stats = {}
            oriented_faces = orient_faces_by_visibility(
                grounded_mesh.vertices,
                grounded_mesh.faces,
                views=DEFAULT_VIEWS,
                resolution=DEFAULT_RESOLUTION,
                stats=orient_stats
            )
            grounded_mesh.faces = oriented_faces
            preserve_mesh_textures(grounded_mesh, orig_tex_info)
            self.log(f"   Flipped {orient_stats.get('faces_flipped', 0):,} faces to outward CCW FrontSide")

            if self.export_steps_dir:
                s2_bytes = trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": grounded_mesh}), include_normals=True)
                s2_path = self.export_steps_dir / "step2_shell_orient.glb"
                s2_path.write_bytes(s2_bytes)
                emit_step({
                    "step": 2,
                    "name": "Shell Orienting",
                    "status": "completed",
                    "description": f"Z-Buffer visibility raycast, flipped {orient_stats.get('faces_flipped', 0)} faces to CCW FrontSide",
                    "fileSize": len(s2_bytes),
                    "fileSizeFormatted": format_file_size(len(s2_bytes)),
                    "faces": len(grounded_mesh.faces),
                    "vertices": len(grounded_mesh.vertices),
                    "facesFlipped": orient_stats.get('faces_flipped', 0),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s1_bbox,
                    "textureFormat": orig_tex_info.get("default_format", "PNG/JPEG"),
                    "textureRes": "Native",
                    "gpuVramMb": round((len(s2_bytes) * 2.2) / (1024 * 1024), 2),
                    "modelFile": "step2_shell_orient.glb"
                }, t_step_start=t_s2)

            # 3. Extract Texture & UV Processing
            raw_uv = getattr(raw_mesh.visual, "uv", None)
            if raw_uv is None:
                raw_uv = getattr(grounded_mesh.visual, "uv", np.zeros((len(grounded_mesh.vertices), 2)))

            raw_tex_img = orig_tex_info.get("base_image")
            if raw_tex_img is None and hasattr(grounded_mesh.visual, "material") and hasattr(grounded_mesh.visual.material, "baseColorTexture"):
                raw_tex_img = grounded_mesh.visual.material.baseColorTexture
            if raw_tex_img is None:
                default_dim = 1024 if (self.resolution == "auto" or not isinstance(self.resolution, int)) else self.resolution
                raw_tex_img = Image.new("RGB", (default_dim, default_dim), (200, 200, 200))

            # Auto-Resolution determination or Manual Clamping
            is_auto = self.resolution is None or (isinstance(self.resolution, str) and self.resolution.lower() == "auto")
            downscale_info = {}
            uv_adjust_info = {}

            if is_auto:
                self.log("▶️ [Phase 3/5] Auto-evaluating adaptive resolution (Texel Density & Square UV Packing)...")
                # 1. Evaluate whether downscaling is possible via can_downscale_texture
                can_downscale, optimal_res, downscale_info = can_downscale_texture(
                    grounded_mesh,
                    raw_tex_img,
                    uv=raw_uv,
                    target_res="auto",
                    logger_fn=self.log
                )
                target_res = optimal_res

                # 2. In both branches, apply maximize_uv_space to maximize UV canvas utilization
                grounded_mesh, raw_tex_img, raw_uv, uv_adjust_info = maximize_uv_space(
                    grounded_mesh,
                    raw_tex_img,
                    uv=raw_uv,
                    target_res=target_res,
                    logger_fn=self.log
                )
                self.resolution = target_res
            else:
                # Enforce NO-UPSCALE policy: clamp requested resolution
                target_res = clamp_target_resolution(self.resolution, raw_tex_img.size, logger_fn=self.log)
                self.resolution = target_res

            t_s3 = time.perf_counter()
            self.log(f"▶️ [Phase 3/5] Texture Processing ({target_res}x{target_res} + 16px Dilation)...")

            uv_stats = {}
            if self.rechart_uv:
                self.log(f"   Re-charting UV islands with xatlas (High-Density Packing, {target_res}x{target_res})...")
                baked_mesh, dilated_pil = rechart_and_bake_high_density(
                    grounded_mesh,
                    target_res=target_res,
                    source_image=raw_tex_img,
                    source_uv=raw_uv,
                    dilation_padding=16,
                    double_sided=False,
                    stats=uv_stats
                )
                self.log(f"   ✓ High-Density UV: {uv_stats.get('uv_coverage_ratio_percent', 0)}% coverage | Texel Density: {uv_stats.get('texel_density_linear', 0)} px/unit")
            else:
                self.log(f"   Direct Master UV mode: Resampling Lanczos ({target_res}x{target_res}) + 16px dilation...")
                baked_mesh, dilated_pil = direct_resample_texture(
                    grounded_mesh,
                    source_image=raw_tex_img,
                    target_res=target_res,
                    dilation_padding=16
                )

            # Ensure doubleSided=True so browser viewer does not backface-cull triangles
            if hasattr(baked_mesh, "visual") and hasattr(baked_mesh.visual, "material") and baked_mesh.visual.material is not None:
                baked_mesh.visual.material.doubleSided = True

            # Optimize texture before export (defense-in-depth: JPEG if opaque, optimized PNG if alpha)
            opt_pil = optimize_mesh_texture_for_export(baked_mesh, orig_tex_info=orig_tex_info)
            if opt_pil is not None:
                dilated_pil = opt_pil

            tex_fmt = getattr(dilated_pil, "format", "JPEG")
            s3_bytes = None
            if self.export_steps_dir:
                s3_bytes = trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": baked_mesh}), include_normals=True)
                s3_bytes = set_doublesided_material(s3_bytes)
                s3_path = self.export_steps_dir / "step3_uv_bake.glb"
                s3_path.write_bytes(s3_bytes)
                s3_bbox = [round(float(x), 3) for x in (baked_mesh.bounds[1] - baked_mesh.bounds[0]).tolist()] if hasattr(baked_mesh, 'bounds') and baked_mesh.bounds is not None else s1_bbox
                step3_data = {
                    "step": 3,
                    "name": "UV & Texture Bake",
                    "status": "completed",
                    "description": f"Master UV texture resampled to {target_res}x{target_res} with 16px boundary dilation",
                    "fileSize": len(s3_bytes),
                    "fileSizeFormatted": format_file_size(len(s3_bytes)),
                    "faces": len(baked_mesh.faces),
                    "vertices": len(baked_mesh.vertices),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s3_bbox,
                    "textureFormat": f"{tex_fmt} (Dilated 16px)",
                    "textureRes": f"{target_res}x{target_res}",
                    "gpuVramMb": round((target_res * target_res * 4 * 1.33) / (1024 * 1024), 2),
                    "modelFile": "step3_uv_bake.glb"
                }
                if is_auto:
                    step3_data["autoResolution"] = {
                        "enabled": True,
                        "canDownscale": downscale_info.get("can_downscale", False),
                        "optimalResolution": target_res,
                        "uvSpaceMaximized": uv_adjust_info.get("adjusted", False)
                    }
                emit_step(step3_data, t_step_start=t_s3)

            # 4. Extract Palette
            t_s4 = time.perf_counter()
            self.log("▶️ [Phase 4/5] Extracting 10-color Dominant Palette...")
            img_rgb = np.asarray(dilated_pil)
            v_uv = baked_mesh.visual.uv % 1.0
            px = np.clip((v_uv[:, 0] * (img_rgb.shape[1] - 1)).astype(int), 0, img_rgb.shape[1] - 1)
            py = np.clip(((1.0 - v_uv[:, 1]) * (img_rgb.shape[0] - 1)).astype(int), 0, img_rgb.shape[0] - 1)
            surface_pixels = img_rgb[py, px]

            palette_data = extract_palette(image=dilated_pil, sample_pixels=surface_pixels, n_colors=10)
            self.log(f"   Primary color: {palette_data['primaryColor']}")
            self.log(f"   Palette: {palette_data['palette']}")

            # Save intermediate GLB
            intermediate_glb = tmp_dir / "intermediate_baked.glb"
            scene = trimesh.Scene({"Model": baked_mesh})
            if s3_bytes is None:
                s3_bytes = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
            intermediate_glb.write_bytes(s3_bytes)

            if self.export_steps_dir:
                s4_bytes = embed_gltf_extras(s3_bytes, {"palette": palette_data["palette"], "primaryColor": palette_data["primaryColor"]})
                s4_path = self.export_steps_dir / "step4_palette.glb"
                s4_path.write_bytes(s4_bytes)
                emit_step({
                    "step": 4,
                    "name": "Palette Extraction",
                    "status": "completed",
                    "description": f"10 dominant surface colors extracted via KMeans, primary: {palette_data['primaryColor']}",
                    "fileSize": len(s4_bytes),
                    "fileSizeFormatted": format_file_size(len(s4_bytes)),
                    "faces": len(baked_mesh.faces),
                    "vertices": len(baked_mesh.vertices),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s3_bbox,
                    "textureFormat": "PNG (Dilated 16px)",
                    "textureRes": f"{self.resolution}x{self.resolution}",
                    "palette": palette_data["palette"],
                    "paletteDetails": palette_data.get("paletteDetails", []),
                    "primaryColor": palette_data["primaryColor"],
                    "gpuVramMb": round((self.resolution * self.resolution * 4 * 1.33) / (1024 * 1024), 2),
                    "modelFile": "step4_palette.glb"
                }, t_step_start=t_s4)

            # 5. Invoke Node.js Transform3D Optimizer
            t_s5 = time.perf_counter()
            self.log("▶️ [Phase 5/5] Node.js Transform3D: Smooth Normals + Meshopt + KTX2 UASTC...")
            intermediate_opt_glb = tmp_dir / "intermediate_opt.glb"

            node_cmd = [
                "node", str(NODE_SCRIPT),
                str(intermediate_glb),
                str(intermediate_opt_glb),
                "--pos-bits", "14",
                "--normal-bits", "12",
                "--weld", "0.0001",
                "--reorder",
                "--meshopt",
                "--texture-max-dim", str(self.resolution),
                "--json"
            ]

            s5_path = None
            if self.export_steps_dir:
                s5_path = self.export_steps_dir / "step5_meshopt.glb"
                node_cmd.extend(["--export-intermediate-meshopt", str(s5_path)])

            if self.smooth_normals:
                node_cmd.append("--smooth-normals")
            else:
                node_cmd.append("--no-smooth-normals")

            if self.texture_format == "webp":
                node_cmd.extend(["--webp", "--webp-quality", "85"])
            else:
                node_cmd.extend(["--ktx2", "--ktx2-mode", "uastc", "--ktx2-level", "2", "--ktx2-rdo", "1.0"])

            if not self.double_sided:
                node_cmd.append("--single-sided")
            else:
                node_cmd.append("--keep-double-sided")

            proc = subprocess.run(node_cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not intermediate_opt_glb.exists():
                raise RuntimeError(f"Transform3D optimization failed: {proc.stderr or proc.stdout}")

            # Parse JSON output from node
            node_summary = {}
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        node_summary = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass

            if self.export_steps_dir and s5_path and s5_path.exists():
                s5_size = s5_path.stat().st_size
                emit_step({
                    "step": 5,
                    "name": "Meshopt Compression",
                    "status": "completed",
                    "description": "EXT_meshopt_compression: 14-bit position, 12-bit normal, vertex cache reorder",
                    "fileSize": s5_size,
                    "fileSizeFormatted": format_file_size(s5_size),
                    "faces": node_summary.get("trianglesAfter", len(baked_mesh.faces)),
                    "vertices": node_summary.get("verticesAfter", len(baked_mesh.vertices)),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s3_bbox,
                    "textureFormat": "PNG (Dilated 16px)",
                    "textureRes": f"{self.resolution}x{self.resolution}",
                    "palette": palette_data["palette"],
                    "paletteDetails": palette_data.get("paletteDetails", []),
                    "primaryColor": palette_data["primaryColor"],
                    "gpuVramMb": round((self.resolution * self.resolution * 4 * 1.33) / (1024 * 1024), 2),
                    "modelFile": "step5_meshopt.glb"
                }, t_step_start=t_s5)

            # 6. Embed glTF extras & configure material (Final Step)
            t_s6 = time.perf_counter()
            final_bytes = intermediate_opt_glb.read_bytes()
            if not self.double_sided:
                final_bytes = set_frontside_material(final_bytes)

            extras_payload = {
                "palette": palette_data["palette"],
                "primaryColor": palette_data["primaryColor"],
                "paletteDetails": palette_data["paletteDetails"],
                "resolution": f"{self.resolution}x{self.resolution}",
                "texture_format": self.texture_format.upper(),
                "policy": "STRICT 0-DECIMATION (--ratio 1.0)",
                "tool": "poc-optimize-3d-model v1.0.0"
            }
            final_bytes = embed_gltf_extras(final_bytes, extras_payload)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(final_bytes)

            final_size = len(final_bytes)
            total_elapsed = time.perf_counter() - t_total_start
            elapsed = time.time() - t0

            if self.export_steps_dir:
                s6_path = self.export_steps_dir / "step6_final.glb"
                s6_path.write_bytes(final_bytes)
                emit_step({
                    "step": 6,
                    "name": "KTX2 GPU Compression",
                    "status": "completed",
                    "description": f"Basis Universal {self.texture_format.upper()} Level 2 Mipmaps, direct GPU transcode, 100% faces preserved",
                    "fileSize": final_size,
                    "fileSizeFormatted": format_file_size(final_size),
                    "faces": node_summary.get("trianglesAfter", len(baked_mesh.faces)),
                    "vertices": node_summary.get("verticesAfter", len(baked_mesh.vertices)),
                    "drawCalls": 1,
                    "meshes": 1,
                    "primitives": 1,
                    "bbox": s3_bbox,
                    "textureFormat": self.texture_format.upper(),
                    "textureRes": f"{self.resolution}x{self.resolution}",
                    "palette": palette_data["palette"],
                    "paletteDetails": palette_data.get("paletteDetails", []),
                    "primaryColor": palette_data["primaryColor"],
                    "gpuVramMb": round((self.resolution * self.resolution * 1.0 * 1.33) / (1024 * 1024), 2),
                    "modelFile": "step6_final.glb"
                }, t_step_start=t_s6)

            summary = {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "input_size_mb": round(raw_size / 1024 / 1024, 2),
                "output_size_mb": round(final_size / 1024 / 1024, 2),
                "saved_percent": round((1 - final_size / raw_size) * 100, 2),
                "initial_faces": initial_faces,
                "final_faces": node_summary.get("trianglesAfter", len(baked_mesh.faces)),
                "initial_vertices": initial_verts,
                "final_vertices": node_summary.get("verticesAfter", len(baked_mesh.vertices)),
                "faces_preserved_percent": round((node_summary.get("trianglesAfter", len(baked_mesh.faces)) / initial_faces) * 100, 2),
                "palette": palette_data["palette"],
                "primary_color": palette_data["primaryColor"],
                "elapsed_seconds": round(total_elapsed, 3),
                "elapsedSeconds": round(total_elapsed, 3),
                "elapsedFormatted": format_duration(total_elapsed),
                "totalDurationSeconds": round(total_elapsed, 3),
                "stepDurations": {
                    s.get("name", f"step_{s.get('step')}"): {
                        "step": s["step"],
                        "durationSeconds": s.get("durationSeconds", 0.0),
                        "durationFormatted": s.get("durationFormatted", "0s"),
                        "totalDurationSeconds": s.get("totalDurationSeconds", 0.0),
                        "seconds": s.get("durationSeconds", 0.0),
                        "ms": s.get("durationMs", 0.0)
                    }
                    for s in steps_list
                },
                "steps": steps_list
            }

            if self.export_steps_dir:
                try:
                    payload = {
                        "success": True,
                        "model": input_path.name,
                        "outputDir": str(self.export_steps_dir),
                        "summary": summary,
                        "steps": steps_list
                    }
                    (self.export_steps_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
                except Exception:
                    pass

            self.log("=" * 65)
            self.log(f"🎉 OPTIMIZATION COMPLETED IN {total_elapsed:.2f}s!")
            self.log(f"   Final Size: {summary['output_size_mb']} MB (Saved {summary['saved_percent']}%)")
            self.log(f"   Triangles: {summary['final_faces']:,} / {initial_faces:,} ({summary['faces_preserved_percent']}% preserved)")
            self.log(f"   Output saved to: {output_path}")
            self.log("=" * 65)
            return summary

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
