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
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import trimesh

from optimizer.core.cleaner import clean_and_repair_mesh, auto_ground_and_center
from optimizer.core.shell_orient import orient_faces_by_visibility, DEFAULT_VIEWS, DEFAULT_RESOLUTION
from optimizer.core.uv_baker import rebake_texture_xatlas, direct_resample_texture
from optimizer.core.palette import extract_palette, embed_gltf_extras

MODULE_ROOT = Path(__file__).resolve().parent
NODE_SCRIPT = MODULE_ROOT / "node" / "optimize_meshopt.mjs"


def set_frontside_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is false for all materials in a binary GLB."""
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
        if mat.get("doubleSided") is not False:
            mat["doubleSided"] = False
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


class ModelOptimizer:
    def __init__(
        self,
        resolution: int = 1024,
        texture_format: str = "ktx2",
        rechart_uv: bool = False,
        smooth_normals: bool = True,
        double_sided: bool = False,
        verbose: bool = True
    ):
        self.resolution = resolution
        self.texture_format = texture_format.lower()
        self.rechart_uv = rechart_uv
        self.smooth_normals = smooth_normals
        self.double_sided = double_sided
        self.verbose = verbose

    def log(self, msg: str):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)

    def optimize(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        t0 = time.time()
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        raw_size = input_path.stat().st_size
        self.log("=" * 65)
        self.log(f"🚀 STARTING 3D MODEL OPTIMIZATION: {input_path.name}")
        self.log(f"   Input size: {raw_size / 1024 / 1024:.2f} MB ({raw_size:,} bytes)")
        self.log(f"   Resolution: {self.resolution}x{self.resolution}")
        self.log(f"   Texture format: {self.texture_format.upper()}")
        self.log(f"   UV Re-charting: {'ENABLED (xatlas)' if self.rechart_uv else 'DIRECT MASTER UV'}")
        self.log(f"   Rule 11 Compliance: STRICT ZERO-DECIMATION (100% faces preserved)")
        self.log("=" * 65)

        tmp_dir = Path(tempfile.mkdtemp(prefix="poc_opt_3d_"))

        try:
            # 1. Load Raw Mesh
            self.log("▶️ [Phase 1/5] Loading & Geometric Cleaning...")
            raw_mesh = trimesh.load(str(input_path), force="mesh", process=False)
            initial_faces = len(raw_mesh.faces)
            initial_verts = len(raw_mesh.vertices)
            self.log(f"   Raw mesh: {initial_faces:,} faces, {initial_verts:,} vertices")

            cleaned_mesh = clean_and_repair_mesh(raw_mesh)
            grounded_mesh, translation = auto_ground_and_center(cleaned_mesh)
            self.log(f"   Grounded base at Y=0, translation applied: {np.round(translation, 4)}")

            # 2. Shell Orient (Visibility Z-Buffer Raycast)
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
            self.log(f"   Flipped {orient_stats.get('faces_flipped', 0):,} faces to outward CCW FrontSide")

            # 3. Extract Texture & UV Processing
            self.log(f"▶️ [Phase 3/5] Texture Processing ({self.resolution}x{self.resolution} + 16px Dilation)...")
            raw_uv = getattr(raw_mesh.visual, "uv", None)
            if raw_uv is None:
                raw_uv = getattr(grounded_mesh.visual, "uv", np.zeros((len(grounded_mesh.vertices), 2)))

            raw_tex_img = None
            if hasattr(grounded_mesh.visual, "material") and hasattr(grounded_mesh.visual.material, "baseColorTexture"):
                raw_tex_img = grounded_mesh.visual.material.baseColorTexture
            if raw_tex_img is None:
                raw_tex_img = Image.new("RGB", (self.resolution, self.resolution), (200, 200, 200))

            if self.rechart_uv:
                self.log("   Re-charting UV islands with xatlas...")
                baked_mesh, dilated_pil = rebake_texture_xatlas(
                    grounded_mesh,
                    source_image=raw_tex_img,
                    source_uv=raw_uv,
                    target_res=self.resolution,
                    dilation_padding=16
                )
            else:
                self.log("   Direct Master UV mode: Resampling Lanczos + 16px dilation...")
                baked_mesh, dilated_pil = direct_resample_texture(
                    grounded_mesh,
                    source_image=raw_tex_img,
                    target_res=self.resolution,
                    dilation_padding=16
                )

            # 4. Extract Palette
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
            intermediate_glb.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))

            # 5. Invoke Node.js Transform3D Optimizer
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

            # Embed glTF extras & configure material
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
            elapsed = time.time() - t0

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
                "elapsed_seconds": round(elapsed, 2)
            }

            self.log("=" * 65)
            self.log(f"🎉 OPTIMIZATION COMPLETED IN {elapsed:.2f}s!")
            self.log(f"   Final Size: {summary['output_size_mb']} MB (Saved {summary['saved_percent']}%)")
            self.log(f"   Triangles: {summary['final_faces']:,} / {initial_faces:,} ({summary['faces_preserved_percent']}% preserved)")
            self.log(f"   Output saved to: {output_path}")
            self.log("=" * 65)
            return summary

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
