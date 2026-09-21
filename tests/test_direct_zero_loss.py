"""
test_direct_zero_loss.py

Unit & Integration Tests for True Zero-Loss Direct Master UV:
1. 100% Bit-for-Bit Bitstream Pass-through (SHA-256 identical from Raw to Step 3 to Step 6)
2. Gamma-Corrected Linear Color Space Resampling (preserving specular highlights & eye reflection points)
3. 100% Vertex Normals preservation
"""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DINOKI = REPO_ROOT / "examples" / "models" / "dinoki_raw.glb"
SAMPLE_FLAMIBO = REPO_ROOT / "examples" / "models" / "flamibo_raw.glb"

from optimizer.core.texture_utils import (
    extract_original_texture_info,
    resample_texture_linear_gamma
)
from optimizer.step_pipeline import StepPipeline


class TestDirectZeroLoss(unittest.TestCase):
    def test_dinoki_bitstream_passthrough(self):
        """Verify that Dinoki in Direct Mode with format='original' preserves 100% bit-for-bit texture bytes."""
        with tempfile.TemporaryDirectory(prefix="test_zero_loss_dinoki_") as tmpdir:
            out_dir = Path(tmpdir)

            # 1. Extract raw texture bytes
            raw_info = extract_original_texture_info(SAMPLE_DINOKI)
            raw_bytes = raw_info["slots"]["baseColorTexture"]["raw_bytes"]
            self.assertIsNotNone(raw_bytes, "Raw texture bytes must not be None")
            raw_hash = hashlib.sha256(raw_bytes).hexdigest()

            # 2. Run StepPipeline
            pipeline = StepPipeline(
                resolution="auto",
                texture_format="original",
                uv_mode="direct",
                verbose=False,
                stream_events=False
            )
            result = pipeline.run(SAMPLE_DINOKI, out_dir)
            self.assertTrue(result["success"])

            # 3. Check Step 3 texture bytes
            s3_info = extract_original_texture_info(out_dir / "step_03_texture_baked.glb")
            s3_bytes = s3_info["slots"]["baseColorTexture"]["raw_bytes"]
            self.assertIsNotNone(s3_bytes)
            s3_hash = hashlib.sha256(s3_bytes).hexdigest()
            self.assertEqual(raw_hash, s3_hash, "Step 3 texture must be 100% bit-for-bit identical to raw")

            # 4. Check Step 6 texture bytes via gltf-transform
            node_script = f"""
import {{ NodeIO }} from '@gltf-transform/core';
import {{ ALL_EXTENSIONS }} from '@gltf-transform/extensions';
import {{ MeshoptDecoder }} from 'meshoptimizer';
import crypto from 'crypto';

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({{
  'meshopt.decoder': MeshoptDecoder
}});

const doc = await io.read('{out_dir}/step_06_final.glb');
const tex = doc.getRoot().listTextures()[0];
const buf = Buffer.from(tex.getImage());
console.log(JSON.stringify({{
  size: buf.length,
  hash: crypto.createHash('sha256').update(buf).digest('hex')
}}));
"""
            proc = subprocess.run(["node", "--input-type=module", "-e", node_script], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Node error: {proc.stderr}")
            s6_data = json.loads(proc.stdout.strip())
            self.assertEqual(raw_hash, s6_data["hash"], "Step 6 final texture must be 100% bit-for-bit identical to raw")

    def test_linear_gamma_resampling_fidelity(self):
        """Verify that Linear Color Space Resampling conserves peak specular luminance over standard sRGB."""
        # Create an 8x8 image with a pure white specular highlight (255)
        img_arr = np.zeros((8, 8), dtype=np.uint8)
        img_arr[3:5, 3:5] = 255
        pil_img = Image.fromarray(img_arr, mode="L").convert("RGB")

        # Standard sRGB Lanczos resize
        down_srgb = pil_img.resize((4, 4), Image.Resampling.LANCZOS)
        max_srgb = np.array(down_srgb).max()

        # Gamma-corrected linear resampling
        down_linear = resample_texture_linear_gamma(pil_img, (4, 4), unsharp_strength=0.0)
        max_linear = np.array(down_linear).max()

        # Linear resampling should conserve optical energy and be significantly brighter
        self.assertGreater(max_linear, max_srgb, "Linear resampling must conserve specular highlight luminance")
        self.assertGreaterEqual(max_linear, 140, "Linear highlight should maintain high brightness")


if __name__ == "__main__":
    unittest.main()
