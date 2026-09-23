"""
test_ktx2_vram_skip.py

Step 7 skips KTX2 when the Step 6 texture VRAM estimate (metrics totalGpuVramBytes) is below
StepPipeline(ktx2_min_vram_mb=20.0): for small textures KTX2 UASTC grows the file while the VRAM is
already small. The skip must be explicit: extras state the real texture format and the reason,
metrics carry gpuCompressionSkipped. ktx2_min_vram_mb=0 always runs KTX2.
Both runs skip Step 3 (face reduction) so the texture VRAM under test is the one the original
geometry and its 1536x1536 texture produce.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from optimizer.step_pipeline import StepPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DINOKI = REPO_ROOT / "examples" / "models" / "dinoki_raw.glb"
MB = 1024 * 1024


def _read_glb(glb_path: Path):
    """Returns (glTF JSON, BIN chunk bytes) of a GLB file."""
    data = glb_path.read_bytes()
    json_len, json_type = struct.unpack("<I4s", data[12:20])
    assert json_type == b"JSON", f"First chunk of {glb_path.name} is not JSON"
    gltf = json.loads(data[20:20 + json_len].decode("utf-8"))
    bin_start = 20 + json_len
    bin_len, bin_type = struct.unpack("<I4s", data[bin_start:bin_start + 8])
    assert bin_type == b"BIN\x00", f"Second chunk of {glb_path.name} is not BIN"
    return gltf, data[bin_start + 8:bin_start + 8 + bin_len]


def _image_bytes(gltf: dict, bin_chunk: bytes, index: int = 0) -> bytes:
    view = gltf["bufferViews"][gltf["images"][index]["bufferView"]]
    start = view.get("byteOffset", 0)
    return bin_chunk[start:start + view["byteLength"]]


class TestKtx2VramSkip(unittest.TestCase):
    """Dinoki keeps its 1536x1536 JPEG (12 MB texture VRAM): default skips KTX2, threshold 0 runs it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="test_ktx2_vram_skip_")
        tmp = Path(cls._tmp.name)
        cls.skip_dir = tmp / "default"
        StepPipeline(verbose=False, stream_events=False, skip_steps=[3]).run(DINOKI, cls.skip_dir)
        cls.skip = json.loads((cls.skip_dir / "metrics.json").read_text())
        cls.ktx2_dir = tmp / "threshold_0"
        StepPipeline(ktx2_min_vram_mb=0, verbose=False, stream_events=False, skip_steps=[3]).run(DINOKI, cls.ktx2_dir)
        cls.ktx2 = json.loads((cls.ktx2_dir / "metrics.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_default_skips_ktx2_below_threshold(self):
        steps = self.skip["steps"]
        self.assertEqual(len(steps), 8)
        m6 = steps[6]["metrics"]
        m7 = steps[7]["metrics"]
        self.assertLess(m6["totalGpuVramBytes"], 20 * MB)

        step6_gltf, step6_bin = _read_glb(self.skip_dir / "step_06_meshopt.glb")
        step7_file = self.skip_dir / "step_07_final.glb"
        step7_gltf, step7_bin = _read_glb(step7_file)
        self.assertNotIn("KHR_texture_basisu", step7_gltf.get("extensionsUsed", []))
        self.assertEqual([img["mimeType"] for img in step7_gltf["images"]],
                         [img["mimeType"] for img in step6_gltf["images"]])
        self.assertEqual(_image_bytes(step7_gltf, step7_bin), _image_bytes(step6_gltf, step6_bin))

        # Extras state the real texture format and why KTX2 was skipped
        extras = step7_gltf["extras"]
        self.assertEqual(m6["textures"][0]["format"], "JPEG")
        self.assertEqual(extras["texture_format"], m7["textures"][0]["format"])
        self.assertEqual(extras["texture_format"], "JPEG")
        self.assertTrue(extras["gpu_compression"].startswith("skipped"), extras["gpu_compression"])
        self.assertEqual(extras["gpu_compression"], "skipped: texture VRAM 12.00 MB < 20 MB")

        self.assertIs(m7["gpuCompressionSkipped"], True)
        self.assertEqual(m7["gpuCompressionReason"], "texture VRAM 12.00 MB < 20 MB")
        self.assertEqual(m7["textureVramEstimateBytes"], m6["totalGpuVramBytes"])
        self.assertEqual(steps[7]["details"]["gpuCompressionSkipped"], True)

        # Step 7 only adds frontSide material + extras JSON on top of the Step 6 file
        step6_size = (self.skip_dir / "step_06_meshopt.glb").stat().st_size
        self.assertLessEqual(step7_file.stat().st_size, step6_size + 4096)

        for step_info in steps:
            if step_info["skipped"]:
                continue
            self.assertEqual(step_info["metrics"]["faces"], steps[0]["metrics"]["faces"], step_info["file"])

    def test_threshold_zero_runs_ktx2(self):
        steps = self.ktx2["steps"]
        self.assertEqual(len(steps), 8)
        m7 = steps[7]["metrics"]
        step7_gltf, _ = _read_glb(self.ktx2_dir / "step_07_final.glb")
        self.assertIn("KHR_texture_basisu", step7_gltf["extensionsUsed"])
        self.assertEqual([img["mimeType"] for img in step7_gltf["images"]], ["image/ktx2"])
        self.assertEqual(step7_gltf["extras"]["texture_format"], "KTX2")
        self.assertEqual(step7_gltf["extras"]["gpu_compression"], "ktx2 uastc")
        self.assertIs(m7["gpuCompressionSkipped"], False)
        self.assertEqual(m7["textureVramEstimateBytes"], steps[6]["metrics"]["totalGpuVramBytes"])
        for step_info in steps:
            if step_info["skipped"]:
                continue
            self.assertEqual(step_info["metrics"]["faces"], steps[0]["metrics"]["faces"], step_info["file"])

    def test_invalid_threshold_raises(self):
        for bad in (-1, -0.5, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                StepPipeline(ktx2_min_vram_mb=bad)
        for bad in ("20", None, True, [20]):
            with self.assertRaises(TypeError, msg=repr(bad)):
                StepPipeline(ktx2_min_vram_mb=bad)
        self.assertEqual(StepPipeline().ktx2_min_vram_mb, 20.0)
        self.assertEqual(StepPipeline(ktx2_min_vram_mb=0).ktx2_min_vram_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
