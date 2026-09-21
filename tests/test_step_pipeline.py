"""
test_step_pipeline.py

Unit and integration tests for:
- optimizer/inspect_metrics.mjs (Comprehensive 3D Metrics Engine)
- optimizer/step_pipeline.py (Step-by-Step 7-stage Pipeline & Metrics)
"""

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = REPO_ROOT / "optimizer" / "inspect_metrics.mjs"
SAMPLE_DINOKI = REPO_ROOT / "examples" / "sample_dinoki.glb"

from optimizer.step_pipeline import StepPipeline, inspect_glb_metrics
from optimizer.core.glb_utils import set_doublesided_material
from tests.fixtures import create_mock_glb


class TestMetricsEngine(unittest.TestCase):
    def test_inspect_metrics_cli_json(self):
        """Verify inspect_metrics.mjs CLI outputs valid JSON with all required fields."""
        cmd = ["node", str(INSPECT_SCRIPT), str(SAMPLE_DINOKI), "--compact"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"inspect_metrics error: {proc.stderr}")

        data = json.loads(proc.stdout)

        # Core required metrics
        self.assertIn("fileSizeBytes", data)
        self.assertIn("fileSizeFormatted", data)
        self.assertEqual(data["faces"], 45000)
        self.assertEqual(data["vertices"], 36648)
        self.assertGreaterEqual(data["meshes"], 1)
        self.assertGreaterEqual(data["primitives"], 1)
        self.assertGreaterEqual(data["drawCalls"], 1)

        # UV Channels
        self.assertIn("uvChannels", data)
        self.assertTrue(len(data["uvChannels"]) > 0)
        uv0 = data["uvChannels"][0]
        self.assertIn("channel", uv0)
        self.assertIn("bounds", uv0)
        self.assertIn("min", uv0)
        self.assertIn("max", uv0)

        # Textures & GPU VRAM
        self.assertIn("textures", data)
        self.assertTrue(len(data["textures"]) > 0)
        tex = data["textures"][0]
        self.assertEqual(tex["format"], "JPEG")
        self.assertEqual(tex["resolutionFormatted"], "1536x1536")
        self.assertIn("gpuVramUncompressedBytes", tex)
        self.assertIn("gpuVramKtx2Bytes", tex)
        self.assertIn("estimatedGpuVramBytes", tex)
        self.assertIn("totalGpuVramBytes", data)
        self.assertIn("totalGpuVramFormatted", data)

        # Materials
        self.assertIn("materials", data)
        self.assertTrue(len(data["materials"]) > 0)
        mat = data["materials"][0]
        self.assertIn("alphaMode", mat)
        self.assertIn("doubleSided", mat)

        # Bounding Box
        self.assertIn("boundingBox", data)
        bbox = data["boundingBox"]
        self.assertIn("min", bbox)
        self.assertIn("max", bbox)
        self.assertIn("dimensions", bbox)
        self.assertIn("center", bbox)
        self.assertEqual(len(bbox["dimensions"]), 3)

        # Extensions & Extras
        self.assertIn("extensions", data)
        self.assertIn("palette", data)
        self.assertIn("primaryColor", data)


class TestStepPipeline(unittest.TestCase):
    def test_full_step_pipeline_execution(self):
        """Verify all 7 GLB files are generated with 100% Zero-Decimation preservation."""
        with tempfile.TemporaryDirectory(prefix="test_step_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            pipeline = StepPipeline(
                resolution=512,
                texture_format="ktx2",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False
            )

            result = pipeline.run(SAMPLE_DINOKI, out_dir)
            self.assertTrue(result["success"])

            # Verify metrics.json
            metrics_file = out_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            metrics_data = json.loads(metrics_file.read_text())

            self.assertEqual(len(metrics_data["steps"]), 7)
            expected_files = [
                "step_00_raw.glb",
                "step_01_cleaned_grounded.glb",
                "step_02_oriented.glb",
                "step_03_texture_baked.glb",
                "step_04_palette_tagged.glb",
                "step_05_meshopt.glb",
                "step_06_final.glb"
            ]

            for idx, expected_file in enumerate(expected_files):
                fpath = out_dir / expected_file
                self.assertTrue(fpath.exists(), f"Missing step file: {expected_file}")
                self.assertGreater(fpath.stat().st_size, 100000, f"File too small: {expected_file}")

                step_info = metrics_data["steps"][idx]
                self.assertEqual(step_info["file"], expected_file)
                # Rule 11 Verification: 100% triangles preserved across all steps
                self.assertEqual(step_info["metrics"]["faces"], 45000)

            # Rule 11 & Texture Preservation Verification for geometric steps (Step 1 & 2)
            step1_tex = metrics_data["steps"][1]["metrics"]["textures"][0]
            step2_tex = metrics_data["steps"][2]["metrics"]["textures"][0]
            self.assertEqual(step1_tex["format"], "JPEG", "Step 1 texture must preserve original JPEG format")
            self.assertEqual(step2_tex["format"], "JPEG", "Step 2 texture must preserve original JPEG format")
            # Step 1 and 2 file size should stay close to raw (1.93 MB) instead of blowing up to PNG (> 5 MB)
            self.assertLess(metrics_data["steps"][1]["metrics"]["fileSizeBytes"], 3 * 1024 * 1024)
            self.assertLess(metrics_data["steps"][2]["metrics"]["fileSizeBytes"], 3 * 1024 * 1024)

            # Check final step has KTX2 and meshopt extensions
            final_step = metrics_data["steps"][6]["metrics"]
            self.assertIn("EXT_meshopt_compression", final_step["extensions"]["used"])
            self.assertIn("KHR_texture_basisu", final_step["extensions"]["used"])
            self.assertGreater(len(final_step["palette"]), 0)

            # Step 3 must have FrontSide rendering (doubleSided=False) when double_sided=False
            step3_metrics = metrics_data["steps"][3]["metrics"]
            self.assertFalse(step3_metrics["materials"][0]["doubleSided"], "Step 3 material must be doubleSided=False for FrontSide")

    def test_auto_resolution_step_pipeline(self):
        """Verify pipeline execution with resolution='auto' calculates optimal resolution and executes all 7 steps."""
        with tempfile.TemporaryDirectory(prefix="test_auto_res_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            pipeline = StepPipeline(
                resolution="auto",
                texture_format="webp",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False
            )

            result = pipeline.run(SAMPLE_DINOKI, out_dir)
            self.assertTrue(result["success"])

            # Verify metrics.json
            metrics_file = out_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            metrics_data = json.loads(metrics_file.read_text())

            self.assertEqual(len(metrics_data["steps"]), 7)

            # Step 3 must re-chart with xatlas on the 1:1 capacity canvas (1536 source -> 1024 max POT, NO-UPSCALE)
            step3 = metrics_data["steps"][3]
            self.assertEqual(step3["metrics"]["uvMode"], "xatlas")
            self.assertIn("1024x1024", step3["metrics"]["textureResolution"])
            self.assertEqual(step3["metrics"]["faces"], 45000)
            self.assertGreater(step3["metrics"]["uvCoverageRatio"], 0.0)
            self.assertFalse(step3["metrics"]["materials"][0]["doubleSided"])

            # Final step must preserve 100% faces
            final_step = metrics_data["steps"][6]
            self.assertEqual(final_step["metrics"]["faces"], 45000)

    def test_rechart_uv_step_pipeline_adaptive_metrics(self):
        """Verify Step 3 adaptive recharting, downscaling evaluation, and FrontSide material."""
        with tempfile.TemporaryDirectory(prefix="test_rechart_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            pipeline = StepPipeline(
                resolution=1024,
                texture_format="webp",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False
            )

            result = pipeline.run(SAMPLE_DINOKI, out_dir)
            self.assertTrue(result["success"])

            metrics_file = out_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            metrics_data = json.loads(metrics_file.read_text())

            step3 = metrics_data["steps"][3]
            m3 = step3["metrics"]
            self.assertIn("downscaled", m3)
            self.assertIn("originalResolution", m3)
            self.assertIn("finalResolution", m3)
            self.assertIn("uvCoverageRatio", m3)
            self.assertIn("texelDensityDelta", m3)
            self.assertGreater(m3["uvCoverageRatio"], 0.0)

            # Material must be FrontSide (doubleSided=False)
            self.assertFalse(m3["materials"][0]["doubleSided"], "Step 3 material must be FrontSide")

            # Final step must preserve 100% faces
            self.assertEqual(metrics_data["steps"][6]["metrics"]["faces"], 45000)

    def test_auto_detect_double_sided_input(self):
        """Verify a doubleSided=true source keeps doubleSided on Step 3 and Step 6 without --double-sided."""
        with tempfile.TemporaryDirectory(prefix="test_double_sided_pipeline_") as tmpdir:
            tmp = Path(tmpdir)
            ds_input = tmp / "sample_dinoki_double_sided.glb"
            ds_input.write_bytes(set_doublesided_material(SAMPLE_DINOKI.read_bytes()))
            out_dir = tmp / "out"

            pipeline = StepPipeline(
                resolution=512,
                texture_format="ktx2",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False
            )

            result = pipeline.run(ds_input, out_dir)
            self.assertTrue(result["success"])

            for step_file in ("step_03_texture_baked.glb", "step_06_final.glb"):
                materials = _read_glb_json(out_dir / step_file).get("materials", [])
                self.assertTrue(len(materials) > 0, f"No materials in {step_file}")
                for mat in materials:
                    self.assertIs(mat.get("doubleSided"), True, f"{step_file} material must keep doubleSided=True")

            # 100% triangles preserved across all steps
            metrics_data = json.loads((out_dir / "metrics.json").read_text())
            self.assertEqual(len(metrics_data["steps"]), 7)
            initial_faces = metrics_data["steps"][0]["metrics"]["faces"]
            for step_info in metrics_data["steps"]:
                self.assertEqual(step_info["metrics"]["faces"], initial_faces, f"Face count changed at {step_info['file']}")


class TestStepPipelineMockGlb(unittest.TestCase):
    """Synthetic textured icosphere (the CI smoke-test input) run through all 7 steps."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="test_mock_glb_pipeline_")
        tmp = Path(cls._tmp.name)
        raw_glb = tmp / "mock.glb"
        cls.orig_faces = create_mock_glb(raw_glb)
        StepPipeline(
            resolution=512,
            texture_format="webp",
            verbose=False,
            stream_events=False
        ).run(raw_glb, tmp / "out")
        cls.final_gltf = _read_glb_json(tmp / "out" / "step_06_final.glb")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_zero_decimation_synthetic_mesh(self):
        """Rule 11: the final GLB keeps 100% of the synthetic mesh triangles."""
        prim = self.final_gltf["meshes"][0]["primitives"][0]
        tris = self.final_gltf["accessors"][prim["indices"]]["count"] // 3
        self.assertEqual(tris, self.orig_faces)

    def test_extras_and_palette_embedding(self):
        """The final GLB embeds the palette and optimization metadata in glTF extras."""
        extras = self.final_gltf.get("extras", {})
        self.assertIn("palette", extras, "glTF extras missing 'palette'")
        self.assertIn("primaryColor", extras, "glTF extras missing 'primaryColor'")
        self.assertIn("policy", extras, "glTF extras missing 'policy'")
        self.assertEqual(extras["policy"], "STRICT 0-DECIMATION (--ratio 1.0)")
        self.assertGreaterEqual(len(extras["palette"]), 1)


def _read_glb_json(glb_path: Path) -> dict:
    """Parses the JSON chunk of a GLB file."""
    data = glb_path.read_bytes()
    chunk_len, chunk_type = struct.unpack("<I4s", data[12:20])
    assert chunk_type == b"JSON", f"First chunk of {glb_path.name} is not JSON"
    return json.loads(data[20:20 + chunk_len].decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
