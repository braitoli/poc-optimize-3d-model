"""
test_step_pipeline.py

Unit and integration tests for:
- optimizer/inspect_metrics.mjs (Comprehensive 3D Metrics Engine)
- optimizer/step_pipeline.py (Step-by-Step 8-stage Pipeline & Metrics)
"""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = REPO_ROOT / "optimizer" / "inspect_metrics.mjs"
SAMPLE_DINOKI = REPO_ROOT / "examples" / "sample_dinoki.glb"

from optimizer.step_pipeline import StepPipeline, inspect_glb_metrics
from optimizer.core.glb_utils import set_doublesided_material
from tests.fixtures import create_mock_glb, create_sphere_grid_glb


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
        """Verify all 7 GLB files are generated with 100% Zero-Decimation preservation (Step 3, the
        only step allowed to remove triangles, is skipped)."""
        with tempfile.TemporaryDirectory(prefix="test_step_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            pipeline = StepPipeline(
                texture_format="ktx2",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False,
                skip_steps=[3]
            )

            result = pipeline.run(SAMPLE_DINOKI, out_dir)
            self.assertTrue(result["success"])

            # Verify metrics.json
            metrics_file = out_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            metrics_data = json.loads(metrics_file.read_text())

            self.assertEqual(len(metrics_data["steps"]), 8)
            expected_files = [
                "step_00_raw.glb",
                "step_01_cleaned_grounded.glb",
                "step_02_oriented.glb",
                None,  # Step 3 (face reduction) is skipped: no GLB, no metrics
                "step_04_texture_baked.glb",
                "step_05_palette_tagged.glb",
                "step_06_meshopt.glb",
                "step_07_final.glb"
            ]

            for idx, expected_file in enumerate(expected_files):
                step_info = metrics_data["steps"][idx]
                self.assertEqual(step_info["file"], expected_file)
                if expected_file is None:
                    self.assertTrue(step_info["skipped"], "Step 3 must be recorded as skipped")
                    self.assertIsNone(step_info["metrics"])
                    self.assertFalse((out_dir / "step_03_face_reduced.glb").exists())
                    continue

                fpath = out_dir / expected_file
                self.assertTrue(fpath.exists(), f"Missing step file: {expected_file}")
                self.assertGreater(fpath.stat().st_size, 100000, f"File too small: {expected_file}")
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

            # Final step keeps meshopt; dinoki's 1536x1536 texture (12 MB VRAM < 20 MB default) skips KTX2
            final_step = metrics_data["steps"][7]["metrics"]
            self.assertIn("EXT_meshopt_compression", final_step["extensions"]["used"])
            self.assertNotIn("KHR_texture_basisu", final_step["extensions"]["used"])
            self.assertIs(final_step["gpuCompressionSkipped"], True)
            self.assertEqual(final_step["textures"][0]["format"], "JPEG")
            self.assertGreater(len(final_step["palette"]), 0)

            # Step 4 must have FrontSide rendering (doubleSided=False) when double_sided=False
            step4_metrics = metrics_data["steps"][4]["metrics"]
            self.assertFalse(step4_metrics["materials"][0]["doubleSided"], "Step 4 material must be doubleSided=False for FrontSide")

    def test_default_downscale_keeps_original_when_fit_not_smaller(self):
        """Default (downscale on, exact): Dinoki's source UVs overlap (per-face UV area ~123% of its
        1536x1536 canvas), so the 1:1 fit is larger than the original and Step 4 keeps it."""
        with tempfile.TemporaryDirectory(prefix="test_default_downscale_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            StepPipeline(texture_format="webp", verbose=False, stream_events=False,
                         skip_steps=[3]).run(SAMPLE_DINOKI, out_dir)
            metrics_data = json.loads((out_dir / "metrics.json").read_text())
            self.assertEqual(len(metrics_data["steps"]), 8)

            m4 = metrics_data["steps"][4]["metrics"]
            self.assertIs(m4["downscale"], True)
            self.assertEqual(m4["sizeMode"], "exact")
            self.assertEqual(m4["uvMode"], "xatlas")
            self.assertEqual(m4["fitResolution"] % 4, 0)
            self.assertGreaterEqual(m4["fitResolution"] ** 2, 1536 * 1536)
            self.assertEqual(m4["originalResolution"], "1536x1536")
            self.assertEqual(m4["finalResolution"], "1536x1536")
            self.assertIs(m4["downscaled"], False)
            self.assertTrue(m4["decision"].startswith("kept_original"), m4["decision"])
            self.assertEqual(m4["texelDensityRatio"], 1.0)
            self.assertFalse(m4["materials"][0]["doubleSided"])

            # Original texture bitstream kept, never resized in Steps 6-7
            self.assertEqual(_glb_image_bytes(out_dir / "step_04_texture_baked.glb"),
                             _glb_image_bytes(out_dir / "step_00_raw.glb"))
            for step in (6, 7):
                self.assertEqual(metrics_data["steps"][step]["metrics"]["textures"][0]["resolutionFormatted"], "1536x1536")
            for step_info in metrics_data["steps"]:
                if step_info["skipped"]:
                    continue
                self.assertEqual(step_info["metrics"]["faces"], 45000)

    def test_downscale_off_keeps_original_uvs_and_texture(self):
        """downscale off: Step 4 exports the Step 2 mesh unchanged (UVs, faces) with the original
        texture bitstream; Steps 6-7 keep the original texture dimensions."""
        with tempfile.TemporaryDirectory(prefix="test_downscale_off_pipeline_") as tmpdir:
            out_dir = Path(tmpdir)
            StepPipeline(downscale=False, texture_format="webp", verbose=False, stream_events=False,
                         skip_steps=[3]).run(SAMPLE_DINOKI, out_dir)
            metrics_data = json.loads((out_dir / "metrics.json").read_text())

            m4 = metrics_data["steps"][4]["metrics"]
            self.assertIs(m4["downscale"], False)
            self.assertIsNone(m4["fitResolution"])
            self.assertEqual(m4["decision"], "kept_original: downscale off")
            self.assertIs(m4["downscaled"], False)
            self.assertEqual(m4["texelDensityRatio"], 1.0)
            self.assertEqual(m4["originalResolution"], "1536x1536")
            self.assertEqual(m4["finalResolution"], "1536x1536")
            self.assertFalse(m4["materials"][0]["doubleSided"], "Step 4 material must be FrontSide")

            step2 = trimesh.load(str(out_dir / "step_02_oriented.glb"), force="mesh", process=False)
            step4 = trimesh.load(str(out_dir / "step_04_texture_baked.glb"), force="mesh", process=False)
            self.assertTrue(np.array_equal(step2.faces, step4.faces))
            self.assertTrue(np.array_equal(step2.visual.uv, step4.visual.uv))
            self.assertEqual(_glb_image_bytes(out_dir / "step_04_texture_baked.glb"),
                             _glb_image_bytes(out_dir / "step_00_raw.glb"))

            for step in (6, 7):
                self.assertEqual(metrics_data["steps"][step]["metrics"]["textures"][0]["resolutionFormatted"], "1536x1536")
            self.assertEqual(metrics_data["steps"][7]["metrics"]["faces"], 45000)

    def test_invalid_options_raise(self):
        with self.assertRaises(ValueError):
            StepPipeline(size_mode="bogus")
        with self.assertRaises(TypeError):
            StepPipeline(downscale="on")
        with self.assertRaises(TypeError):
            StepPipeline(resolution=1024)

    def test_cli_rejects_removed_and_unknown_options(self):
        for extra in (["--resolution", "512"], ["--size-mode", "bogus"], ["--downscale", "maybe"]):
            with tempfile.TemporaryDirectory(prefix="test_cli_options_") as tmpdir:
                proc = subprocess.run(
                    [sys.executable, "-m", "optimizer.step_pipeline", str(SAMPLE_DINOKI), "--output-dir", tmpdir, *extra],
                    capture_output=True, text=True, cwd=REPO_ROOT
                )
                self.assertEqual(proc.returncode, 2, f"{extra}: {proc.stderr}")
                self.assertEqual(list(Path(tmpdir).iterdir()), [], f"{extra}: pipeline must not run")

    def test_auto_detect_double_sided_input(self):
        """Verify a doubleSided=true source keeps doubleSided on Step 4 and Step 7 without --double-sided."""
        with tempfile.TemporaryDirectory(prefix="test_double_sided_pipeline_") as tmpdir:
            tmp = Path(tmpdir)
            ds_input = tmp / "sample_dinoki_double_sided.glb"
            ds_input.write_bytes(set_doublesided_material(SAMPLE_DINOKI.read_bytes()))
            out_dir = tmp / "out"

            pipeline = StepPipeline(
                texture_format="ktx2",
                smooth_normals=True,
                double_sided=False,
                verbose=False,
                stream_events=False,
                skip_steps=[3]
            )

            result = pipeline.run(ds_input, out_dir)
            self.assertTrue(result["success"])

            for step_file in ("step_04_texture_baked.glb", "step_07_final.glb"):
                materials = _read_glb_json(out_dir / step_file).get("materials", [])
                self.assertTrue(len(materials) > 0, f"No materials in {step_file}")
                for mat in materials:
                    self.assertIs(mat.get("doubleSided"), True, f"{step_file} material must keep doubleSided=True")

            # 100% triangles preserved across all steps
            metrics_data = json.loads((out_dir / "metrics.json").read_text())
            self.assertEqual(len(metrics_data["steps"]), 8)
            initial_faces = metrics_data["steps"][0]["metrics"]["faces"]
            for step_info in metrics_data["steps"]:
                if step_info["skipped"]:
                    continue
                self.assertEqual(step_info["metrics"]["faces"], initial_faces, f"Face count changed at {step_info['file']}")


class TestStepPipelineMockGlb(unittest.TestCase):
    """Synthetic textured icosphere (the CI smoke-test input) run through all 8 steps, with the
    Step 3 face reduction skipped so every triangle survives."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="test_mock_glb_pipeline_")
        tmp = Path(cls._tmp.name)
        raw_glb = tmp / "mock.glb"
        cls.orig_faces = create_mock_glb(raw_glb)
        StepPipeline(
            texture_format="webp",
            verbose=False,
            stream_events=False,
            skip_steps=[3]
        ).run(raw_glb, tmp / "out")
        cls.final_gltf = _read_glb_json(tmp / "out" / "step_07_final.glb")

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


class TestStepPipelineRechart(unittest.TestCase):
    """Synthetic sphere grid whose islands use a small part of a 2048x2048 texture: the 1:1 fit is
    far smaller than the original, so downscale re-charts it (exact) or keeps it (pot-up >= 2048)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="test_rechart_pipeline_")
        tmp = Path(cls._tmp.name)
        cls.raw_glb = tmp / "grid.glb"
        cls.orig_faces = create_sphere_grid_glb(cls.raw_glb, src_res=2048)
        cls.exact_dir = tmp / "exact"
        # ktx2_min_vram_mb=0: the fit canvas is below the 20 MB VRAM threshold, force the KTX2 path
        StepPipeline(texture_format="ktx2", ktx2_min_vram_mb=0, verbose=False, stream_events=False,
                     skip_steps=[3]).run(cls.raw_glb, cls.exact_dir)
        cls.exact = json.loads((cls.exact_dir / "metrics.json").read_text())
        cls.pot_up_dir = tmp / "pot_up"
        StepPipeline(size_mode="pot-up", texture_format="webp", verbose=False, stream_events=False,
                     skip_steps=[3]).run(cls.raw_glb, cls.pot_up_dir)
        cls.pot_up = json.loads((cls.pot_up_dir / "metrics.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exact_recharts_at_fit_resolution(self):
        m4 = self.exact["steps"][4]["metrics"]
        fit = m4["fitResolution"]
        self.assertEqual(fit % 4, 0)
        self.assertLess(fit, 2048)
        self.assertIs(m4["downscaled"], True)
        self.assertTrue(m4["decision"].startswith("rechart"), m4["decision"])
        self.assertEqual(m4["originalResolution"], "2048x2048")
        self.assertEqual(m4["finalResolution"], f"{fit}x{fit}")
        self.assertEqual(m4["textures"][0]["resolutionFormatted"], f"{fit}x{fit}")
        self.assertGreaterEqual(m4["texelDensityRatio"], 0.97)
        for step_info in self.exact["steps"]:
            if step_info["skipped"]:
                continue
            self.assertEqual(step_info["metrics"]["faces"], self.orig_faces, step_info["file"])

    def test_exact_ktx2_keeps_fit_resolution(self):
        """Step 7 compresses the non power-of-two canvas to KTX2 without resizing it."""
        fit = self.exact["steps"][4]["metrics"]["fitResolution"]
        final = self.exact_dir / "step_07_final.glb"
        self.assertIn("KHR_texture_basisu", _read_glb_json(final)["extensionsUsed"])
        self.assertEqual(_ktx2_dimensions(_glb_image_bytes(final)), (fit, fit))

    def test_pot_up_keeps_original_when_canvas_not_smaller(self):
        m4 = self.pot_up["steps"][4]["metrics"]
        self.assertEqual(m4["sizeMode"], "pot-up")
        self.assertLess(m4["fitResolution"], 2048)
        self.assertIs(m4["downscaled"], False)
        self.assertTrue(m4["decision"].startswith("kept_original"), m4["decision"])
        self.assertIn("2048x2048 >= original 2048x2048", m4["decision"])
        self.assertEqual(m4["finalResolution"], "2048x2048")
        self.assertEqual(m4["texelDensityRatio"], 1.0)
        self.assertEqual(_glb_image_bytes(self.pot_up_dir / "step_04_texture_baked.glb"),
                         _glb_image_bytes(self.pot_up_dir / "step_00_raw.glb"))


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


def _glb_image_bytes(glb_path: Path, index: int = 0) -> bytes:
    """Raw bytes of an embedded glTF image (bufferView-backed)."""
    gltf, bin_chunk = _read_glb(glb_path)
    view = gltf["bufferViews"][gltf["images"][index]["bufferView"]]
    start = view.get("byteOffset", 0)
    return bin_chunk[start:start + view["byteLength"]]


def _ktx2_dimensions(data: bytes):
    """(pixelWidth, pixelHeight) from a KTX2 header."""
    assert data[:12] == b"\xabKTX 20\xbb\r\n\x1a\n", "not a KTX2 file"
    return struct.unpack("<II", data[20:28])


def _read_glb_json(glb_path: Path) -> dict:
    """Parses the JSON chunk of a GLB file."""
    data = glb_path.read_bytes()
    chunk_len, chunk_type = struct.unpack("<I4s", data[12:20])
    assert chunk_type == b"JSON", f"First chunk of {glb_path.name} is not JSON"
    return json.loads(data[20:20 + chunk_len].decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
