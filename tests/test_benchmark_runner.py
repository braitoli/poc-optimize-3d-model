"""
tests/test_benchmark_runner.py

Unit and integration tests for scripts/benchmark_runner.py:
- Verifies profiler CLI invocation and argument parsing.
- Verifies high-precision time.perf_counter() metrics capture.
- Verifies Step 1 & Step 2 elapsed times and intermediate file metrics.
- Verifies Rule 11 Zero-Decimation geometric integrity check across steps.
- Verifies Texture resolution & GPU VRAM reduction calculations.
- Verifies structured JSON generation and disk output.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_runner.py"
SAMPLE_DINOKI = REPO_ROOT / "examples" / "sample_dinoki.glb"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path("python3")

from scripts.benchmark_runner import run_model_benchmark, calculate_geometry_vram, calculate_texture_vram


class TestBenchmarkRunner(unittest.TestCase):

    def test_vram_helper_formulas(self):
        """Validates geometry and texture GPU VRAM calculation helper functions."""
        # Texture: 1024x1024 RGBA uncompressed with 1.333 mipmaps
        uncompressed_vram = calculate_texture_vram(1024, 1024, is_ktx2=False)
        self.assertEqual(uncompressed_vram, int(round(1024 * 1024 * 4 * (4.0 / 3.0))))

        # Texture: 1024x1024 KTX2 with 1.333 mipmaps
        ktx2_vram = calculate_texture_vram(1024, 1024, is_ktx2=True)
        self.assertEqual(ktx2_vram, int(round(1024 * 1024 * 1 * (4.0 / 3.0))))
        self.assertAlmostEqual(uncompressed_vram / ktx2_vram, 4.0, delta=0.01)

        # Geometry: 45,000 faces and 36,648 vertices (<65,535 vertices -> 2B indices)
        geo_vram = calculate_geometry_vram(45000, 36648)
        expected_indices = 45000 * 3 * 2
        expected_verts = 36648 * 32
        self.assertEqual(geo_vram, expected_indices + expected_verts)

    def test_run_model_benchmark_dinoki(self):
        """Validates benchmark execution on dinoki model and verifies all required metric keys."""
        with tempfile.TemporaryDirectory(prefix="test_bench_dinoki_") as tmpdir:
            workdir = Path(tmpdir) / "dinoki"
            res = run_model_benchmark(
                model_key="dinoki",
                model_path=SAMPLE_DINOKI,
                texture_format="ktx2",
                workdir=workdir,
                clean_workdir=False,
                verbose=False
            )

            # 1. Timing metrics
            timing = res["timing"]
            self.assertGreater(timing["total_pipeline_execution_seconds"], 0.0)
            self.assertGreater(timing["total_pipeline_execution_ms"], 0.0)
            self.assertGreater(timing["step_01_elapsed_seconds"], 0.0)
            self.assertGreater(timing["step_02_elapsed_seconds"], 0.0)

            remaining = timing["remaining_step_durations"]
            self.assertIn("step_00_raw", remaining)
            self.assertIn("step_03_texture_baked", remaining)
            self.assertIn("step_04_palette_tagged", remaining)
            self.assertIn("step_05_meshopt", remaining)
            self.assertIn("step_06_final", remaining)

            self.assertEqual(len(timing["all_steps"]), 7)

            # 2. File size metrics
            fs = res["file_size"]
            self.assertEqual(fs["raw_input_bytes"], 2026696)
            self.assertEqual(fs["step_01_intermediate_file"], "step_01_cleaned_grounded.glb")
            self.assertGreater(fs["step_01_intermediate_bytes"], 1000000)
            # Dinoki keeps its 1536x1536 texture (1:1 fit is larger), and KTX2 UASTC of that texture
            # is larger on disk than the source JPEG: the file can grow while GPU VRAM shrinks.
            self.assertGreater(fs["final_output_bytes"], 0)
            self.assertEqual(fs["saved_bytes"], fs["raw_input_bytes"] - fs["final_output_bytes"])

            # 3. Rule 11 Zero-Decimation geometric integrity check
            geo = res["geometry_rule11"]
            self.assertEqual(geo["raw_triangles"], 45000)
            self.assertEqual(geo["step_01_triangles"], 45000)
            self.assertEqual(geo["step_02_triangles"], 45000)
            self.assertEqual(geo["final_triangles"], 45000)
            self.assertTrue(geo["zero_decimation_verified"])
            self.assertEqual(geo["triangles_preserved_percent"], 100.0)

            # 4. Textures and GPU VRAM before vs after
            tv = res["textures_and_vram"]
            self.assertEqual(tv["texture_resolution_before"], "1536x1536")
            # Dinoki's 1:1 fit is larger than its 1536x1536 texture, so Step 3 keeps the original
            self.assertEqual(tv["texture_resolution_after"], "1536x1536")
            self.assertEqual(tv["texture_format_before"], "JPEG")
            self.assertEqual(tv["texture_format_after"], "KTX2")
            self.assertGreater(tv["total_gpu_vram_saved_percent"], 50.0)
            self.assertGreater(tv["texture_vram_saved_percent"], 70.0)

            # 5. Intermediate files preservation check
            step1_file = workdir / "step_01_cleaned_grounded.glb"
            self.assertTrue(step1_file.exists())
            step2_file = workdir / "step_02_oriented.glb"
            self.assertTrue(step2_file.exists())
            final_file = workdir / "step_06_final.glb"
            self.assertTrue(final_file.exists())

    def test_cli_json_only(self):
        """Validates CLI invocation with --json-only outputs valid parseable JSON."""
        cmd = [
            str(PYTHON_BIN),
            str(SCRIPT_PATH),
            "--model", "dinoki",
            "--json-only"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Benchmark runner failed: {proc.stderr}")

        data = json.loads(proc.stdout)
        self.assertEqual(data["tool"], "poc-optimize-3d-model benchmark_runner")
        self.assertIn("results", data)
        self.assertEqual(data["results"]["model_key"], "dinoki")
        self.assertTrue(data["results"]["geometry_rule11"]["zero_decimation_verified"])


if __name__ == "__main__":
    unittest.main()
