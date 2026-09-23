"""
tests/test_benchmark_runner.py

Unit and integration tests for scripts/benchmark_runner.py:
- Verifies profiler CLI invocation and argument parsing.
- Verifies high-precision time.perf_counter() metrics capture.
- Verifies Step 1 & Step 2 elapsed times and intermediate file metrics.
- Verifies the Rule 11 geometric integrity check: Step 3 is the only step allowed to remove
  triangles (within its quality budget), every other step preserves 100% of what it is given.
- Verifies Texture resolution & GPU VRAM reduction calculations.
- Verifies structured JSON generation and disk output.
- Verifies a failing model is recorded (reason, errorType, step) and the run continues.
"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from optimizer.core.face_reduce import DEFAULT_QUALITY_BUDGET_PERCENT, ENGINES as REDUCE_ENGINES

DEFAULT_REDUCE_ENGINE = REDUCE_ENGINES[0]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_runner.py"
SAMPLE_DINOKI = REPO_ROOT / "examples" / "sample_dinoki.glb"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path("python3")

from scripts import benchmark_runner
from scripts.benchmark_runner import run_model_benchmark, calculate_geometry_vram, calculate_texture_vram
from tests.fixtures import create_mock_glb


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
            self.assertIn("step_03_face_reduced", remaining)
            self.assertIn("step_04_texture_baked", remaining)
            self.assertIn("step_05_palette_tagged", remaining)
            self.assertIn("step_06_meshopt", remaining)
            self.assertIn("step_07_final", remaining)

            self.assertEqual(len(timing["all_steps"]), 8)

            # 2. File size metrics
            fs = res["file_size"]
            self.assertEqual(fs["raw_input_bytes"], 2026696)
            self.assertEqual(fs["step_01_intermediate_file"], "step_01_cleaned_grounded.glb")
            self.assertGreater(fs["step_01_intermediate_bytes"], 1000000)
            # Step 3 drops most of dinoki's triangles and Step 7 keeps the (small) texture
            # uncompressed, so the final file is smaller than the raw input.
            self.assertGreater(fs["final_output_bytes"], 0)
            self.assertLess(fs["final_output_bytes"], fs["raw_input_bytes"])
            self.assertEqual(fs["saved_bytes"], fs["raw_input_bytes"] - fs["final_output_bytes"])

            # 3. Rule 11: Step 3 is the only step that may remove triangles, within its budget
            geo = res["geometry_rule11"]
            self.assertEqual(geo["raw_triangles"], 45000)
            self.assertEqual(geo["step_01_triangles"], 45000)
            self.assertEqual(geo["step_02_triangles"], 45000)
            self.assertTrue(geo["step_03_enabled"])
            self.assertEqual(geo["step_03_engine"], DEFAULT_REDUCE_ENGINE)
            self.assertEqual(geo["step_03_triangles_before"], 45000)
            self.assertLess(geo["step_03_triangles_after"], 45000)
            self.assertEqual(geo["step_03_triangles_removed"],
                             geo["step_03_triangles_before"] - geo["step_03_triangles_after"])
            self.assertGreater(geo["step_03_reduction_percent"], 0.0)
            self.assertEqual(geo["step_03_quality_budget_percent"], DEFAULT_QUALITY_BUDGET_PERCENT)
            self.assertLessEqual(geo["step_03_deviation_percent"], geo["step_03_quality_budget_percent"])
            self.assertTrue(geo["step_03_within_quality_budget"])
            # Every step after Step 3 keeps 100% of the triangles Step 3 left
            self.assertEqual(geo["final_triangles"], geo["step_03_triangles_after"])
            self.assertTrue(geo["zero_decimation_verified"])
            self.assertEqual(geo["triangles_preserved_percent"], 100.0)

            # 4. Textures and GPU VRAM before vs after
            metrics_data = json.loads((workdir / "metrics.json").read_text())
            step4_metrics = metrics_data["steps"][4]["metrics"]
            step7_metrics = metrics_data["steps"][7]["metrics"]
            tv = res["textures_and_vram"]
            self.assertEqual(tv["texture_resolution_before"], "1536x1536")
            self.assertEqual(tv["texture_format_before"], "JPEG")
            # Step 4 bakes a PNG onto its 1:1 fit canvas rather than keeping the original texture
            # Step 4 re-charts either way here: on its own because the 1:1 canvas is smaller than
            # the original texture, or forced when Step 3's merge left the model without UVs
            self.assertTrue(step4_metrics["decision"].startswith("rechart"), step4_metrics["decision"])
            self.assertTrue(step4_metrics["downscaled"])
            fit = step4_metrics["fitResolution"]
            self.assertEqual(tv["texture_resolution_after"], f"{fit}x{fit}")
            self.assertEqual(tv["texture_format_after"], "PNG")
            # That canvas still costs less than the 20 MB KTX2 threshold, so Step 7 keeps it as is
            self.assertLess(tv["texture_vram_after_bytes"], 20 * 1024 * 1024)
            self.assertIs(step7_metrics["gpuCompressionSkipped"], True)
            self.assertTrue(step7_metrics["gpuCompressionReason"].startswith("texture VRAM "),
                            step7_metrics["gpuCompressionReason"])
            self.assertIn("< 20 MB", step7_metrics["gpuCompressionReason"])

            # 5. Intermediate files preservation check
            step1_file = workdir / "step_01_cleaned_grounded.glb"
            self.assertTrue(step1_file.exists())
            step2_file = workdir / "step_02_oriented.glb"
            self.assertTrue(step2_file.exists())
            step3_file = workdir / "step_03_face_reduced.glb"
            self.assertTrue(step3_file.exists())
            final_file = workdir / "step_07_final.glb"
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
        self.assertEqual(data["failures"], [])

    def test_failing_models_are_recorded_and_run_continues(self):
        """A model that raises is recorded as failed and the next model still runs; exit code 1."""
        with tempfile.TemporaryDirectory(prefix="test_bench_failures_") as tmpdir:
            tmp = Path(tmpdir)
            not_glb = tmp / "not_a_model.glb"
            not_glb.write_bytes(b"plain text, not a binary glTF container\n" * 8)
            good = tmp / "mock.glb"
            create_mock_glb(good)
            out_json = tmp / "bench.json"
            models = {
                "dinoki": {**benchmark_runner.BASELINE_MODELS["dinoki"], "path": not_glb},
                "koidrax": {**benchmark_runner.BASELINE_MODELS["koidrax"], "path": tmp / "missing.glb"},
                "vulparon": {**benchmark_runner.BASELINE_MODELS["vulparon"], "path": good},
            }
            argv = ["benchmark_runner.py", "--all", "--mode", "optimized", "--format", "webp",
                    "--json-only", "-o", str(out_json)]
            with mock.patch.dict(benchmark_runner.BASELINE_MODELS, models), \
                    mock.patch.object(sys, "argv", argv), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    benchmark_runner.main()
            self.assertEqual(ctx.exception.code, 1)

            data = json.loads(out_json.read_text())
            failures = {f["model_key"]: f for f in data["failures"]}
            self.assertEqual(list(failures), ["dinoki", "koidrax"])
            for f in failures.values():
                self.assertEqual(set(f), {"model_key", "model_path", "mode", "error", "errorType", "step"})
                self.assertTrue(f["error"])
                self.assertEqual(f["mode"], "optimized")
            self.assertEqual(failures["dinoki"]["step"], 0)
            self.assertEqual(failures["koidrax"]["errorType"], "FileNotFoundError")
            self.assertIn("missing.glb", failures["koidrax"]["error"])
            self.assertIsNone(failures["koidrax"]["step"])

            # The model after the failures still ran to completion
            self.assertEqual(data["models_failed_count"], 2)
            self.assertEqual(data["models_profiled_count"], 1)
            self.assertEqual(data["results"]["model_key"], "vulparon")
            self.assertTrue(data["results"]["geometry_rule11"]["zero_decimation_verified"])


if __name__ == "__main__":
    unittest.main()
