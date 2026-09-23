"""
test_step_skipping.py

Switching individual pipeline steps off (StepPipeline(skip_steps=...) / --skip-steps).
A skipped step writes no GLB, records itself as skipped, and nothing downstream reads what it
would have produced; the steps that do run still produce a complete final model.
"""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from optimizer.step_pipeline import StepPipeline
from tests.fixtures import create_mock_glb

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
FINAL_STEP = StepPipeline.TOTAL_STEPS - 1


def read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length])


def run_pipeline(out_dir: Path, source: Path, **kwargs) -> dict:
    pipeline = StepPipeline(
        texture_format="original",
        verbose=False,
        stream_events=False,
        **kwargs
    )
    return pipeline.run(source, out_dir)


class TestSkipValidation(unittest.TestCase):
    def test_mandatory_steps_cannot_be_skipped(self):
        for step in (0, FINAL_STEP):
            with self.subTest(step=step):
                with self.assertRaises(ValueError) as ctx:
                    StepPipeline(skip_steps=[step])
                self.assertIn(f"Step {step} cannot be skipped", str(ctx.exception))

    def test_optional_steps_are_every_step_in_between(self):
        self.assertEqual(StepPipeline.OPTIONAL_STEPS, tuple(range(1, FINAL_STEP)))
        StepPipeline(skip_steps=list(StepPipeline.OPTIONAL_STEPS), reduce_ops=["repair"])

    def test_a_step_index_must_be_an_integer(self):
        with self.assertRaises(TypeError):
            StepPipeline(skip_steps=["3"])
        with self.assertRaises(TypeError):
            StepPipeline(skip_steps=[True])

    def test_an_unknown_step_is_rejected(self):
        with self.assertRaises(ValueError):
            StepPipeline(skip_steps=[99])

    def test_merging_faces_needs_the_uv_rechart_step(self):
        # Step 3's merge collapses edges and throws the model's UVs away; only Step 4 makes new ones
        with self.assertRaises(ValueError) as ctx:
            StepPipeline(skip_steps=[4], reduce_ops=["repair", "merge"])
        self.assertIn("merge", str(ctx.exception))
        self.assertIn("Step 4", str(ctx.exception))

    def test_merging_is_fine_when_step_3_itself_is_skipped(self):
        StepPipeline(skip_steps=[3, 4], reduce_ops=["repair", "merge"])


class TestSkippedStepsInAPipelineRun(unittest.TestCase):
    """One real run with three steps switched off, inspected from every angle."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="skip_steps_")
        tmp_dir = Path(cls._tmp.name)
        cls.source = tmp_dir / "input.glb"
        cls.raw_faces = create_mock_glb(cls.source, subdivisions=3)
        cls.out_dir = tmp_dir / "out"
        cls.skipped = [2, 3, 5]
        cls.result = run_pipeline(cls.out_dir, cls.source, skip_steps=cls.skipped)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_a_skipped_step_writes_no_glb(self):
        for step in self.skipped:
            with self.subTest(step=step):
                self.assertFalse((self.out_dir / StepPipeline.STEP_DEFINITIONS[step]["file"]).exists())

    def test_every_other_step_writes_its_glb(self):
        for definition in StepPipeline.STEP_DEFINITIONS:
            if definition["step"] in self.skipped:
                continue
            with self.subTest(step=definition["step"]):
                self.assertTrue((self.out_dir / definition["file"]).exists())

    def test_the_summary_lists_what_was_skipped(self):
        summary = self.result["summary"]
        self.assertEqual(summary["skippedSteps"], self.skipped)
        self.assertEqual(
            summary["files"],
            [d["file"] for d in StepPipeline.STEP_DEFINITIONS if d["step"] not in self.skipped]
        )

    def test_metrics_json_marks_each_skipped_step(self):
        steps = {s["step"]: s for s in json.loads((self.out_dir / "metrics.json").read_text())["steps"]}
        self.assertEqual(sorted(steps), list(range(StepPipeline.TOTAL_STEPS)))
        for step, entry in steps.items():
            with self.subTest(step=step):
                if step in self.skipped:
                    self.assertTrue(entry["skipped"])
                    self.assertIsNone(entry["file"])
                    self.assertIsNone(entry["metrics"])
                else:
                    self.assertFalse(entry["skipped"])
                    self.assertIsNotNone(entry["metrics"])

    def test_skipping_the_palette_step_leaves_no_palette_behind(self):
        # 5 is skipped in this run: nothing downstream may invent a palette for the final model
        extras = read_glb_json(self.out_dir / StepPipeline.STEP_DEFINITIONS[FINAL_STEP]["file"]).get("extras", {})
        self.assertNotIn("palette", extras)
        self.assertNotIn("primaryColor", extras)
        self.assertIsNone(self.result["summary"]["primaryColor"])
        self.assertEqual(self.result["summary"]["palette"], [])

    def test_skipping_face_reduction_preserves_every_triangle(self):
        # 3 is skipped in this run, so the strict zero-decimation rule applies end to end
        summary = self.result["summary"]
        self.assertFalse(summary["faceReduction"]["enabled"])
        self.assertEqual(summary["finalFaces"], self.raw_faces)
        self.assertTrue(summary["zeroDecimationVerified"])


class TestStepDependenciesAreDropped(unittest.TestCase):
    def test_skipping_the_bake_step_keeps_the_original_texture(self):
        with tempfile.TemporaryDirectory(prefix="skip_bake_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=3)
            result = run_pipeline(tmp_dir / "out", source, skip_steps=[3, 4])
            # create_mock_glb paints a 256x256 texture; with no re-chart it is carried through as is
            self.assertEqual(result["summary"]["files"].count("step_04_texture_baked.glb"), 0)
            extras = read_glb_json(tmp_dir / "out" / "step_07_final.glb")["extras"]
            self.assertEqual(extras["resolution"], "256x256")

    def test_face_reduction_runs_without_any_other_optional_step(self):
        with tempfile.TemporaryDirectory(prefix="only_reduce_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            raw_faces = create_mock_glb(source, subdivisions=3)
            result = run_pipeline(
                tmp_dir / "out", source,
                skip_steps=[1, 2, 5, 6],
                reduce_ops=["repair", "isolated", "hidden", "merge"],
                # A sphere this coarse cannot be collapsed at all within the default budgets
                reduce_quality_budget=2.0,
                reduce_normal_budget=30.0
            )
            reduction = result["summary"]["faceReduction"]
            self.assertTrue(reduction["enabled"])
            self.assertEqual(reduction["facesBefore"], raw_faces)
            self.assertLess(reduction["facesAfter"], raw_faces)
            self.assertTrue(reduction["withinQualityBudget"])
            # Everything after Step 3 still preserves what it was given
            self.assertEqual(result["summary"]["finalFaces"], reduction["facesAfter"])
            self.assertTrue(result["summary"]["zeroDecimationVerified"])


class TestSkipStepsOnTheCli(unittest.TestCase):
    def test_cli_emits_a_step_skipped_event_for_each_one(self):
        with tempfile.TemporaryDirectory(prefix="skip_cli_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=2)
            proc = subprocess.run(
                [
                    str(PYTHON_BIN), "-m", "optimizer.step_pipeline", str(source),
                    "--output-dir", str(tmp_dir / "out"),
                    "--format", "original",
                    "--skip-steps", "3,5",
                    "--quiet"
                ],
                cwd=str(REPO_ROOT), capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            skipped = [e["step"] for e in events if e["event"] == "step_skipped"]
            completed = [e["step"] for e in events if e["event"] == "step_complete"]
            self.assertEqual(skipped, [3, 5])
            self.assertEqual(completed, [0, 1, 2, 4, 6, 7])

    def test_cli_rejects_a_non_numeric_step(self):
        with tempfile.TemporaryDirectory(prefix="skip_cli_bad_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=1)
            proc = subprocess.run(
                [
                    str(PYTHON_BIN), "-m", "optimizer.step_pipeline", str(source),
                    "--output-dir", str(tmp_dir / "out"),
                    "--skip-steps", "palette",
                    "--quiet"
                ],
                cwd=str(REPO_ROOT), capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 1)
            error = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(error["event"], "pipeline_error")
            self.assertIn("palette", error["error"])


if __name__ == "__main__":
    unittest.main()
