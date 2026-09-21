"""
test_pipeline_errors.py

The single error path of the pipeline:
- optimizer/core/errors.py (PipelineAbort)
- StepPipeline.run attaches the failing step number to any exception that escapes
- `python -m optimizer.step_pipeline` reports a failure as one `pipeline_error` NDJSON line on
  stdout (full traceback on stderr) and exits 1; the success path keeps its NDJSON events.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from optimizer.core.errors import PipelineAbort
from optimizer.step_pipeline import StepPipeline
from tests.fixtures import create_mock_glb

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "optimizer.step_pipeline", str(input_path), "--output-dir", str(output_dir), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT
    )


def stdout_events(stdout: str):
    """Every non-empty stdout line must be one JSON event."""
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


class TestPipelineAbort(unittest.TestCase):
    def test_carries_reason_and_optional_step(self):
        err = PipelineAbort("texture missing", step=3)
        self.assertEqual(err.reason, "texture missing")
        self.assertEqual(err.step, 3)
        self.assertEqual(str(err), "texture missing")
        self.assertIsNone(PipelineAbort("no step").step)


class TestRunAttachesStep(unittest.TestCase):
    def test_missing_input_fails_at_step_0(self):
        with tempfile.TemporaryDirectory(prefix="test_run_missing_") as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                StepPipeline(verbose=False, stream_events=False).run(tmp / "missing.glb", tmp / "out")
            self.assertEqual(ctx.exception.step, 0)

    def test_failure_in_a_later_step_carries_that_step(self):
        with tempfile.TemporaryDirectory(prefix="test_run_step4_") as tmpdir:
            tmp = Path(tmpdir)
            glb = tmp / "mock.glb"
            create_mock_glb(glb)
            with mock.patch("optimizer.step_pipeline.extract_palette", side_effect=PipelineAbort("palette boom")):
                with self.assertRaises(PipelineAbort) as ctx:
                    StepPipeline(texture_format="webp", verbose=False, stream_events=False).run(glb, tmp / "out")
            self.assertEqual(ctx.exception.step, 4)
            self.assertEqual(ctx.exception.reason, "palette boom")


class TestCliPipelineError(unittest.TestCase):
    def assert_single_pipeline_error(self, proc: subprocess.CompletedProcess, step):
        self.assertEqual(proc.returncode, 1, proc.stderr)
        events = stdout_events(proc.stdout)
        errors = [e for e in events if e["event"] == "pipeline_error"]
        self.assertEqual(len(errors), 1, proc.stdout)
        self.assertIs(events[-1], errors[0], "pipeline_error must be the last stdout event")
        err = errors[0]
        self.assertEqual(set(err), {"event", "error", "errorType", "step"})
        self.assertEqual(err["step"], step)
        self.assertIsInstance(err["error"], str)
        self.assertTrue(err["error"].strip(), "error reason must not be empty")
        self.assertTrue(err["errorType"])
        self.assertNotIn("Traceback", err["error"])
        self.assertIn("Traceback (most recent call last)", proc.stderr, "full traceback goes to stderr")
        return err

    def test_missing_input_file(self):
        with tempfile.TemporaryDirectory(prefix="test_cli_missing_") as tmpdir:
            tmp = Path(tmpdir)
            proc = run_cli(tmp / "does_not_exist.glb", tmp / "out")
            err = self.assert_single_pipeline_error(proc, step=0)
            self.assertEqual(err["errorType"], "FileNotFoundError")
            self.assertIn("does_not_exist.glb", err["error"])

    def test_non_glb_input(self):
        with tempfile.TemporaryDirectory(prefix="test_cli_non_glb_") as tmpdir:
            tmp = Path(tmpdir)
            bogus = tmp / "not_a_model.glb"
            bogus.write_bytes(b"plain text, not a binary glTF container\n" * 8)
            proc = run_cli(bogus, tmp / "out")
            self.assert_single_pipeline_error(proc, step=0)
            self.assertEqual([e["event"] for e in stdout_events(proc.stdout)], ["pipeline_error"])

    def test_success_emits_only_step_and_complete_events(self):
        with tempfile.TemporaryDirectory(prefix="test_cli_ok_") as tmpdir:
            tmp = Path(tmpdir)
            glb = tmp / "mock.glb"
            create_mock_glb(glb)
            proc = run_cli(glb, tmp / "out", "--format", "webp", "--quiet")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            events = stdout_events(proc.stdout)
            self.assertEqual([e["event"] for e in events], ["step_complete"] * 7 + ["pipeline_complete"])
            self.assertEqual([e["step"] for e in events[:7]], list(range(7)))


if __name__ == "__main__":
    unittest.main()
