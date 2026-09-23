"""
test_smooth_normals_option.py

Step 6's angle-weighted smooth normals: on by default, switchable, and recorded in the metrics so
the dashboard can state what actually happened instead of assuming it.

Welding is restricted to vertices whose normals already agree, so a hard edge - one position, two
normals, which is how glTF draws a crease - survives the pass. Welding a position whole flattened
every crease in the model (measured on dinoki: 5,195 creased positions in, 0 out).
"""

import unittest
from unittest import mock

from optimizer.step_pipeline import StepPipeline


class TestSmoothNormalsOption(unittest.TestCase):
    def test_on_by_default(self):
        self.assertTrue(StepPipeline().smooth_normals)

    def test_explicitly_switchable(self):
        self.assertFalse(StepPipeline(smooth_normals=False).smooth_normals)
        self.assertTrue(StepPipeline(smooth_normals=True).smooth_normals)

    def test_cli_defaults_to_the_pipeline_default(self):
        """Neither --smooth-normals nor --no-smooth-normals leaves the decision to the pipeline."""
        parser_args = ["input.glb", "--output-dir", "out"]
        with mock.patch("sys.argv", ["step_pipeline"] + parser_args):
            from optimizer import step_pipeline
            parser = step_pipeline.argparse.ArgumentParser()
            parser.add_argument("--smooth-normals", dest="smooth_normals",
                                action="store_true", default=None)
            parser.add_argument("--no-smooth-normals", dest="smooth_normals",
                                action="store_false", default=None)
            self.assertIsNone(parser.parse_args([]).smooth_normals)
            self.assertTrue(parser.parse_args(["--smooth-normals"]).smooth_normals)
            self.assertFalse(parser.parse_args(["--no-smooth-normals"]).smooth_normals)


if __name__ == "__main__":
    unittest.main()
