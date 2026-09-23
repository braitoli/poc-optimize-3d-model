"""
test_uv_island_merge.py

Step 4's UV island merging: chart the mesh at several merge levels and keep the one whose islands
need the smallest canvas. Fewer islands mean less chart border, and border is what the packer has
to surround with gutter padding.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from optimizer.core.uv_baker import (
    ISLAND_MERGE_LEVELS,
    _count_uv_islands,
    get_adaptive_chart_options,
    plan_uv_canvas,
    uv_boundary_length
)
from optimizer.step_pipeline import StepPipeline
from tests.fixtures import create_mock_glb


def quad_uv(offset=(0.0, 0.0)):
    """A unit square as two triangles, plus its UVs at `offset`."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) + np.asarray(offset)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return uv, faces


class TestBoundaryMeasurements(unittest.TestCase):
    def test_boundary_length_of_one_island_is_its_perimeter(self):
        uv, faces = quad_uv()
        self.assertAlmostEqual(uv_boundary_length(uv, faces), 4.0, places=6)

    def test_the_shared_edge_of_two_triangles_is_not_boundary(self):
        # The diagonal is used by both triangles, so only the square's four sides count
        uv, faces = quad_uv()
        diagonal = np.linalg.norm(uv[0] - uv[2])
        self.assertLess(uv_boundary_length(uv, faces), 4.0 + diagonal)

    def test_two_islands_have_twice_the_border_of_one(self):
        uv_a, faces_a = quad_uv()
        uv_b, faces_b = quad_uv(offset=(5.0, 0.0))
        uv = np.vstack([uv_a, uv_b])
        faces = np.vstack([faces_a, faces_b + len(uv_a)])
        self.assertEqual(_count_uv_islands(faces), 2)
        self.assertAlmostEqual(uv_boundary_length(uv, faces), 8.0, places=6)

    def test_island_count_follows_the_uv_topology(self):
        _, faces = quad_uv()
        self.assertEqual(_count_uv_islands(faces), 1)


class TestChartMergeLevels(unittest.TestCase):
    def test_a_higher_level_lets_charts_grow_further(self):
        base = get_adaptive_chart_options(1000, merge_level=0)
        merged = get_adaptive_chart_options(1000, merge_level=2)
        self.assertGreater(merged.max_cost, base.max_cost)
        for penalty in ("normal_deviation_weight", "straightness_weight", "normal_seam_weight", "texture_seam_weight"):
            with self.subTest(penalty=penalty):
                self.assertLess(getattr(merged, penalty), getattr(base, penalty))

    def test_level_zero_is_the_unchanged_segmentation(self):
        base = xatlas_fields(get_adaptive_chart_options(1000))
        self.assertEqual(base, xatlas_fields(get_adaptive_chart_options(1000, merge_level=0)))


def xatlas_fields(options):
    return {
        field: getattr(options, field)
        for field in (
            "max_cost", "max_iterations", "normal_deviation_weight", "roundness_weight",
            "straightness_weight", "normal_seam_weight", "texture_seam_weight"
        )
    }


class TestPlanReportsIslandMerging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        normalized = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        cls.uv = np.column_stack([
            0.5 + np.arctan2(normalized[:, 0], normalized[:, 2]) / (2 * np.pi),
            0.5 - np.arcsin(normalized[:, 1]) / np.pi
        ])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=cls.uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (256, 256), (120, 160, 90))
            )
        )
        cls.mesh = mesh
        cls.image = mesh.visual.material.baseColorTexture

    def plan(self, merge_islands: bool):
        return plan_uv_canvas(
            self.mesh,
            source_image=self.image,
            source_uv=self.uv,
            size_mode="exact",
            unwrap_method="xatlas",
            merge_islands=merge_islands
        )

    def test_merging_off_charts_a_single_level(self):
        merge = self.plan(merge_islands=False)["island_merge"]
        self.assertFalse(merge["enabled"])
        self.assertEqual(len(merge["levels"]), 1)
        self.assertEqual(merge["chosenLevel"], 0)
        self.assertEqual(merge["islandsBefore"], merge["islandsAfter"])
        self.assertEqual(merge["boundaryReductionPercent"], 0.0)

    def test_merging_on_tries_several_levels_and_keeps_the_smallest_canvas(self):
        plan = self.plan(merge_islands=True)
        merge = plan["island_merge"]
        self.assertTrue(merge["enabled"])
        self.assertGreater(len(merge["levels"]), 1)
        self.assertLessEqual(len(merge["levels"]), ISLAND_MERGE_LEVELS)
        # Whatever it picked must be the best canvas it measured, and the plan must use it
        best = min(level["canvas"] for level in merge["levels"])
        chosen = next(l for l in merge["levels"] if l["level"] == merge["chosenLevel"])
        self.assertEqual(chosen["canvas"], best)
        self.assertEqual(plan["fit_resolution"], best)
        self.assertLessEqual(merge["canvasAfter"], merge["canvasBefore"])

    def test_every_level_reports_what_it_measured(self):
        for level in self.plan(merge_islands=True)["island_merge"]["levels"]:
            with self.subTest(level=level["level"]):
                self.assertGreater(level["islands"], 0)
                self.assertGreater(level["boundaryTexels"], 0.0)
                self.assertGreater(level["canvas"], 0)


class TestIslandMergeInThePipeline(unittest.TestCase):
    def run_pipeline(self, out_dir: Path, source: Path, merge_uv_islands: bool):
        return StepPipeline(
            texture_format="original",
            merge_uv_islands=merge_uv_islands,
            verbose=False,
            stream_events=False,
            skip_steps=[5, 6]
        ).run(source, out_dir)

    def test_step_4_records_the_island_metrics(self):
        with tempfile.TemporaryDirectory(prefix="island_merge_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=4)
            result = self.run_pipeline(tmp_dir / "out", source, merge_uv_islands=True)

            details = next(s for s in result["steps"] if s["step"] == 4)["details"]
            self.assertTrue(details["mergeUvIslands"])
            merge = details["islandMerge"]
            self.assertTrue(merge["enabled"])
            self.assertEqual(merge["method"], "xatlas")
            self.assertGreater(merge["islandsAfter"], 0)
            self.assertGreater(merge["boundaryTexelsAfter"], 0.0)
            self.assertEqual(
                merge["canvasAfter"], min(level["canvas"] for level in merge["levels"])
            )

    def test_switching_it_off_charts_once(self):
        with tempfile.TemporaryDirectory(prefix="island_merge_off_") as tmp:
            tmp_dir = Path(tmp)
            source = tmp_dir / "input.glb"
            create_mock_glb(source, subdivisions=4)
            result = self.run_pipeline(tmp_dir / "out", source, merge_uv_islands=False)

            details = next(s for s in result["steps"] if s["step"] == 4)["details"]
            self.assertFalse(details["mergeUvIslands"])
            self.assertEqual(len(details["islandMerge"]["levels"]), 1)

    def test_the_option_must_be_a_bool(self):
        with self.assertRaises(TypeError):
            StepPipeline(merge_uv_islands="on")


if __name__ == "__main__":
    unittest.main()
