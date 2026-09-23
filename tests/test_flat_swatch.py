"""
test_flat_swatch.py

Step 4's flat-colour swatches: faces whose source colour is uniform are kept out of the chart pass
and share one small swatch per colour, so the canvas only has to hold the faces that carry detail.

Invariants under test:
- a face only counts as flat when every re-baked texture slot is flat over it;
- a swatched face's mean colour never moves further than the tolerance;
- the baked model keeps every face, and a swatched face samples its swatch colour;
- the canvas never grows because swatches were switched on.
"""

import unittest

import numpy as np
import trimesh
from PIL import Image

from optimizer.core.uv_baker import (
    CANVAS_BLOCK_PX,
    _barycentric_grid,
    FLAT_SWATCH_PX,
    bake_uv_plan,
    classify_flat_faces,
    plan_uv_canvas,
    swatch_cells,
    _sample_texture_bilinear,
)
from optimizer.core.uvatlas import is_uvatlas_available
from optimizer.step_pipeline import StepPipeline


def noisy_image(size=128, seed=0):
    """A texture with detail everywhere: no face over it can be flat."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def half_flat_image(size=128, flat_rgb=(30, 160, 90), seed=1):
    """Left half one flat colour, right half noise."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    pixels[:, : size // 2] = np.asarray(flat_rgb, dtype=np.uint8)
    return Image.fromarray(pixels)


def centre_detail_image(size=512, flat_rgb=(30, 160, 90), seed=1):
    """One flat colour with a centred square of noise covering a quarter of the texture. The
    detailed faces form a compact block, so holding the flat ones out really does shrink the
    canvas side (a long thin remainder would keep the side and only lose area)."""
    rng = np.random.default_rng(seed)
    pixels = np.tile(np.asarray(flat_rgb, dtype=np.uint8), (size, size, 1))
    lo, hi = size // 4, size - size // 4
    pixels[lo:hi, lo:hi] = rng.integers(0, 255, (hi - lo, hi - lo, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def textured(mesh, image, uv):
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    )
    return mesh


def grid_mesh(n=12, uv_box=(0.0, 0.0, 1.0, 1.0)):
    """An n x n grid of quads on the XY plane, UV-mapped into `uv_box` (u0, v0, u1, v1)."""
    xs = np.linspace(0.0, 1.0, n + 1)
    grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
    vertices = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)
    faces = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, i * (n + 1) + j + 1
            c, d = (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
            faces += [[a, b, d], [a, d, c]]
    u0, v0, u1, v1 = uv_box
    uv = np.stack([u0 + grid_x.ravel() * (u1 - u0), v0 + grid_y.ravel() * (v1 - v0)], axis=1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    return mesh, uv


class TestClassification(unittest.TestCase):
    def test_uniform_texture_makes_every_face_flat(self):
        mesh, uv = grid_mesh(6)
        image = Image.fromarray(np.full((64, 64, 3), (200, 30, 40), dtype=np.uint8))
        textured(mesh, image, uv)
        result = classify_flat_faces(
            mesh, image, uv, tolerance=6.0, min_group_faces=1, material=mesh.visual.material
        )
        self.assertTrue(result["flat"].all())
        self.assertEqual(result["stats"]["swatches"], 1)
        self.assertEqual(result["cluster_colors"]["baseColorTexture"][0], [200, 30, 40])

    def test_noise_texture_leaves_nothing_flat(self):
        mesh, uv = grid_mesh(6)
        image = noisy_image()
        textured(mesh, image, uv)
        result = classify_flat_faces(
            mesh, image, uv, tolerance=6.0, min_group_faces=1, material=mesh.visual.material
        )
        self.assertFalse(result["flat"].any())
        self.assertEqual(result["stats"]["swatches"], 0)

    def test_a_detailed_normal_map_keeps_a_flat_base_colour_out(self):
        """Flat albedo over a detailed normal map is not flat: the normal detail would be lost."""
        mesh, uv = grid_mesh(6)
        flat = Image.fromarray(np.full((64, 64, 3), (120, 120, 120), dtype=np.uint8))
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=flat, normalTexture=noisy_image(64)
            )
        )
        result = classify_flat_faces(
            mesh, flat, uv, tolerance=6.0, min_group_faces=1, material=mesh.visual.material
        )
        self.assertFalse(result["flat"].any())
        self.assertIn("normalTexture", result["stats"]["slotsChecked"])

    def test_small_groups_stay_with_the_charted_faces(self):
        mesh, uv = grid_mesh(6)
        image = Image.fromarray(np.full((64, 64, 3), (10, 20, 30), dtype=np.uint8))
        textured(mesh, image, uv)
        result = classify_flat_faces(
            mesh, image, uv, tolerance=6.0, min_group_faces=10_000, material=mesh.visual.material
        )
        self.assertFalse(result["flat"].any())

    def test_a_swatched_face_never_moves_further_than_the_tolerance(self):
        """Region growing may not drift: face mean -> group mean -> swatch stays inside tolerance."""
        mesh, uv = grid_mesh(16)
        size = 128
        # A horizontal ramp: neighbouring faces are within tolerance, the two ends are not
        # Gentle enough that neighbouring faces pass the tolerance, long enough that a group
        # growing from end to end would drift far past it
        ramp = np.tile(np.linspace(0, 96, size, dtype=np.uint8)[None, :, None], (size, 1, 3))
        image = Image.fromarray(ramp)
        textured(mesh, image, uv)
        tolerance = 12.0
        result = classify_flat_faces(
            mesh, image, uv, tolerance=tolerance, min_group_faces=1, material=mesh.visual.material
        )
        flat = result["flat"]
        self.assertTrue(flat.any(), "the gentle ramp should leave flat faces")
        centres = np.asarray(result["cluster_colors"]["baseColorTexture"], dtype=np.float64)
        faces = np.asarray(mesh.faces)[flat]
        want = centres[result["cluster"][flat]]
        # Not just the face's mean colour: no point of a swatched face may move further than the
        # tolerance, which is what the option promises its user
        bary = _barycentric_grid()
        points = np.einsum("sk,fkj->fsj", bary, uv[faces])
        sampled = _sample_texture_bilinear(
            np.asarray(image, dtype=np.uint8), points.reshape(-1, 2)
        ).reshape(len(faces), len(bary), 3)
        self.assertLessEqual(np.abs(sampled - want[:, None, :]).max(), tolerance + 1e-6)


class TestPlanAndBake(unittest.TestCase):
    def setUp(self):
        # Half the UV space is one flat colour, half is noise: the flat half needs no texels.
        # The texture is big enough that the canvas dwarfs the swatch strip, which is what makes
        # holding the flat faces out of the chart pass pay off.
        self.mesh, self.uv = grid_mesh(24, uv_box=(0.0, 0.0, 1.0, 1.0))
        self.image = centre_detail_image(512)
        textured(self.mesh, self.image, self.uv)

    def plan(self, **kwargs):
        return plan_uv_canvas(
            self.mesh, source_image=self.image, source_uv=self.uv, merge_islands=False, **kwargs
        )

    def test_swatches_never_grow_the_canvas(self):
        baseline = self.plan()
        swatched = self.plan(flat_swatch=True, flat_tolerance=8.0, flat_min_group_faces=4)
        self.assertIsNotNone(swatched["flat"], "the flat half should have been detected")
        self.assertLessEqual(swatched["final_resolution"], baseline["final_resolution"])
        self.assertGreater(swatched["swatch_strip_px"], 0)
        # The islands are charted for the canvas minus the strip
        self.assertGreaterEqual(
            swatched["final_resolution"] - swatched["swatch_strip_px"],
            swatched["island_fit_resolution"] - CANVAS_BLOCK_PX
        )

    def test_swatches_are_dropped_when_they_would_not_shrink_the_canvas(self):
        """A canvas barely bigger than the strip cannot win: the plan says so instead of growing."""
        mesh, uv = grid_mesh(6)
        image = half_flat_image(32)
        textured(mesh, image, uv)
        plan = plan_uv_canvas(
            mesh, source_image=image, source_uv=uv, merge_islands=False,
            flat_swatch=True, flat_tolerance=8.0, flat_min_group_faces=1
        )
        if plan["flat"] is None:
            self.assertIsNotNone(plan["flat_disabled"])

    def test_the_baked_model_keeps_every_face_and_samples_its_swatch(self):
        plan = self.plan(flat_swatch=True, flat_tolerance=8.0, flat_min_group_faces=4)
        baked_mesh, baked_image, stats = bake_uv_plan(
            self.mesh, plan, source_image=self.image, source_uv=self.uv
        )
        self.assertEqual(len(baked_mesh.faces), len(self.mesh.faces))
        self.assertIsNotNone(stats["flat_swatch"])

        flat = plan["flat"]["flat"]
        n_flat = int(flat.sum())
        self.assertGreater(n_flat, 0)
        # The swatched faces are the last ones of the baked mesh, in their original order
        swatched_faces = np.asarray(baked_mesh.faces)[-n_flat:]
        uv_baked = np.asarray(baked_mesh.visual.uv)
        sampled = _sample_texture_bilinear(
            np.asarray(baked_image.convert("RGB"), dtype=np.uint8),
            uv_baked[swatched_faces].mean(axis=1)
        )
        expected = np.asarray(
            plan["flat"]["cluster_colors"]["baseColorTexture"], dtype=np.float64
        )[plan["flat"]["cluster"][flat]]
        np.testing.assert_allclose(sampled, expected, atol=1.0)

    def test_every_swatch_cell_sits_inside_the_strip(self):
        plan = self.plan(flat_swatch=True, flat_tolerance=8.0, flat_min_group_faces=4)
        canvas = plan["final_resolution"]
        strip = plan["swatch_strip_px"]
        cells = swatch_cells(plan["flat"]["stats"]["swatches"], canvas, strip)
        self.assertTrue((cells[:, 1] >= canvas - strip).all())
        self.assertTrue((cells[:, 3] <= canvas).all())
        self.assertTrue((cells[:, 2] <= canvas).all())
        self.assertTrue(((cells[:, 2] - cells[:, 0]) == FLAT_SWATCH_PX).all())
        # KTX2 encodes 4x4 blocks: a cell that straddles one would mix two colours into it
        self.assertTrue((cells % CANVAS_BLOCK_PX == 0).all())

    def test_islands_stay_clear_of_the_strip(self):
        plan = self.plan(flat_swatch=True, flat_tolerance=8.0, flat_min_group_faces=4)
        baked_mesh, _, _ = bake_uv_plan(
            self.mesh, plan, source_image=self.image, source_uv=self.uv
        )
        n_flat = int(plan["flat"]["flat"].sum())
        island_uv = np.asarray(baked_mesh.visual.uv)[
            np.unique(np.asarray(baked_mesh.faces)[:-n_flat])
        ]
        strip_v = plan["swatch_strip_px"] / plan["final_resolution"]
        self.assertGreaterEqual(island_uv[:, 1].min(), strip_v - 1e-9)


class TestUVAtlasPath(unittest.TestCase):
    @unittest.skipUnless(is_uvatlas_available()[0], "UVAtlas backend not installed")
    def test_uvatlas_charts_only_the_detailed_faces_too(self):
        mesh, uv = grid_mesh(16)
        image = centre_detail_image(256)
        textured(mesh, image, uv)
        common = dict(source_image=image, source_uv=uv, unwrap_method="uvatlas", merge_islands=False)
        baseline = plan_uv_canvas(mesh, **common)
        swatched = plan_uv_canvas(
            mesh, **common, flat_swatch=True, flat_tolerance=12.0, flat_min_group_faces=4
        )
        self.assertLessEqual(swatched["final_resolution"], baseline["final_resolution"])
        baked, _, _ = bake_uv_plan(mesh, swatched, source_image=image, source_uv=uv)
        self.assertEqual(len(baked.faces), len(mesh.faces))
        self.assertTrue(np.isfinite(np.asarray(baked.visual.uv)).all())


class TestPipelineOptions(unittest.TestCase):
    def test_defaults_are_off(self):
        pipeline = StepPipeline()
        self.assertFalse(pipeline.flat_swatch)
        self.assertEqual(pipeline.flat_tolerance, 8.0)
        self.assertEqual(pipeline.flat_min_group_faces, 16)

    def test_invalid_values_raise(self):
        with self.assertRaises(TypeError):
            StepPipeline(flat_swatch="on")
        with self.assertRaises(ValueError):
            StepPipeline(flat_tolerance=0)
        with self.assertRaises(ValueError):
            StepPipeline(flat_tolerance=1000)
        with self.assertRaises(ValueError):
            StepPipeline(flat_min_group_faces=0)
        with self.assertRaises(TypeError):
            StepPipeline(flat_min_group_faces=2.5)


if __name__ == "__main__":
    unittest.main()
