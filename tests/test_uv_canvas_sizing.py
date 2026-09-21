"""
test_uv_canvas_sizing.py

Canvas sizing of the Step 3 re-chart (downscale on) at the source's 1:1 average texel density:
- T = sqrt(sum_f uvArea_f * W * H / sum_f area3D_f), summed per face (overlapping or mirrored
  source UVs count once per use).
- exact:    smallest square S x S holding the packed islands at 1:1, S % 4 == 0, T_new/T_src >= 0.97.
- pot-up:   smallest power of two >= the exact fit, T_new/T_src >= 0.97.
- pot-down: largest power of two <= the exact fit (islands scaled down, lossy).
"""

import functools
import unittest

import numpy as np
import trimesh

from optimizer.core.uv_baker import (
    SIZE_MODES,
    bake_uv_plan,
    plan_uv_canvas,
    rechart_and_bake_high_density,
    texel_density,
)
from optimizer.core.uvatlas import is_uvatlas_available
from tests.fixtures import make_sphere_grid_case

UVATLAS_OK = is_uvatlas_available()[0]
SRC_RES = 2048
MIN_RATIO = 0.97


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@functools.lru_cache(maxsize=None)
def _run(unwrap_method: str, size_mode: str):
    mesh, img, uv = make_sphere_grid_case(SRC_RES)
    return rechart_and_bake_high_density(
        mesh,
        source_image=img,
        source_uv=uv,
        size_mode=size_mode,
        unwrap_method=unwrap_method,
        return_stats=True,
    )


class TestTexelDensity(unittest.TestCase):
    def test_single_quad(self):
        # 2 x 2 world-unit quad mapped onto a quarter of a 512 x 256 texture:
        # uv area 0.25 -> 0.25 * 512 * 256 = 32768 px over 4 units^2 -> sqrt(8192) texels/unit
        verts = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=np.float64)
        self.assertAlmostEqual(texel_density(verts, faces, uv, 512, 256), np.sqrt(8192.0), places=6)

    def test_mirrored_uvs_count_once_per_face(self):
        # Two quads sharing the same UV square (mirrored UVs): uv area is summed per face,
        # so the density equals that of a single quad, not half of it.
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        uv = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64), (2, 1))
        self.assertAlmostEqual(texel_density(verts, faces, uv, 100, 100), 100.0, places=6)

    def test_zero_uv_area_raises(self):
        mesh = trimesh.creation.box()
        img = make_sphere_grid_case(64)[1]
        with self.assertRaises(ValueError):
            plan_uv_canvas(mesh, img, np.zeros((len(mesh.vertices), 2)), size_mode="exact")

    def test_unknown_size_mode_raises(self):
        mesh, img, uv = make_sphere_grid_case(256)
        with self.assertRaises(ValueError):
            plan_uv_canvas(mesh, img, uv, size_mode="bogus")
        with self.assertRaises(ValueError):
            plan_uv_canvas(mesh, img, uv, size_mode="exact", unwrap_method="bogus")
        self.assertEqual(SIZE_MODES, ("exact", "pot-up", "pot-down"))


class _SizingChecks:
    UNWRAP = None

    def test_exact(self):
        _, img, stats = _run(self.UNWRAP, "exact")
        fit = stats["fit_resolution"]
        print(f"\n[sizing] {self.UNWRAP} exact: fit={fit} ratio={stats['texel_density_ratio']:.4f}")
        self.assertEqual(fit % 4, 0)
        self.assertEqual(stats["final_resolution"], fit)
        self.assertEqual(img.size, (fit, fit))
        self.assertGreaterEqual(stats["texel_density_ratio"], MIN_RATIO)

    def test_pot_up(self):
        _, img, stats = _run(self.UNWRAP, "pot-up")
        fit, final = stats["fit_resolution"], stats["final_resolution"]
        print(f"\n[sizing] {self.UNWRAP} pot-up: fit={fit} final={final} ratio={stats['texel_density_ratio']:.4f}")
        self.assertTrue(_is_pow2(final))
        self.assertGreaterEqual(final, fit)
        self.assertLess(final, 2 * fit)
        self.assertEqual(img.size, (final, final))
        self.assertGreaterEqual(stats["texel_density_ratio"], MIN_RATIO)

    def test_pot_down(self):
        _, img, stats = _run(self.UNWRAP, "pot-down")
        fit, final = stats["fit_resolution"], stats["final_resolution"]
        print(f"\n[sizing] {self.UNWRAP} pot-down: fit={fit} final={final} ratio={stats['texel_density_ratio']:.4f}")
        self.assertTrue(_is_pow2(final))
        self.assertLessEqual(final, fit)
        self.assertGreater(2 * final, fit)
        self.assertEqual(img.size, (final, final))
        # Islands are scaled down to fit, but not by more than the canvas shrink requires
        self.assertLess(stats["texel_density_ratio"], 1.0)
        self.assertGreaterEqual(stats["texel_density_ratio"], 0.9 * final / fit)

    def test_bake_uses_plan_canvas(self):
        # (UVAtlas is not deterministic between runs, so bake the very plan that was sized)
        mesh, img, uv = make_sphere_grid_case(SRC_RES)
        plan = plan_uv_canvas(mesh, img, uv, size_mode="pot-down", unwrap_method=self.UNWRAP)
        self.assertGreater(plan["texel_density_source"], 0.0)
        _, baked, stats = bake_uv_plan(mesh, plan, img, uv)
        self.assertEqual(stats["fit_resolution"], plan["fit_resolution"])
        self.assertEqual(stats["final_resolution"], plan["final_resolution"])
        self.assertEqual(baked.size, (plan["final_resolution"], plan["final_resolution"]))

    def test_faces_preserved(self):
        mesh, _, _ = make_sphere_grid_case(SRC_RES)
        for mode in SIZE_MODES:
            out_mesh, _, _ = _run(self.UNWRAP, mode)
            self.assertEqual(len(out_mesh.faces), len(mesh.faces))


class TestXatlasSizing(_SizingChecks, unittest.TestCase):
    UNWRAP = "xatlas"


@unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
class TestUVAtlasSizing(_SizingChecks, unittest.TestCase):
    UNWRAP = "uvatlas"


if __name__ == "__main__":
    unittest.main()
