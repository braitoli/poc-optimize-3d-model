"""
test_uv_gutter.py

Gutter guarantee of rechart_and_bake_high_density at the FINAL texture resolution:
- Every size mode packs the atlas at its final canvas, so the gap between UV charts stays
  >= 3 px in final-texture pixels (xatlas and Microsoft UVAtlas). pot-down is the case where
  islands shrink below 1:1: the charts must be re-packed at the smaller canvas instead of
  scaling a larger layout down (which would shrink the gutters with it).
- EDT dilation fills every gutter pixel near a chart (no flat background colour left).
"""

import functools
import unittest

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from optimizer.core import uv_baker
from optimizer.core.uv_baker import rechart_and_bake_high_density
from optimizer.core.uvatlas import is_uvatlas_available
from tests.fixtures import make_sphere_grid_case

MIN_GAP_PX = 3.0
DILATION_PADDING = 16
UVATLAS_OK = is_uvatlas_available()[0]
SRC_RES = 2048


@functools.lru_cache(maxsize=None)
def _run(unwrap_method: str, size_mode: str):
    mesh, img, uv = make_sphere_grid_case(SRC_RES)
    return rechart_and_bake_high_density(
        mesh,
        source_image=img,
        source_uv=uv,
        size_mode=size_mode,
        unwrap_method=unwrap_method,
        dilation_padding=DILATION_PADDING,
        return_stats=True,
    )


def _chart_labels(faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Chart id per face: connected components of faces sharing a welded UV vertex."""
    _, weld = np.unique(np.round(uv * 1e6).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    n_f, n_v = len(faces), int(weld.max()) + 1
    rows = np.repeat(np.arange(n_f), 3)
    cols = n_f + weld[faces].reshape(-1)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_f + n_v, n_f + n_v))
    _, labels = connected_components(graph, directed=False)
    return labels[:n_f]


def _coverage(faces: np.ndarray, uv: np.ndarray, res: int):
    """Returns (chart-label image with -1 for empty pixels, EDT distance to nearest covered pixel,
    chart label of the nearest covered pixel)."""
    sel, fid, _ = uv_baker._rasterize_uv_atlas(faces, uv, res)
    label_img = np.full(res * res, -1, dtype=np.int64)
    label_img[sel] = _chart_labels(faces, uv)[fid]
    label_img = label_img.reshape(res, res)
    dist, ind = ndimage.distance_transform_edt(label_img < 0, return_indices=True)
    nearest = label_img[ind[0], ind[1]]
    return label_img, dist, nearest


def _min_chart_gap_px(faces: np.ndarray, uv: np.ndarray, res: int) -> float:
    """Minimum gap (final-texture pixels) between distinct charts: over all 4-neighbour pixel
    pairs whose nearest-chart labels differ, the minimum of dist(p) + dist(q)."""
    _, dist, nearest = _coverage(faces, uv, res)
    gaps = [
        (dist[:, :-1] + dist[:, 1:])[nearest[:, :-1] != nearest[:, 1:]],
        (dist[:-1, :] + dist[1:, :])[nearest[:-1, :] != nearest[1:, :]],
    ]
    gaps = np.concatenate(gaps)
    return float(gaps.min()) if len(gaps) else float("inf")


class TestUVGutterAtFinalResolution(unittest.TestCase):
    def _check_min_gap(self, unwrap_method, size_mode):
        mesh, img, stats = _run(unwrap_method, size_mode)
        final_res = stats["final_resolution"]
        self.assertEqual(img.size, (final_res, final_res))

        gap = _min_chart_gap_px(np.asarray(mesh.faces), np.asarray(mesh.visual.uv), final_res)
        print(f"\n[uv_gutter] {unwrap_method} {size_mode} {final_res}px (fit {stats['fit_resolution']}): "
              f"min chart gap = {gap:.3f} px")
        self.assertGreaterEqual(gap, MIN_GAP_PX)

    def _check_dilation_fills_gutters(self, unwrap_method, size_mode):
        mesh, img, _ = _run(unwrap_method, size_mode)
        res = img.size[0]
        label_img, dist, _ = _coverage(np.asarray(mesh.faces), np.asarray(mesh.visual.uv), res)
        rgb = np.asarray(img.convert("RGB"))
        background_like = np.all(rgb > 0, axis=-1)
        # Premise: pixels beyond the dilation radius keep the flat background colour
        far = dist > DILATION_PADDING
        if np.any(far):
            self.assertTrue(np.all(background_like[far]))
        near_gutter = (label_img < 0) & (dist <= 8)
        self.assertTrue(np.any(near_gutter))
        self.assertEqual(int(np.count_nonzero(near_gutter & background_like)), 0)

    # --- exact (non power-of-two canvas at 1:1) --------------------------------------------
    def test_xatlas_exact_gutter(self):
        self._check_min_gap("xatlas", "exact")

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_exact_gutter(self):
        self._check_min_gap("uvatlas", "exact")

    # --- pot-down (islands shrink below 1:1, re-packed at the smaller canvas) ---------------
    def test_xatlas_pot_down_gutter(self):
        self._check_min_gap("xatlas", "pot-down")

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_pot_down_gutter(self):
        self._check_min_gap("uvatlas", "pot-down")

    def test_xatlas_pot_down_dilation_fills_gutters(self):
        self._check_dilation_fills_gutters("xatlas", "pot-down")

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_pot_down_dilation_fills_gutters(self):
        self._check_dilation_fills_gutters("uvatlas", "pot-down")

    # --- pot-up (islands at 1:1 or better on the larger power-of-two canvas) ----------------
    def test_xatlas_pot_up_gutter(self):
        self._check_min_gap("xatlas", "pot-up")

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_pot_up_gutter(self):
        self._check_min_gap("uvatlas", "pot-up")


if __name__ == "__main__":
    unittest.main()
