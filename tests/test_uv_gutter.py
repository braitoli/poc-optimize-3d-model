"""
test_uv_gutter.py

Gutter guarantee of rechart_and_bake_high_density at the FINAL texture resolution:
- When Bước B downscales (2048 -> 1024) the atlas must be re-packed at the final
  resolution, so the gap between UV charts stays >= 3 px in final-texture pixels
  (xatlas and Microsoft UVAtlas).
- EDT dilation fills every gutter pixel near a chart (no flat background colour left).
- Without downscale nothing is re-packed and the gap guarantee still holds.
"""

import functools
import unittest

import numpy as np
from PIL import Image
import trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from optimizer.core import uv_baker
from optimizer.core.uv_baker import rechart_and_bake_high_density
from optimizer.core.uvatlas import is_uvatlas_available

MIN_GAP_PX = 3.0
DILATION_PADDING = 16
UVATLAS_OK = is_uvatlas_available()[0]

# Source UVs are squeezed into a small square [U0, U0 + SPAN]^2 (~9% of the canvas is covered)
U0, SPAN = 0.30, 0.40


def _make_case(src_res: int):
    """6x6 grid of small icospheres (11,520 faces, ~200 charts so the packer's minimum padding is
    actually reached) with planar-projected source UVs and a 3-stripe source texture."""
    parts = []
    for i in range(6):
        for j in range(6):
            part = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
            part.apply_translation([i, j, 0.0])
            parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    xy = np.asarray(mesh.vertices)[:, :2]
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    uv = U0 + xy * SPAN
    # Red / green / blue vertical stripes across the UV region. Every source texel and every
    # bilinear blend of neighbouring stripes has at least one zero channel, whereas the flat
    # canvas background (mean sampled surface colour) has all three channels > 0.
    img = np.zeros((src_res, src_res, 3), dtype=np.uint8)
    xs = np.arange(src_res) / float(src_res - 1)
    img[:, xs < U0 + SPAN / 3.0, 0] = 255
    img[:, (xs >= U0 + SPAN / 3.0) & (xs < U0 + 2.0 * SPAN / 3.0), 1] = 255
    img[:, xs >= U0 + 2.0 * SPAN / 3.0, 2] = 255
    return mesh, Image.fromarray(img, mode="RGB"), uv


@functools.lru_cache(maxsize=None)
def _run(unwrap_method: str, src_res: int, target_res: int):
    mesh, img, uv = _make_case(src_res)
    return rechart_and_bake_high_density(
        mesh,
        target_res=target_res,
        source_image=img,
        source_uv=uv,
        dilation_padding=DILATION_PADDING,
        return_stats=True,
        unwrap_method=unwrap_method,
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
    def _check_min_gap(self, unwrap_method, src_res, target_res, expect_downscale):
        mesh, img, stats = _run(unwrap_method, src_res, target_res)
        final_res = 1024
        self.assertEqual(stats["downscaled"], expect_downscale)
        self.assertEqual(img.size, (final_res, final_res))
        self.assertEqual(stats["final_resolution"], final_res)
        self.assertFalse(stats.get("uvatlas_fallback", False))

        gap = _min_chart_gap_px(np.asarray(mesh.faces), np.asarray(mesh.visual.uv), final_res)
        print(f"\n[uv_gutter] {unwrap_method} {src_res}->{final_res}: min chart gap = {gap:.3f} px")
        self.assertGreaterEqual(gap, MIN_GAP_PX)
        self.assertIs(stats.get("repacked_at_final_resolution"), expect_downscale)

    def _check_dilation_fills_gutters(self, unwrap_method, src_res, target_res):
        mesh, img, _ = _run(unwrap_method, src_res, target_res)
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

    # --- Downscale path (2048 -> 1024) -----------------------------------------------------
    def test_xatlas_downscaled_gutter(self):
        self._check_min_gap("xatlas", 2048, 2048, expect_downscale=True)

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_downscaled_gutter(self):
        self._check_min_gap("uvatlas", 2048, 2048, expect_downscale=True)

    def test_xatlas_downscaled_dilation_fills_gutters(self):
        self._check_dilation_fills_gutters("xatlas", 2048, 2048)

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_downscaled_dilation_fills_gutters(self):
        self._check_dilation_fills_gutters("uvatlas", 2048, 2048)

    # --- No-downscale path (1024 source, 1024 target) ----------------------------------------
    def test_xatlas_no_downscale_no_repack(self):
        self._check_min_gap("xatlas", 1024, 1024, expect_downscale=False)

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_no_downscale_no_repack(self):
        self._check_min_gap("uvatlas", 1024, 1024, expect_downscale=False)


if __name__ == "__main__":
    unittest.main()
