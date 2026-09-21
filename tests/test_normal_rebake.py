"""
test_normal_rebake.py

rechart_and_bake_high_density re-charts the UVs, so every texture slot of the material must be
re-baked into the NEW layout, not only baseColorTexture:
- normalTexture (tangent space): decoded with the OLD per-face tangent frame, re-encoded with the
  NEW one, so the object-space normal the viewer reconstructs at a surface point is unchanged.
  Per-face frame (three.js without TANGENT attribute): T = dp/du, B = dp/dv in trimesh's v-up UV
  space, Gram-Schmidt against the interpolated vertex normal N, B = cross(N, T) with B's handedness.
- A flat normal map stays flat; other slots (metallicRoughness) are resampled like base colour.
- The exported GLB carries the NEW normal map, never the original passthrough bytes.
"""

import functools
import io
import unittest

import numpy as np
from PIL import Image
import trimesh
import xatlas

from optimizer.core import uv_baker
from optimizer.core.texture_utils import preserve_mesh_textures
from optimizer.core.uv_baker import rechart_and_bake_high_density
from optimizer.core.uvatlas import is_uvatlas_available

UVATLAS_OK = is_uvatlas_available()[0]
SRC_RES = 1024
DILATION_PADDING = 16
TILT_DEG = 25.0
TILT_DIR = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
FLAT = np.array([128, 128, 255])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _n_gt(n: np.ndarray) -> np.ndarray:
    """Ground-truth object-space normal field: the surface normal tilted (up to 25 deg) toward
    a fixed world direction."""
    t = TILT_DIR[None, :] - n * (n @ TILT_DIR)[:, None]
    return _unit(n + np.tan(np.radians(TILT_DEG)) * t)


def _frames(p_tri: np.ndarray, uv_tri: np.ndarray, n: np.ndarray):
    """Per-face tangent frame (T, B) orthonormalized against n; valid=False for degenerate UVs."""
    dp1, dp2 = p_tri[:, 1] - p_tri[:, 0], p_tri[:, 2] - p_tri[:, 0]
    d1, d2 = uv_tri[:, 1] - uv_tri[:, 0], uv_tri[:, 2] - uv_tri[:, 0]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    valid = np.abs(r) > 1e-9 * np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
    r = np.where(valid, r, 1.0)[:, None]
    t = (dp1 * d2[:, 1:2] - dp2 * d1[:, 1:2]) / r
    b = (dp2 * d1[:, 0:1] - dp1 * d2[:, 0:1]) / r
    t = _unit(t - n * np.sum(n * t, axis=1, keepdims=True))
    bh = np.cross(n, t)
    bh *= np.where(np.sum(bh * b, axis=1) < 0.0, -1.0, 1.0)[:, None]
    return t, bh, valid


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, C) image at v-up UVs (same texel mapping as the baker)."""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    im = img.astype(np.float64)
    top = im[y0, x0] * (1.0 - fx) + im[y0, x1] * fx
    bot = im[y1, x0] * (1.0 - fx) + im[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def _build_mesh(verts, faces, normals) -> trimesh.Trimesh:
    # Fresh mesh per use: Trimesh.copy() drops the explicit (cached) vertex normals
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)


@functools.lru_cache(maxsize=None)
def _make_case():
    """4 icospheres (5,120 faces), analytic vertex normals, seam-split vertices of an xatlas source
    layout that is then mirrored + rotated 90 deg, (u, v) -> (1 - v, 1 - u): every old tangent frame
    differs from the new layout's (charts rotated, handedness flipped).
    Returns (vertices, faces, vertex_normals, source_uv, normal map encoding _n_gt in the OLD
    layout, stripe texture)."""
    verts, faces, normals = [], [], []
    for c in [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (1.5, 1.5, 0.3)]:
        part = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        faces.append(np.asarray(part.faces) + sum(len(v) for v in verts))
        verts.append(np.asarray(part.vertices) + np.asarray(c))
        normals.append(_unit(np.asarray(part.vertices)))
    verts, faces, normals = np.vstack(verts), np.vstack(faces), np.vstack(normals)

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32))
    p_opts = xatlas.PackOptions()
    p_opts.resolution = SRC_RES
    p_opts.padding = 4
    p_opts.bilinear = True
    atlas.generate(pack_options=p_opts)
    vmap, src_faces, uv = atlas[0]
    vmap = np.asarray(vmap, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    source_uv = np.column_stack([1.0 - uv[:, 1], 1.0 - uv[:, 0]])
    verts, faces, normals = verts[vmap], np.asarray(src_faces, dtype=np.int64), normals[vmap]
    mesh = _build_mesh(verts, faces, normals)

    # Encode _n_gt as a tangent-space map in the OLD layout (per-texel interpolated N, per-face frame)
    f = np.asarray(mesh.faces)
    sel, fid, bary = uv_baker._rasterize_uv_atlas(f, source_uv, SRC_RES)
    n = _unit(np.einsum("pk,pkj->pj", bary, np.asarray(mesh.vertex_normals)[f[fid]]))
    t, b, valid = _frames(np.asarray(mesh.vertices)[f[fid]], source_uv[f[fid]], n)
    assert valid.all()
    g = _n_gt(n)
    n_ts = np.stack([np.sum(g * t, 1), np.sum(g * b, 1), np.sum(g * n, 1)], axis=1)
    flat = np.tile(FLAT.astype(np.uint8), (SRC_RES * SRC_RES, 1))
    flat[sel] = np.clip(np.round((n_ts * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    covered = np.zeros(SRC_RES * SRC_RES, dtype=bool)
    covered[sel] = True
    nmap = uv_baker.dilate_texture(
        flat.reshape(SRC_RES, SRC_RES, 3), covered.reshape(SRC_RES, SRC_RES), padding=DILATION_PADDING
    )

    stripes = np.zeros((SRC_RES, SRC_RES, 3), dtype=np.uint8)
    xs = np.arange(SRC_RES)
    stripes[:, xs < SRC_RES // 3, 0] = 255
    stripes[:, (xs >= SRC_RES // 3) & (xs < 2 * SRC_RES // 3), 1] = 200
    stripes[:, xs >= 2 * SRC_RES // 3, 2] = 150
    stripes[(np.arange(SRC_RES) // 64) % 2 == 1] //= 2
    return verts, faces, normals, source_uv, nmap, stripes


@functools.lru_cache(maxsize=None)
def _run(unwrap_method: str, flat_normal: bool):
    verts, faces, normals, source_uv, nmap, stripes = _make_case()
    mesh = _build_mesh(verts, faces, normals)
    base = Image.fromarray(stripes, mode="RGB")
    normal = Image.new("RGB", (SRC_RES, SRC_RES), tuple(int(c) for c in FLAT)) if flat_normal \
        else Image.fromarray(nmap, mode="RGB")
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base, normalTexture=normal,
        metallicRoughnessTexture=Image.fromarray(stripes, mode="RGB"),
        metallicFactor=0.0, roughnessFactor=0.7
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=source_uv, material=mat)
    # Like Step 1/2: textures carry the original encoded bytes for bit-for-bit passthrough export
    slots = {}
    for slot in ("baseColorTexture", "normalTexture", "metallicRoughnessTexture"):
        buf = io.BytesIO()
        getattr(mat, slot).save(buf, format="PNG")
        slots[slot] = {"format": "PNG", "raw_bytes": buf.getvalue()}
    preserve_mesh_textures(mesh, {"slots": slots, "default_format": "PNG"})

    out_mesh, baked, stats = rechart_and_bake_high_density(
        mesh, source_image=base, source_uv=source_uv, dilation_padding=DILATION_PADDING,
        return_stats=True, unwrap_method=unwrap_method
    )
    return mesh, out_mesh, baked, stats, slots


def _angular_errors_deg(out_mesh) -> np.ndarray:
    """Decode the baked normal map at NEW face centroids with the NEW per-face frames and compare
    the object-space result to the ground-truth field."""
    f = np.asarray(out_mesh.faces)
    uv = np.asarray(out_mesh.visual.uv)
    nimg = np.asarray(out_mesh.visual.material.normalTexture.convert("RGB"))
    n = _unit(np.asarray(out_mesh.vertex_normals)[f].mean(axis=1))
    t, b, valid = _frames(np.asarray(out_mesh.vertices)[f], uv[f], n)
    n_ts = _bilinear(nimg, uv[f].mean(axis=1)) / 255.0 * 2.0 - 1.0
    n_os = _unit(n_ts[:, 0:1] * t + n_ts[:, 1:2] * b + n_ts[:, 2:3] * n)
    cos = np.clip(np.sum(n_os * _n_gt(n), axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos[valid]))


class TestNormalMapRebake(unittest.TestCase):
    def _check_rebake(self, unwrap_method):
        mesh, out_mesh, baked, stats, _ = _run(unwrap_method, False)
        err = _angular_errors_deg(out_mesh)
        print(f"\n[normal_rebake] {unwrap_method}: {len(err)} faces, mean {err.mean():.3f} deg, "
              f"p95 {np.percentile(err, 95):.3f} deg, max {err.max():.3f} deg, "
              f"degenerate px {stats.get('normal_rebake_degenerate_pixels')}")
        self.assertLess(float(err.mean()), 1.0)  # measured ~0.19 deg (before the fix ~25 deg)
        self.assertLess(float(np.percentile(err, 95)), 2.0)  # measured ~0.36 deg (before ~44 deg)

        out_normal = out_mesh.visual.material.normalTexture
        self.assertIn("normalTexture", stats["rebaked_texture_slots"])
        self.assertIn("baseColorTexture", stats["rebaked_texture_slots"])
        self.assertIsNot(out_normal, mesh.visual.material.normalTexture)
        self.assertEqual(out_normal.size, baked.size)

    def test_xatlas_normal_rebake(self):
        self._check_rebake("xatlas")

    @unittest.skipUnless(UVATLAS_OK, "Microsoft UVAtlas backend not available")
    def test_uvatlas_normal_rebake(self):
        self._check_rebake("uvatlas")

    def test_flat_normal_map_stays_flat(self):
        _, out_mesh, baked, stats, _ = _run("xatlas", True)
        nimg = np.asarray(out_mesh.visual.material.normalTexture.convert("RGB")).astype(np.int64)
        res = nimg.shape[0]
        sel, _, _ = uv_baker._rasterize_uv_atlas(np.asarray(out_mesh.faces), np.asarray(out_mesh.visual.uv), res)
        inside = nimg.reshape(-1, 3)[sel]
        dev = int(np.abs(inside - FLAT[None, :]).max())
        print(f"\n[normal_rebake] flat map: max deviation inside charts {dev} over {len(sel)} texels")
        self.assertLessEqual(dev, 2)

    def test_metallic_roughness_resampled_like_base_color(self):
        _, out_mesh, baked, stats, _ = _run("xatlas", True)
        self.assertIn("metallicRoughnessTexture", stats["rebaked_texture_slots"])
        mr = np.asarray(out_mesh.visual.material.metallicRoughnessTexture.convert("RGB"))
        self.assertTrue(np.array_equal(mr, np.asarray(baked.convert("RGB"))))

    def test_exported_glb_carries_new_normal_map(self):
        _, out_mesh, _, _, slots = _run("xatlas", False)
        glb = trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": out_mesh}), include_normals=True)
        loaded = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb", force="mesh", process=False)
        exported = np.asarray(loaded.visual.material.normalTexture.convert("RGB"))
        expected = np.asarray(out_mesh.visual.material.normalTexture.convert("RGB"))
        old_map = _make_case()[4]
        self.assertTrue(np.array_equal(exported, expected))
        self.assertFalse(exported.shape == old_map.shape and np.array_equal(exported, old_map))
        self.assertFalse(slots["normalTexture"]["raw_bytes"] in glb)


if __name__ == "__main__":
    unittest.main()
