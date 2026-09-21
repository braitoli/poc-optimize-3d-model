"""
test_no_fallbacks.py

No silent fallbacks: anything missing, invalid or failing raises (PipelineAbort in Python, a thrown
Error / non-zero exit in the Node scripts) with a specific reason instead of substituting other
behaviour. Perf-only fallbacks with identical output stay, but are logged.

- optimizer/node/optimize_meshopt.mjs: strict CLI parsing, basisu and sharp required, real texture
  size (aspect-preserving, never upscaling resize), empty images rejected.
- optimizer/inspect_metrics.mjs: unreadable texture size.
- optimizer/core/uvatlas.py: Open3D is the only backend, no retry, no UV clamping.
- optimizer/core/uv_baker.py: NaN samples, non-PBR source material, invalid source UVs.
- optimizer/core/texture_utils.py: unreadable / multi-mesh / untextured inputs, unknown formats,
  missing encoded bytes, impossible export formats.
- optimizer/core/cleaner.py, palette.py, shell_orient.py.
"""

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import trimesh

from optimizer.core import shell_orient, uv_baker, uvatlas
from optimizer.core.cleaner import clean_and_repair_mesh
from optimizer.core.errors import PipelineAbort
from optimizer.core.palette import embed_gltf_extras, extract_palette
from optimizer.core.texture_utils import (
    extract_original_texture_info,
    optimize_mesh_texture_for_export,
    preserve_mesh_textures,
)
from tests.fixtures import build_glb, image_bytes, quad, textured_material

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO_ROOT / "optimizer" / "node" / "optimize_meshopt.mjs"
INSPECT_SCRIPT = REPO_ROOT / "optimizer" / "inspect_metrics.mjs"
HAS_BASISU = shutil.which("basisu") is not None

# Prints [{mime, size}] for every texture of a GLB (size read from the encoded image, KTX2 included).
DUMP_TEXTURES_JS = r"""
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(process.argv[1]);
console.log(JSON.stringify(doc.getRoot().listTextures().map(t => ({ mime: t.getMimeType(), size: t.getSize() }))));
"""

# Calls compressTexturesKtx2 / compressTexturesWebp in process on a texture with zero bytes.
EMPTY_IMAGE_JS = r"""
import { Document } from '@gltf-transform/core';
// argv[1] = mode, argv[2] = module URL (a URL in argv[1] would look like a direct run of the script)
const { compressTexturesKtx2, compressTexturesWebp } = await import(process.argv[2]);
const doc = new Document();
doc.createTexture('empty').setImage(new Uint8Array(0)).setMimeType('image/png');
try {
  await (process.argv[1] === 'ktx2' ? compressTexturesKtx2(doc) : compressTexturesWebp(doc));
  console.log('NO_ERROR');
} catch (err) {
  console.log('ERROR ' + err.message);
}
"""

# Makes optimize_meshopt.mjs's own `import('sharp')` fail, as if the package were missing.
# (@gltf-transform/functions -> ndarray-pixels also imports sharp statically; that import is left
# alone so the script loads and its own handling of a missing sharp is what gets tested.)
BLOCK_SHARP_JS = r"""
import { registerHooks } from 'node:module';
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'sharp' && (context.parentURL || '').endsWith('/optimize_meshopt.mjs')) {
      throw new Error("Cannot find package 'sharp' (blocked by test)");
    }
    return nextResolve(specifier, context);
  }
});
"""


def textured_quad_glb(path: Path, data: bytes, mime: str) -> Path:
    build_glb(path, meshes=[[quad(material=0)]], images=[(data, mime)], materials=[textured_material()])
    return path


def run_meshopt(*args: str, env=None, node_args=()) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", *node_args, str(NODE_SCRIPT), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env
    )


def summary_of(proc: subprocess.CompletedProcess) -> dict:
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"no JSON summary in stdout: {proc.stdout!r} / stderr: {proc.stderr!r}")


def texture_sizes(path: Path) -> list:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", DUMP_TEXTURES_JS, str(path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


class _TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="test_no_fallbacks_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------------------------
# optimize_meshopt.mjs
# ---------------------------------------------------------------------------------------------
class TestMeshoptCliArguments(_TmpDirCase):
    def setUp(self):
        super().setUp()
        self.src = textured_quad_glb(self.tmp / "in.glb", image_bytes(), "image/png")
        self.dst = self.tmp / "out.glb"

    def assert_rejected(self, *args: str, mention: str):
        proc = run_meshopt(str(self.src), str(self.dst), "--no-ktx2", "--json", *args)
        self.assertNotEqual(proc.returncode, 0, f"{args} must be rejected: {proc.stdout}")
        self.assertIn(mention, proc.stderr)
        self.assertFalse(self.dst.exists(), "nothing may be written for rejected arguments")

    def test_unknown_flag(self):
        self.assert_rejected("--bogus-flag", mention="--bogus-flag")

    def test_unexpected_positional_argument(self):
        self.assert_rejected("third.glb", mention="third.glb")

    def test_flag_missing_its_value(self):
        self.assert_rejected("--pos-bits", mention="--pos-bits")
        # the next flag is not a value
        self.assert_rejected("--texture-max-dim", "--reorder", mention="--texture-max-dim")

    def test_invalid_numbers(self):
        cases = [
            ("--pos-bits", "abc"), ("--pos-bits", "0"), ("--pos-bits", "17"), ("--pos-bits", "12.5"),
            ("--weld", "-1"), ("--weld", "nan"), ("--weld", ""),
            ("--ktx2-level", "5"), ("--ktx2-rdo", "-0.5"), ("--ktx2-rdo-d", "0"), ("--ktx2-rdo-d", "70000"),
            ("--ktx2-threads", "0"),
            ("--texture-max-dim", "0"), ("--texture-max-dim", "12px"), ("--ktx2-max-dim", "-4"),
            ("--webp-quality", "0"), ("--webp-quality", "101"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value):
                self.assert_rejected(flag, value, mention=flag)

    def test_invalid_ktx2_mode(self):
        self.assert_rejected("--ktx2-mode", "astc", mention="--ktx2-mode")

    def test_decimation_requests_are_still_ignored(self):
        proc = run_meshopt(str(self.src), str(self.dst), "--no-ktx2", "--json", "--ratio", "0.5", "--target-faces=1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = summary_of(proc)
        self.assertEqual(summary["trianglesAfter"], summary["trianglesBefore"])


class TestMeshoptRequiredTools(_TmpDirCase):
    def setUp(self):
        super().setUp()
        self.src = textured_quad_glb(self.tmp / "in.glb", image_bytes(), "image/png")
        self.dst = self.tmp / "out.glb"

    def test_missing_basisu_fails(self):
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "node").symlink_to(shutil.which("node"))
        env = {**os.environ, "PATH": str(bin_dir)}
        proc = run_meshopt(str(self.src), str(self.dst), "--textures-only", "--ktx2", "--json", env=env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("basisu", proc.stderr)
        self.assertFalse(self.dst.exists())

    def _run_without_sharp(self, *flags: str) -> subprocess.CompletedProcess:
        hook = self.tmp / "block_sharp.mjs"
        hook.write_text(BLOCK_SHARP_JS)
        return run_meshopt(str(self.src), str(self.dst), "--textures-only", "--json", *flags,
                           node_args=("--import", hook.as_uri()))

    def test_missing_sharp_fails_webp(self):
        proc = self._run_without_sharp("--webp")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("sharp", proc.stderr)
        self.assertFalse(self.dst.exists())

    @unittest.skipUnless(HAS_BASISU, "basisu CLI not on PATH")
    def test_missing_sharp_fails_ktx2(self):
        proc = self._run_without_sharp("--ktx2")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("sharp", proc.stderr)
        self.assertFalse(self.dst.exists())


class TestMeshoptTextureInputs(_TmpDirCase):
    def _empty_image(self, mode: str) -> str:
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", EMPTY_IMAGE_JS, mode, NODE_SCRIPT.as_uri()],
            cwd=str(REPO_ROOT), capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip().splitlines()[-1]

    def test_zero_byte_image_ktx2(self):
        out = self._empty_image("ktx2")
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertIn("empty", out)

    def test_zero_byte_image_webp(self):
        out = self._empty_image("webp")
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertIn("empty", out)

    @unittest.skipUnless(HAS_BASISU, "basisu CLI not on PATH")
    def test_undecodable_image_fails(self):
        src = textured_quad_glb(self.tmp / "in.glb", b"definitely not an image" * 4, "image/png")
        proc = run_meshopt(str(src), str(self.tmp / "out.glb"), "--textures-only", "--ktx2", "--json")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("texture 0", proc.stderr)


@unittest.skipUnless(HAS_BASISU, "basisu CLI not on PATH")
class TestMeshoptTextureResize(_TmpDirCase):
    def _encode(self, data: bytes, mime: str, *flags: str) -> list:
        src = textured_quad_glb(self.tmp / "in.glb", data, mime)
        dst = self.tmp / "out.glb"
        proc = run_meshopt(str(src), str(dst), "--textures-only", "--json", *flags)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return texture_sizes(dst)

    def test_ktx2_downscale_keeps_aspect_ratio(self):
        textures = self._encode(image_bytes((64, 32)), "image/png", "--ktx2", "--texture-max-dim", "32")
        self.assertEqual(textures, [{"mime": "image/ktx2", "size": [32, 16]}])

    def test_ktx2_never_upscales_webp_input(self):
        # The WebP size used to be unknown, which forced a square resample to the max dimension
        textures = self._encode(image_bytes((64, 32), fmt="WEBP"), "image/webp", "--ktx2", "--texture-max-dim", "128")
        self.assertEqual(textures, [{"mime": "image/ktx2", "size": [64, 32]}])

    def test_ktx2_smaller_than_max_dim_is_untouched(self):
        textures = self._encode(image_bytes((16, 8)), "image/png", "--ktx2", "--texture-max-dim", "32")
        self.assertEqual(textures, [{"mime": "image/ktx2", "size": [16, 8]}])

    def test_webp_downscale_keeps_aspect_ratio(self):
        textures = self._encode(image_bytes((64, 32)), "image/png", "--webp", "--texture-max-dim", "32")
        self.assertEqual(textures, [{"mime": "image/webp", "size": [32, 16]}])

    def test_already_ktx2_is_skipped(self):
        src = textured_quad_glb(self.tmp / "in.glb", image_bytes(), "image/png")
        once, twice = self.tmp / "once.glb", self.tmp / "twice.glb"
        first = run_meshopt(str(src), str(once), "--textures-only", "--ktx2", "--json")
        self.assertEqual(first.returncode, 0, first.stderr)
        second = run_meshopt(str(once), str(twice), "--textures-only", "--ktx2", "--json")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(summary_of(second)["ktx2"]["count"], 0)
        self.assertEqual(texture_sizes(twice), [{"mime": "image/ktx2", "size": [64, 32]}])


# ---------------------------------------------------------------------------------------------
# inspect_metrics.mjs
# ---------------------------------------------------------------------------------------------
class TestInspectMetrics(_TmpDirCase):
    def test_unreadable_texture_size_fails(self):
        # gltf-transform cannot size a BMP (getSize() -> null): this used to be reported as 0x0 / 0 VRAM
        src = textured_quad_glb(self.tmp / "in.glb", image_bytes(fmt="BMP"), "image/bmp")
        proc = subprocess.run(["node", str(INSPECT_SCRIPT), str(src), "--compact"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Cannot read the size of texture 0", proc.stderr.splitlines()[0])


# ---------------------------------------------------------------------------------------------
# uvatlas.py
# ---------------------------------------------------------------------------------------------
class _FakeTensorMesh:
    """Stands in for open3d.t.geometry.TriangleMesh: compute_uvatlas fails or yields fixed UVs."""
    calls = 0
    error = None
    uvs = None

    def __init__(self, vertices, faces):
        self.n_faces = len(faces)

    def compute_uvatlas(self, **kwargs):
        type(self).calls += 1
        if type(self).error is not None:
            raise type(self).error
        return 0.1, 6, 1

    @property
    def triangle(self):
        uvs = type(self).uvs
        return {"texture_uvs": SimpleNamespace(numpy=lambda: uvs)}


@unittest.skipUnless(uvatlas.is_open3d_uvatlas_available(), "Open3D UVAtlas not installed")
class TestUVAtlasBackend(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box()

    def _fake_open3d(self, error=None, uvs=None):
        fake_mesh = type("FakeTensorMesh", (_FakeTensorMesh,), {"calls": 0, "error": error, "uvs": uvs})
        fake = SimpleNamespace(
            geometry=uvatlas.o3d.geometry,
            utility=uvatlas.o3d.utility,
            core=SimpleNamespace(Tensor=lambda a: a),
            t=SimpleNamespace(geometry=SimpleNamespace(TriangleMesh=fake_mesh)),
        )
        return mock.patch.object(uvatlas, "o3d", fake), fake_mesh

    def test_cli_backend_removed(self):
        for name in ("unwrap_mesh_uvatlas_cli", "get_uvatlas_cli_path"):
            self.assertFalse(hasattr(uvatlas, name), name)
        self.assertEqual(uvatlas.is_uvatlas_available(), (True, "open3d"))

    def test_unavailable_open3d_reports_none_even_with_cli_env(self):
        with mock.patch.dict(os.environ, {"UVATLAS_BIN": "/bin/echo"}), \
                mock.patch.object(uvatlas, "_OPEN3D_UVATLAS_AVAILABLE", False):
            self.assertEqual(uvatlas.is_uvatlas_available(), (False, "none"))
            with self.assertRaises(PipelineAbort) as ctx:
                uvatlas.unwrap_mesh_uvatlas(self.mesh, target_res=256)
        self.assertIn("Open3D", ctx.exception.reason)

    def test_manifold_repair_requires_open3d(self):
        with mock.patch.object(uvatlas, "_OPEN3D_AVAILABLE", False):
            with self.assertRaises(PipelineAbort):
                uvatlas.ensure_manifold_zero_decimation(self.mesh.vertices, self.mesh.faces)

    def test_failure_is_not_retried(self):
        patch, fake_mesh = self._fake_open3d(error=RuntimeError("isochart exploded"))
        with patch, self.assertRaises(PipelineAbort) as ctx:
            uvatlas.unwrap_mesh_uvatlas(self.mesh, target_res=256)
        self.assertEqual(fake_mesh.calls, 1, "no silent retry with relaxed settings")
        self.assertIn("isochart exploded", ctx.exception.reason)

    def _uvs(self, value=None):
        uvs = np.random.default_rng(0).uniform(0.1, 0.9, (len(self.mesh.faces), 3, 2)).astype(np.float32)
        if value is not None:
            uvs[3, 1, 0] = value
        return uvs

    def test_out_of_range_uvs_raise(self):
        for bad in (1.5, -0.01):
            with self.subTest(value=bad):
                patch, _ = self._fake_open3d(uvs=self._uvs(bad))
                with patch, self.assertRaises(PipelineAbort) as ctx:
                    uvatlas.unwrap_mesh_uvatlas(self.mesh, target_res=256)
                self.assertIn("[0, 1]", ctx.exception.reason)

    def test_non_finite_uvs_raise(self):
        patch, _ = self._fake_open3d(uvs=self._uvs(np.nan))
        with patch, self.assertRaises(PipelineAbort) as ctx:
            uvatlas.unwrap_mesh_uvatlas(self.mesh, target_res=256)
        self.assertIn("non-finite", ctx.exception.reason)

    def test_float_noise_at_the_border_is_accepted(self):
        patch, _ = self._fake_open3d(uvs=self._uvs(np.float32(1.0000001)))
        with patch:
            _, faces, uv, _, _ = uvatlas.unwrap_mesh_uvatlas(self.mesh, target_res=256)
        self.assertEqual(len(faces), len(self.mesh.faces))
        self.assertLessEqual(float(uv.max()), 1.0)


# ---------------------------------------------------------------------------------------------
# uv_baker.py
# ---------------------------------------------------------------------------------------------
class TestUvBakerNoFallbacks(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box()
        self.uv = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]
        ], dtype=np.float64)
        self.img = Image.new("RGB", (128, 128), (200, 90, 40))

    def _pbr(self, **slots):
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=self.img, metallicFactor=0.1, roughnessFactor=0.6, **slots)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)

    def _bake(self, uv=None):
        return uv_baker.rechart_and_bake_high_density(
            self.mesh, source_image=self.img, source_uv=self.uv if uv is None else uv, dilation_padding=4
        )

    def test_non_pbr_material_raises(self):
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, image=self.img)  # SimpleMaterial
        with self.assertRaises(PipelineAbort) as ctx:
            self._bake()
        self.assertIn("SimpleMaterial", ctx.exception.reason)

    def test_missing_material_raises(self):
        with self.assertRaises(PipelineAbort):
            self._bake()

    def test_nan_in_sampled_colours_raises(self):
        self._pbr()
        nan_sampler = lambda img, uv: np.full((len(uv), img.shape[2]), np.nan)
        with mock.patch.object(uv_baker, "_sample_texture_bilinear", side_effect=nan_sampler):
            with self.assertRaises(PipelineAbort) as ctx:
                self._bake()
        self.assertIn("NaN", ctx.exception.reason)

    def test_nan_in_resampled_slot_raises(self):
        self._pbr(metallicRoughnessTexture=Image.new("RGB", (64, 64), (0, 128, 255)))
        real = uv_baker._sample_texture_bilinear
        calls = []

        def sampler(img, uv):
            calls.append(1)
            out = real(img, uv)
            return out if len(calls) == 1 else np.full_like(out, np.nan)

        with mock.patch.object(uv_baker, "_sample_texture_bilinear", side_effect=sampler):
            with self.assertRaises(PipelineAbort) as ctx:
                self._bake()
        self.assertIn("metallicRoughnessTexture", ctx.exception.reason)

    def test_non_finite_source_uv_raises(self):
        self._pbr()
        uv = self.uv.copy()
        uv[2, 0] = np.nan
        with self.assertRaises(PipelineAbort) as ctx:
            self._bake(uv)
        self.assertIn("non-finite", ctx.exception.reason)

    def test_source_uv_count_mismatch_raises(self):
        self._pbr()
        with self.assertRaises(PipelineAbort) as ctx:
            self._bake(self.uv[:-1])
        self.assertIn("7 source UVs for 8 vertices", ctx.exception.reason)


# ---------------------------------------------------------------------------------------------
# texture_utils.py
# ---------------------------------------------------------------------------------------------
class TestExtractOriginalTextureInfo(_TmpDirCase):
    def test_single_textured_mesh(self):
        png = image_bytes()
        info = extract_original_texture_info(textured_quad_glb(self.tmp / "in.glb", png, "image/png"))
        self.assertEqual(info["default_format"], "PNG")
        self.assertEqual(info["base_image"].size, (64, 32))
        self.assertEqual(info["slots"]["baseColorTexture"]["raw_bytes"], png)

    def test_unreadable_file_raises(self):
        bogus = self.tmp / "bogus.glb"
        bogus.write_bytes(b"not a glb at all" * 8)
        with self.assertRaises(PipelineAbort) as ctx:
            extract_original_texture_info(bogus)
        self.assertIn("bogus.glb", ctx.exception.reason)

    def _assert_single_mesh_error(self, path: Path):
        with self.assertRaises(PipelineAbort) as ctx:
            extract_original_texture_info(path)
        self.assertIn("only single-mesh, single-material models are supported", ctx.exception.reason)

    def test_two_meshes_raise(self):
        path = self.tmp / "two_meshes.glb"
        build_glb(path, meshes=[[quad(material=0)], [quad((2, 0, 0), material=0)]],
                  images=[(image_bytes(), "image/png")], materials=[textured_material()])
        self._assert_single_mesh_error(path)

    def test_two_materials_raise(self):
        path = self.tmp / "two_materials.glb"
        build_glb(path, meshes=[[quad(material=0), quad((2, 0, 0), material=1)]],
                  images=[(image_bytes(), "image/png"), (image_bytes((32, 32)), "image/png")],
                  materials=[textured_material(0), textured_material(1)])
        self._assert_single_mesh_error(path)

    def test_instanced_mesh_raises(self):
        path = self.tmp / "instanced.glb"
        build_glb(path, meshes=[[quad(material=0)]], nodes=[0, 0],
                  images=[(image_bytes(), "image/png")], materials=[textured_material()])
        self._assert_single_mesh_error(path)

    def test_untextured_raises(self):
        path = self.tmp / "untextured.glb"
        build_glb(path, meshes=[[quad(material=0)]], materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}}])
        with self.assertRaises(PipelineAbort) as ctx:
            extract_original_texture_info(path)
        self.assertIn("baseColorTexture", ctx.exception.reason)

    def test_unknown_image_format_raises(self):
        path = textured_quad_glb(self.tmp / "bmp.glb", image_bytes(fmt="BMP"), "image/bmp")
        with self.assertRaises(PipelineAbort) as ctx:
            extract_original_texture_info(path)
        self.assertIn("BMP", ctx.exception.reason)

    def _scene_with_base_image(self, img: Image.Image) -> trimesh.Scene:
        mesh = trimesh.creation.box()
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((8, 2)), material=mat)
        return trimesh.Scene({"m": mesh})

    def test_missing_encoded_bytes_raise(self):
        img = Image.open(io.BytesIO(image_bytes()))
        img.load()  # decoding closes the source stream: img.fp is None
        with mock.patch.object(trimesh, "load", return_value=self._scene_with_base_image(img)):
            with self.assertRaises(PipelineAbort) as ctx:
                extract_original_texture_info(self.tmp / "any.glb")
        self.assertIn("encoded bytes", ctx.exception.reason)

    def test_unreadable_encoded_bytes_raise(self):
        img = Image.open(io.BytesIO(image_bytes()))

        class BrokenStream:
            def seek(self, *_):
                raise OSError("stream closed")

        img.fp = BrokenStream()
        with mock.patch.object(trimesh, "load", return_value=self._scene_with_base_image(img)):
            with self.assertRaises(PipelineAbort) as ctx:
                extract_original_texture_info(self.tmp / "any.glb")
        self.assertIn("stream closed", ctx.exception.reason)


class TestTextureExport(unittest.TestCase):
    def _mesh_with(self, img) -> trimesh.Trimesh:
        mesh = trimesh.creation.box()
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((8, 2)), material=mat)
        return mesh

    def test_original_without_encoded_bytes_raises(self):
        mesh = self._mesh_with(Image.new("RGB", (8, 8)))
        with self.assertRaises(PipelineAbort) as ctx:
            optimize_mesh_texture_for_export(mesh, preferred_format="ORIGINAL")
        self.assertIn("ORIGINAL", ctx.exception.reason)

    def test_jpeg_would_drop_alpha_raises(self):
        mesh = self._mesh_with(Image.new("RGBA", (8, 8), (10, 20, 30, 100)))
        with self.assertRaises(PipelineAbort) as ctx:
            optimize_mesh_texture_for_export(mesh, preferred_format="JPEG")
        self.assertIn("alpha", ctx.exception.reason)

    def test_unknown_format_raises(self):
        mesh = self._mesh_with(Image.new("RGB", (8, 8)))
        with self.assertRaises(PipelineAbort) as ctx:
            optimize_mesh_texture_for_export(mesh, preferred_format="TIFF")
        self.assertIn("TIFF", ctx.exception.reason)

    def test_no_texture_raises(self):
        mesh = trimesh.creation.box()
        with self.assertRaises(PipelineAbort):
            optimize_mesh_texture_for_export(mesh, preferred_format="PNG")

    def test_explicit_png_is_encoded_even_for_passthrough_images(self):
        jpeg = image_bytes((8, 8), fmt="JPEG")
        mesh = self._mesh_with(Image.open(io.BytesIO(jpeg)))
        # Like Steps 1-2: the image's save() now writes the original JPEG bytes
        preserve_mesh_textures(mesh, {"slots": {"baseColorTexture": {"format": "JPEG", "raw_bytes": jpeg}}})
        out = optimize_mesh_texture_for_export(mesh, preferred_format="PNG")
        self.assertEqual(out.format, "PNG")
        self.assertTrue(out._fast_save_data.startswith(b"\x89PNG"))
        self.assertEqual(Image.open(io.BytesIO(out._fast_save_data)).format, "PNG")

    def test_original_passthrough_keeps_bytes(self):
        img = Image.new("RGB", (8, 8))
        img._fast_save_data = b"\xff\xd8\xff original jpeg bytes"
        out = optimize_mesh_texture_for_export(self._mesh_with(img), preferred_format="ORIGINAL")
        self.assertIs(out, img)

    def test_preserve_without_slot_info_raises(self):
        mesh = self._mesh_with(Image.new("RGB", (8, 8)))
        with self.assertRaises(PipelineAbort) as ctx:
            preserve_mesh_textures(mesh, {"slots": {}, "default_format": "PNG"})
        self.assertIn("baseColorTexture", ctx.exception.reason)

    def test_preserve_without_raw_bytes_raises(self):
        mesh = self._mesh_with(Image.new("RGB", (8, 8)))
        with self.assertRaises(PipelineAbort):
            preserve_mesh_textures(mesh, {"slots": {"baseColorTexture": {"format": "PNG", "raw_bytes": None}},
                                          "default_format": "PNG"})


# ---------------------------------------------------------------------------------------------
# cleaner.py
# ---------------------------------------------------------------------------------------------
class TestCleaner(unittest.TestCase):
    def test_non_finite_vertices_raise_before_any_change(self):
        mesh = trimesh.creation.box()
        vertices = mesh.vertices.copy()
        vertices[1] = [np.nan, 0.0, 0.0]
        vertices[5] = [0.0, np.inf, np.nan]
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        with self.assertRaises(PipelineAbort) as ctx:
            clean_and_repair_mesh(mesh)
        self.assertEqual(ctx.exception.reason, "mesh has 2 non-finite vertex coordinates")

    def test_repair_errors_propagate(self):
        for name in ("fix_normals", "fix_winding"):
            with self.subTest(name=name):
                with mock.patch(f"trimesh.repair.{name}", side_effect=RuntimeError(f"{name} failed")):
                    with self.assertRaisesRegex(RuntimeError, f"{name} failed"):
                        clean_and_repair_mesh(trimesh.creation.box())


# ---------------------------------------------------------------------------------------------
# palette.py
# ---------------------------------------------------------------------------------------------
class TestPalette(unittest.TestCase):
    def test_no_input_raises(self):
        with self.assertRaises(PipelineAbort):
            extract_palette()

    def test_no_usable_pixels_raise(self):
        with self.assertRaises(PipelineAbort):
            extract_palette(sample_pixels=np.zeros((0, 3), dtype=np.uint8))
        transparent = np.zeros((50, 4), dtype=np.uint8)
        with self.assertRaises(PipelineAbort):
            extract_palette(sample_pixels=transparent)

    def test_wrongly_shaped_sample_pixels_raise(self):
        image = Image.new("RGB", (16, 16), (200, 30, 30))
        for bad in (np.zeros((10, 2)), np.zeros(30), np.zeros((2, 5, 3))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(PipelineAbort):
                    extract_palette(image=image, sample_pixels=bad)

    def test_kmeans_failure_raises(self):
        pixels = np.random.default_rng(0).integers(0, 255, (500, 3))
        with mock.patch("optimizer.core.palette.kmeans2", side_effect=ValueError("kmeans blew up")):
            with self.assertRaises(PipelineAbort) as ctx:
                extract_palette(sample_pixels=pixels)
        self.assertIn("kmeans blew up", ctx.exception.reason)

    def test_fewer_pixels_than_colours(self):
        # A small model (one sample pixel per vertex) has fewer pixels than the 10 requested colours
        pixels = np.array([[250, 0, 0], [0, 250, 0], [0, 0, 250], [250, 0, 0]])
        data = extract_palette(sample_pixels=pixels, n_colors=10)
        self.assertEqual(sorted(data["palette"]), ["#0000fa", "#00fa00", "#fa0000"])
        self.assertEqual(data["primaryColor"], "#fa0000")

    def test_valid_pixels_give_a_palette(self):
        pixels = np.random.default_rng(0).integers(0, 255, (500, 3))
        data = extract_palette(sample_pixels=pixels, n_colors=6)
        self.assertEqual(data["primaryColor"], data["palette"][0])

    def test_embed_rejects_non_glb_bytes(self):
        glb = trimesh.exchange.gltf.export_glb(trimesh.Scene({"m": trimesh.creation.box()}))
        self.assertIn(b'"palette"', embed_gltf_extras(glb, {"palette": ["#ffffff"]}))
        bad_first_chunk = glb[:16] + b"BIN\x00" + glb[20:]
        bad_json = glb[:20] + b"#" + glb[21:]  # first JSON byte '{' replaced: not JSON any more
        for bad in (b"", b"glTF", b"x" * 64, bad_first_chunk, bad_json):
            with self.subTest(size=len(bad)):
                with self.assertRaises(PipelineAbort):
                    embed_gltf_extras(bad, {"palette": []})


# ---------------------------------------------------------------------------------------------
# shell_orient.py
# ---------------------------------------------------------------------------------------------
class TestShellOrient(unittest.TestCase):
    def test_zero_size_bounding_box_raises(self):
        vertices = np.zeros((3, 3))
        with self.assertRaises(PipelineAbort) as ctx:
            shell_orient.orient_faces_by_visibility(vertices, np.array([[0, 1, 2]]))
        self.assertIn("bounding box", ctx.exception.reason)

    def test_pool_startup_failure_uses_threads_with_warning(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        expected = shell_orient._rasterize_votes(mesh.vertices, mesh.faces, 6, 32, max_workers=1)

        def no_fork(method):
            raise ValueError(f"cannot find context for {method!r}")

        with mock.patch.object(shell_orient.mp, "get_context", side_effect=no_fork):
            with self.assertLogs("optimizer.core.shell_orient", level="WARNING") as logs:
                got = shell_orient._rasterize_votes(mesh.vertices, mesh.faces, 6, 32, max_workers=2)
        self.assertIn("thread", "\n".join(logs.output))
        for a, b in zip(expected, got):
            np.testing.assert_array_equal(a, b)

    def test_worker_errors_are_not_retried_on_threads(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bad_faces = np.array([[0, 1, 7]])  # out-of-range index fails inside every worker
        with mock.patch.object(shell_orient.concurrent.futures, "ThreadPoolExecutor") as threads:
            with self.assertRaises(IndexError):
                shell_orient._rasterize_votes(vertices, bad_faces, 4, 16, max_workers=2)
        threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
