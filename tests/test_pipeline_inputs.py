"""
test_pipeline_inputs.py

optimizer/step_pipeline.py refuses what it cannot process instead of substituting something else:
- preflight (before Step 0): node, sharp, basisu for KTX2, Open3D UVAtlas for uv_mode uvatlas;
- input validation from the GLB JSON (before Step 0): already-compressed inputs, no / several mesh
  primitives, no material / baseColorTexture / TEXCOORD_0, UV count != vertex count, undecodable
  texture; non-finite vertices stop Step 1 (cleaner);
- Node sub-step failures are reported as one concise line (full output stays in the traceback);
- Step 7 requires the Node summary to report encoded textures.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from optimizer import step_pipeline
from optimizer.core.errors import PipelineAbort
from optimizer.step_pipeline import StepPipeline, inspect_glb_metrics, preflight_tools, validate_input_glb
from tests.fixtures import build_glb, create_mock_glb, image_bytes, quad, textured_material

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "examples" / "models"

# Makes every `import 'sharp'` fail (loaded through NODE_OPTIONS=--import=...).
BLOCK_ALL_SHARP_JS = r"""
import { registerHooks } from 'node:module';
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'sharp') throw new Error("Cannot find package 'sharp' (blocked by test)");
    return nextResolve(specifier, context);
  }
});
"""


def textured_glb(path: Path, prims=None, **kwargs) -> Path:
    kwargs.setdefault("images", [(image_bytes((64, 64)), "image/png")])
    kwargs.setdefault("materials", [textured_material()])
    build_glb(path, meshes=[prims if prims is not None else [quad(material=0)]], **kwargs)
    return path


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="test_pipeline_inputs_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def node_only_path(self) -> str:
        """A PATH holding `node` and nothing else (no basisu)."""
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        if not (bin_dir / "node").exists():
            (bin_dir / "node").symlink_to(shutil.which("node"))
        return str(bin_dir)


class TestInputValidation(_Tmp):
    def assert_refused(self, path: Path, reason: str):
        with self.assertRaises(PipelineAbort) as ctx:
            validate_input_glb(path)
        self.assertIn(reason, ctx.exception.reason)
        return ctx.exception

    def test_valid_inputs_pass(self):
        validate_input_glb(textured_glb(self.tmp / "ok.glb"))
        validate_input_glb(REPO_ROOT / "examples" / "sample_dinoki.glb")
        # KHR_mesh_quantization alone is fine (not a compressed / optimized output)
        validate_input_glb(MODELS / "zelvanox_raw.glb")

    def test_already_compressed_input(self):
        err = self.assert_refused(MODELS / "coramini.glb", "Input is already optimized/compressed")
        self.assertIn("EXT_meshopt_compression, KHR_texture_basisu", err.reason)
        path = textured_glb(self.tmp / "draco.glb", extensions_used=["KHR_draco_mesh_compression"])
        self.assert_refused(path, "(KHR_draco_mesh_compression)")

    def test_no_mesh(self):
        build_glb(self.tmp / "empty.glb")
        self.assert_refused(self.tmp / "empty.glb", "input has no mesh primitives")

    def test_untextured(self):
        path = textured_glb(self.tmp / "untextured.glb", images=(),
                            materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}}])
        self.assert_refused(path, "no baseColorTexture")
        path = textured_glb(self.tmp / "no_material.glb", prims=[quad()], materials=())
        self.assert_refused(path, "no material")

    def test_multi_material(self):
        path = textured_glb(
            self.tmp / "two_materials.glb", prims=[quad(material=0), quad((2, 0, 0), material=1)],
            images=[(image_bytes(), "image/png"), (image_bytes((32, 32)), "image/png")],
            materials=[textured_material(0), textured_material(1)])
        self.assert_refused(path, "only single-mesh, single-material models are supported")
        path = self.tmp / "two_meshes.glb"
        build_glb(path, meshes=[[quad(material=0)], [quad((2, 0, 0), material=0)]],
                  images=[(image_bytes(), "image/png")], materials=[textured_material()])
        self.assert_refused(path, "only single-mesh, single-material models are supported")
        instanced = textured_glb(self.tmp / "instanced.glb", nodes=[0, 0])
        self.assert_refused(instanced, "only single-mesh, single-material models are supported")

    def test_no_uvs(self):
        path = textured_glb(self.tmp / "no_uv.glb", prims=[quad(uv=False, material=0)])
        self.assert_refused(path, "no TEXCOORD_0")

    def test_uv_count_differs_from_vertex_count(self):
        prim = quad(material=0)
        prim["uv"] = prim["uv"][:3]
        path = textured_glb(self.tmp / "short_uv.glb", prims=[prim])
        self.assert_refused(path, "TEXCOORD_0 has 3 UVs for 4 vertices")

    def test_base_color_through_other_uv_set(self):
        mat = textured_material()
        mat["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"] = 1
        path = textured_glb(self.tmp / "texcoord1.glb", materials=[mat])
        self.assert_refused(path, "TEXCOORD_1")

    def test_undecodable_texture(self):
        path = textured_glb(self.tmp / "corrupt.glb", images=[(b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, "image/png")])
        self.assert_refused(path, "baseColorTexture image")

    def test_bad_data_uri(self):
        path = self.tmp / "data_uri.glb"
        textured_glb(path)
        gltf, bin_chunk = step_pipeline.read_glb(path)
        del gltf["images"][0]["bufferView"]
        gltf["images"][0]["uri"] = "data:image/png;base64,@@not-base64@@"
        build_like = json.dumps(gltf).encode()
        build_like += b" " * (-len(build_like) % 4)
        body = len(build_like).to_bytes(4, "little") + b"JSON" + build_like
        body += len(bin_chunk).to_bytes(4, "little") + b"BIN\x00" + bin_chunk
        path.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12 + len(body)).to_bytes(4, "little") + body)
        self.assert_refused(path, "data: URI")

    def test_unsupported_texture_format(self):
        path = textured_glb(self.tmp / "bmp.glb", images=[(image_bytes(fmt="BMP"), "image/bmp")])
        self.assert_refused(path, "BMP")

    def test_run_refuses_before_step_0(self):
        path = textured_glb(self.tmp / "no_uv.glb", prims=[quad(uv=False, material=0)])
        out = self.tmp / "out"
        with self.assertRaises(PipelineAbort) as ctx:
            StepPipeline(texture_format="webp", verbose=False, stream_events=False).run(path, out)
        self.assertEqual(ctx.exception.step, 0)
        self.assertFalse((out / "step_00_raw.glb").exists())

    def test_nan_vertex_stops_step_1(self):
        prim = quad(material=0)
        prim["positions"][2] = [np.nan, 0.0, 0.0]
        path = textured_glb(self.tmp / "nan.glb", prims=[prim])
        validate_input_glb(path)  # the JSON is fine; the cleaner refuses the geometry
        with self.assertRaises(PipelineAbort) as ctx:
            StepPipeline(texture_format="webp", verbose=False, stream_events=False).run(path, self.tmp / "out")
        self.assertEqual(ctx.exception.reason, "mesh has 1 non-finite vertex coordinates")
        self.assertEqual(ctx.exception.step, 1)

    def test_unknown_texture_format_option(self):
        with self.assertRaises(ValueError):
            StepPipeline(texture_format="png")


class TestPreflight(_Tmp):
    def test_all_tools_present(self):
        preflight_tools("ktx2", "xatlas", True)

    def test_missing_node(self):
        (self.tmp / "empty").mkdir()
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp / "empty")}):
            with self.assertRaises(PipelineAbort) as ctx:
                preflight_tools("webp", "xatlas", True)
        self.assertIn("node", ctx.exception.reason)

    def test_missing_basisu_only_matters_for_ktx2(self):
        with mock.patch.dict(os.environ, {"PATH": self.node_only_path()}):
            with self.assertRaises(PipelineAbort) as ctx:
                preflight_tools("ktx2", "xatlas", True)
            self.assertIn("basisu", ctx.exception.reason)
            preflight_tools("webp", "xatlas", True)
            preflight_tools("original", "xatlas", True)

    def test_missing_sharp(self):
        hook = self.tmp / "block_sharp.mjs"
        hook.write_text(BLOCK_ALL_SHARP_JS)
        with mock.patch.dict(os.environ, {"NODE_OPTIONS": f"--import={hook.as_uri()}"}):
            with self.assertRaises(PipelineAbort) as ctx:
                preflight_tools("webp", "xatlas", True)
        self.assertIn("sharp", ctx.exception.reason)

    def test_missing_open3d_only_matters_for_uvatlas_with_downscale(self):
        with mock.patch.object(step_pipeline, "is_uvatlas_available", return_value=(False, "none")):
            with self.assertRaises(PipelineAbort) as ctx:
                preflight_tools("webp", "uvatlas", True)
            self.assertIn("Open3D", ctx.exception.reason)
            preflight_tools("webp", "uvatlas", False)
            preflight_tools("webp", "xatlas", True)

    def test_run_fails_before_step_0_without_basisu(self):
        glb = self.tmp / "mock.glb"
        create_mock_glb(glb)
        out = self.tmp / "out"
        with mock.patch.dict(os.environ, {"PATH": self.node_only_path()}):
            with self.assertRaises(PipelineAbort) as ctx:
                StepPipeline(texture_format="ktx2", verbose=False, stream_events=False).run(glb, out)
        self.assertIn("basisu", ctx.exception.reason)
        self.assertFalse((out / "step_00_raw.glb").exists(), "preflight runs before Step 0")

    def test_cli_reports_missing_basisu_as_pipeline_error(self):
        glb = self.tmp / "mock.glb"
        create_mock_glb(glb)
        proc = subprocess.run(
            [sys.executable, "-m", "optimizer.step_pipeline", str(glb), "--output-dir", str(self.tmp / "out")],
            capture_output=True, text=True, cwd=REPO_ROOT, env={**os.environ, "PATH": self.node_only_path()}
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual([e["event"] for e in events], ["pipeline_error"])
        self.assertIn("basisu", events[0]["error"])


class TestNodeSubStepReasons(_Tmp):
    def test_failed_node_step_gives_one_line_reason(self):
        proc = subprocess.CompletedProcess(
            ["node"], 1, stdout="",
            stderr="   ⚠️ Warning: texture 0 is shared by colour and data slots; encoding as sRGB.\n"
                   "Fatal optimization error: texture 0: basisu KTX2 encoding failed: ERROR: load_png failed\n"
                   "Error: texture 0: basisu KTX2 encoding failed: ERROR: load_png failed\n"
                   "    at processTextureItem (file:///x/optimize_meshopt.mjs:210:15)\n"
        )
        err = step_pipeline.node_step_failure("Step 7: KTX2 texture compression failed", proc)
        self.assertIsInstance(err, PipelineAbort)
        self.assertEqual(
            err.reason,
            "Step 7: KTX2 texture compression failed: texture 0: basisu KTX2 encoding failed: ERROR: load_png failed"
        )
        self.assertIn("    at processTextureItem", str(err.__cause__), "full output kept on the cause")

    def test_inspect_failure_is_one_line(self):
        glb = self.tmp / "bmp.glb"
        build_glb(glb, meshes=[[quad(material=0)]], images=[(image_bytes(fmt="BMP"), "image/bmp")],
                  materials=[textured_material()])
        with self.assertRaises(PipelineAbort) as ctx:
            inspect_glb_metrics(glb)
        self.assertNotIn("\n", ctx.exception.reason)
        self.assertIn("Cannot read the size of texture 0", ctx.exception.reason)
        self.assertNotIn("Failed to inspect GLB metrics", ctx.exception.reason)

    def test_step7_summary_must_report_encoded_textures(self):
        def proc(summary):
            return subprocess.CompletedProcess(["node"], 0, stdout="noise\n" + json.dumps(summary) + "\n", stderr="")

        ok = {"success": True, "ktx2": {"count": 1, "mode": "uastc"}, "webp": None}
        self.assertEqual(step_pipeline.encoded_texture_summary(proc(ok), "ktx2")["count"], 1)
        for bad in (
            {"success": True, "ktx2": {"count": 0}, "webp": None},
            {"success": True, "ktx2": {"count": 1, "skipped": True}, "webp": None},
            {"success": True, "ktx2": None, "webp": None},
        ):
            with self.subTest(summary=bad):
                with self.assertRaises(PipelineAbort):
                    step_pipeline.encoded_texture_summary(proc(bad), "ktx2")
        with self.assertRaises(PipelineAbort):
            step_pipeline.encoded_texture_summary(proc({"success": True, "ktx2": None, "webp": {"count": 0}}), "webp")
        with self.assertRaises(PipelineAbort):
            step_pipeline.encoded_texture_summary(
                subprocess.CompletedProcess(["node"], 0, stdout="no json here\n", stderr=""), "webp")


if __name__ == "__main__":
    unittest.main()
