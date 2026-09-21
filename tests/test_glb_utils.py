"""
test_glb_utils.py

optimizer/core/glb_utils.py raises on bad input instead of silently returning a default:
- set_frontside_material / set_doublesided_material on non-GLB bytes or a non-JSON first chunk
- set_doublesided_material on a GLB without materials (no default material is injected)
- check_glb_double_sided on non-GLB input, corrupt JSON or an unreadable path
- _decompress_meshopt_if_needed on non-GLB input, a Node failure, or when Node writes nothing
"""

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from optimizer.core.errors import PipelineAbort
from optimizer.core.glb_utils import (
    set_frontside_material,
    set_doublesided_material,
    check_glb_double_sided,
    _decompress_meshopt_if_needed
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MESHOPT_GLB = REPO_ROOT / "examples" / "sample_dinoki_opt.glb"


def make_glb(gltf, bin_chunk: bytes = b"", first_chunk_type: bytes = b"JSON") -> bytes:
    js = gltf if isinstance(gltf, bytes) else json.dumps(gltf).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    body = struct.pack("<I4s", len(js), first_chunk_type) + js
    if bin_chunk:
        bin_chunk += b"\x00" * ((4 - len(bin_chunk) % 4) % 4)
        body += struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def glb_json(glb_bytes: bytes) -> dict:
    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    assert chunk_type == b"JSON"
    return json.loads(glb_bytes[20:20 + chunk_len])


MATERIAL_GLB = make_glb({"asset": {"version": "2.0"}, "materials": [{"name": "m", "doubleSided": False}]})
NO_MATERIAL_GLB = make_glb({"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]})
BAD_INPUTS = {
    "empty": b"",
    "short": b"glTF\x02\x00\x00\x00",
    "not_glb": b"PK\x03\x04" + b"\x00" * 60,
    "bin_first_chunk": make_glb(b"\x00" * 8, first_chunk_type=b"BIN\x00"),
    "truncated_json_chunk": make_glb({"asset": {"version": "2.0"}, "materials": []})[:30],
    "corrupt_json": make_glb(b"{not json"),
}


class TestMaterialSideRewrite(unittest.TestCase):
    def test_bad_input_raises(self):
        for fn in (set_frontside_material, set_doublesided_material):
            for name, data in BAD_INPUTS.items():
                with self.subTest(fn=fn.__name__, input=name):
                    with self.assertRaises(PipelineAbort):
                        fn(data)

    def test_doublesided_without_materials_raises(self):
        with self.assertRaises(PipelineAbort) as ctx:
            set_doublesided_material(NO_MATERIAL_GLB)
        self.assertIn("no materials", ctx.exception.reason)

    def test_rewrites_valid_glb(self):
        ds = set_doublesided_material(MATERIAL_GLB)
        self.assertIs(glb_json(ds)["materials"][0]["doubleSided"], True)
        fs = set_frontside_material(ds)
        self.assertIs(glb_json(fs)["materials"][0]["doubleSided"], False)


class TestCheckGlbDoubleSided(unittest.TestCase):
    def test_valid_bytes_and_path(self):
        ds = set_doublesided_material(MATERIAL_GLB)
        self.assertIs(check_glb_double_sided(MATERIAL_GLB), False)
        self.assertIs(check_glb_double_sided(ds), True)
        with tempfile.TemporaryDirectory(prefix="test_check_ds_") as tmpdir:
            path = Path(tmpdir) / "ds.glb"
            path.write_bytes(ds)
            self.assertIs(check_glb_double_sided(path), True)
            self.assertIs(check_glb_double_sided(str(path)), True)

    def test_bad_input_raises(self):
        with tempfile.TemporaryDirectory(prefix="test_check_ds_bad_") as tmpdir:
            for name, data in BAD_INPUTS.items():
                path = Path(tmpdir) / f"{name}.glb"
                path.write_bytes(data)
                with self.subTest(input=name, kind="bytes"):
                    with self.assertRaises(PipelineAbort):
                        check_glb_double_sided(data)
                with self.subTest(input=name, kind="path"):
                    with self.assertRaises(PipelineAbort):
                        check_glb_double_sided(path)
            with self.assertRaises(FileNotFoundError):
                check_glb_double_sided(Path(tmpdir) / "missing.glb")


class TestDecompressMeshopt(unittest.TestCase):
    def test_non_glb_raises(self):
        with tempfile.TemporaryDirectory(prefix="test_unpack_bad_") as tmpdir:
            tmp = Path(tmpdir)
            bogus = tmp / "bogus.glb"
            bogus.write_bytes(BAD_INPUTS["not_glb"])
            with self.assertRaises(PipelineAbort):
                _decompress_meshopt_if_needed(bogus, tmp)

    def test_glb_without_meshopt_is_returned_as_is(self):
        with tempfile.TemporaryDirectory(prefix="test_unpack_plain_") as tmpdir:
            tmp = Path(tmpdir)
            plain = tmp / "plain.glb"
            plain.write_bytes(MATERIAL_GLB)
            self.assertEqual(_decompress_meshopt_if_needed(plain, tmp), plain)

    def test_node_failure_raises(self):
        with tempfile.TemporaryDirectory(prefix="test_unpack_node_fail_") as tmpdir:
            tmp = Path(tmpdir)
            broken = tmp / "broken_meshopt.glb"
            # gltf-transform refuses to read it: 'Unsupported glTF version, "1.0"'
            broken.write_bytes(make_glb({
                "asset": {"version": "1.0"},
                "extensionsUsed": ["EXT_meshopt_compression"]
            }, bin_chunk=b"\x01" * 16))
            with self.assertRaises(PipelineAbort) as ctx:
                _decompress_meshopt_if_needed(broken, tmp)
            self.assertIn("broken_meshopt.glb", ctx.exception.reason)
            self.assertIn("Unsupported glTF version", ctx.exception.reason)
            self.assertFalse((tmp / "unpacked_broken_meshopt.glb").exists())

    def test_node_writing_nothing_raises(self):
        with tempfile.TemporaryDirectory(prefix="test_unpack_no_output_") as tmpdir:
            tmp = Path(tmpdir)
            done = subprocess.CompletedProcess(args=["node"], returncode=0, stdout=b"", stderr=b"")
            with mock.patch("optimizer.core.glb_utils.subprocess.run", return_value=done):
                with self.assertRaises(PipelineAbort) as ctx:
                    _decompress_meshopt_if_needed(MESHOPT_GLB, tmp)
            self.assertIn("unpacked_sample_dinoki_opt.glb", ctx.exception.reason)

    def test_meshopt_glb_is_decompressed(self):
        with tempfile.TemporaryDirectory(prefix="test_unpack_ok_") as tmpdir:
            tmp = Path(tmpdir)
            out = _decompress_meshopt_if_needed(MESHOPT_GLB, tmp)
            self.assertEqual(out, tmp / "unpacked_sample_dinoki_opt.glb")
            self.assertNotIn("EXT_meshopt_compression", glb_json(out.read_bytes()).get("extensionsUsed", []))


if __name__ == "__main__":
    unittest.main()
