"""
test_pipeline.py

Comprehensive Automated Test Suite & Benchmark Verification for 3D Model Optimization.
Strictly verifies Rule 11 (Zero-Decimation Policy) and 1:1 pipeline parity.

Run via:
    python3 -m unittest tests/test_pipeline.py
    python3 tests/test_pipeline.py
    pytest tests/test_pipeline.py (if pytest installed)
"""

import os
import sys
import json
import struct
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer.pipeline import ModelOptimizer


def create_synthetic_textured_glb(path: Path, subdivisions: int = 2) -> int:
    """Creates a valid textured sphere GLB with UVs and image texture for testing."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    
    # Generate test texture image
    img = Image.new("RGB", (256, 256), (100, 180, 70))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 90, 90], fill=(220, 50, 40))
    draw.ellipse([110, 110, 200, 200], fill=(40, 80, 220))

    # Spherical UV mapping
    norm_v = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    u = 0.5 + np.arctan2(norm_v[:, 0], norm_v[:, 2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(norm_v[:, 1]) / np.pi
    uv = np.column_stack([u, v])

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    scene = trimesh.Scene({"Model": mesh})
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    return len(mesh.faces)


def inspect_glb_metadata(glb_path: Path):
    """Parses binary GLB and extracts JSON chunk, face/vertex count, and extensions."""
    data = glb_path.read_bytes()
    magic, version, total_len = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67, f"Invalid GLB magic: {hex(magic)}"
    assert version == 2, f"Invalid GLB version: {version}"

    json_len, json_type = struct.unpack("<II", data[12:20])
    assert json_type == 0x4E4F534A, f"Invalid JSON chunk type: {hex(json_type)}"
    gltf = json.loads(data[20:20 + json_len].decode("utf-8"))

    prim = gltf["meshes"][0]["primitives"][0]
    accs = gltf["accessors"]
    tris = accs[prim["indices"]]["count"] // 3
    verts = accs[prim["attributes"]["POSITION"]]["count"]
    exts = gltf.get("extensionsUsed", [])
    extras = gltf.get("extras", {})
    return {
        "data_len": len(data),
        "gltf": gltf,
        "tris": tris,
        "verts": verts,
        "exts": exts,
        "extras": extras
    }


class TestRule11ZeroDecimation(unittest.TestCase):
    """
    Test suite for Rule 11 (Zero-Decimation Policy):
    - Asserts 100% geometric triangles preserved (ratio = 1.0).
    - Fails if any decimation or triangle reduction occurs.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_zero_decimation_synthetic_mesh(self):
        """Rule 11 Verification: Synthetic mesh triangle count is 100% preserved."""
        raw_glb = self.tmp_path / "synthetic_raw.glb"
        opt_glb = self.tmp_path / "synthetic_opt.glb"

        orig_faces = create_synthetic_textured_glb(raw_glb, subdivisions=2)
        optimizer = ModelOptimizer(resolution=512, texture_format="webp", rechart_uv=False, verbose=False)
        summary = optimizer.optimize(raw_glb, opt_glb)

        self.assertTrue(opt_glb.exists(), "Optimized GLB was not created")

        # Load both via trimesh
        mesh_before = trimesh.load(str(raw_glb), force="mesh", process=False)
        mesh_after = trimesh.load(str(opt_glb), force="mesh", process=False)

        self.assertEqual(
            len(mesh_after.faces),
            len(mesh_before.faces),
            f"Rule 11 Violation: Faces reduced from {len(mesh_before.faces)} to {len(mesh_after.faces)}"
        )
        self.assertEqual(len(mesh_after.faces), orig_faces)

        # Also verify via glTF accessor indices
        meta = inspect_glb_metadata(opt_glb)
        self.assertEqual(
            meta["tris"],
            orig_faces,
            f"Rule 11 Accessor Violation: Expected {orig_faces} tris, got {meta['tris']}"
        )

    def test_zero_decimation_real_statue(self):
        """Rule 11 Verification: Real statue sample preserves 100% triangles."""
        sample_dinoki = REPO_ROOT / "examples" / "sample_dinoki.glb"
        if not sample_dinoki.exists():
            self.skipTest(f"Sample model not found: {sample_dinoki}")

        opt_glb = self.tmp_path / "dinoki_test_opt.glb"
        optimizer = ModelOptimizer(resolution=512, texture_format="webp", rechart_uv=False, verbose=False)
        summary = optimizer.optimize(sample_dinoki, opt_glb)

        meta_before = inspect_glb_metadata(sample_dinoki)
        meta_after = inspect_glb_metadata(opt_glb)

        self.assertEqual(
            meta_after["tris"],
            meta_before["tris"],
            f"Rule 11 Violation on real statue: Expected {meta_before['tris']} tris, got {meta_after['tris']}"
        )
        self.assertEqual(meta_after["tris"], 45000)


class TestWindingAndNormals(unittest.TestCase):
    """
    Test suite for mesh winding, normals direction, and smooth shading across seams.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_normals_are_valid_and_outward(self):
        """Checks vertex normals are finite, normalized unit vectors, and face outwards."""
        raw_glb = self.tmp_path / "sphere_raw.glb"
        opt_glb = self.tmp_path / "sphere_opt.glb"

        create_synthetic_textured_glb(raw_glb, subdivisions=2)
        optimizer = ModelOptimizer(resolution=512, texture_format="webp", smooth_normals=True, verbose=False)
        optimizer.optimize(raw_glb, opt_glb)

        mesh = trimesh.load(str(opt_glb), force="mesh", process=False)
        normals = mesh.vertex_normals

        # Check finite
        self.assertTrue(np.isfinite(normals).all(), "Vertex normals contain NaN or Inf")

        # Check normalization (lengths ~ 1.0)
        lengths = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(lengths, 1.0, atol=0.05), "Vertex normals are not properly normalized")

        # For a centered sphere, dot product of vertex position and normal should be positive (outward)
        dots = np.sum(mesh.vertices * normals, axis=1)
        positive_ratio = np.mean(dots > 0)
        self.assertGreater(
            positive_ratio,
            0.95,
            f"Too many inward-pointing normals: {positive_ratio * 100:.1f}% outward"
        )


class TestFileIntegrityAndExtensions(unittest.TestCase):
    """
    Verifies GLB binary file structure, extensions, and metadata embedding.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_meshopt_and_texture_extensions(self):
        """Checks EXT_meshopt_compression and texture compression extensions."""
        raw_glb = self.tmp_path / "test_raw.glb"
        opt_glb = self.tmp_path / "test_opt.glb"

        create_synthetic_textured_glb(raw_glb, subdivisions=2)
        optimizer = ModelOptimizer(resolution=512, texture_format="ktx2", verbose=False)
        optimizer.optimize(raw_glb, opt_glb)

        meta = inspect_glb_metadata(opt_glb)
        exts = meta["exts"]

        self.assertIn("EXT_meshopt_compression", exts, "EXT_meshopt_compression is missing from extensionsUsed")
        # Texture extension can be KHR_texture_basisu or EXT_texture_webp
        has_compressed_tex = ("KHR_texture_basisu" in exts) or ("EXT_texture_webp" in exts)
        self.assertTrue(has_compressed_tex, f"Expected texture compression extension in {exts}")

    def test_extras_and_palette_embedding(self):
        """Checks that 10-color palette and optimization metadata are embedded in glTF extras."""
        raw_glb = self.tmp_path / "test_raw.glb"
        opt_glb = self.tmp_path / "test_opt.glb"

        create_synthetic_textured_glb(raw_glb, subdivisions=2)
        optimizer = ModelOptimizer(resolution=512, texture_format="webp", verbose=False)
        optimizer.optimize(raw_glb, opt_glb)

        meta = inspect_glb_metadata(opt_glb)
        extras = meta["extras"]

        self.assertIn("palette", extras, "glTF extras missing 'palette'")
        self.assertIn("primaryColor", extras, "glTF extras missing 'primaryColor'")
        self.assertIn("policy", extras, "glTF extras missing 'policy'")
        self.assertEqual(extras["policy"], "STRICT 0-DECIMATION (--ratio 1.0)")
        self.assertGreaterEqual(len(extras["palette"]), 1)


class TestCompressionAndPerformance(unittest.TestCase):
    """
    Measures file size reduction and execution time.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_real_statue_compression_and_speed(self):
        """Tests compression ratio and execution speed on real statue sample."""
        sample_input = REPO_ROOT / "examples" / "sample_input.glb"
        if not sample_input.exists():
            self.skipTest(f"Sample input not found: {sample_input}")

        opt_glb = self.tmp_path / "sample_opt.glb"
        t0 = time.time()
        optimizer = ModelOptimizer(resolution=1024, texture_format="ktx2", verbose=False)
        summary = optimizer.optimize(sample_input, opt_glb)
        elapsed = time.time() - t0

        raw_size = sample_input.stat().st_size
        opt_size = opt_glb.stat().st_size
        saved_pct = (1.0 - (opt_size / raw_size)) * 100.0

        self.assertLess(opt_size, raw_size, "Optimized file should be smaller than raw input")
        self.assertGreaterEqual(saved_pct, 40.0, f"Expected >= 40% reduction, achieved {saved_pct:.2f}%")
        self.assertLess(elapsed, 30.0, f"Optimization took too long: {elapsed:.2f}s")


class TestPipelineParity1to1(unittest.TestCase):
    """
    Rigorous 1:1 Parity Verification against the production pipeline.
    Validates:
    - 100% Triangle count match.
    - 100% Vertex count match.
    - Equivalent glTF extensions (Meshopt + KTX2 UASTC).
    - File size similarity within 1% margin.
    """

    def test_parity_with_dinoki_production(self):
        sample_dinoki = REPO_ROOT / "examples" / "sample_dinoki.glb"
        prod_reference = Path("/Users/nguyenhoainam/code/braitoli/3DPainting/3DPainting/public/models/khung_long.glb")

        if not sample_dinoki.exists() or not prod_reference.exists():
            self.skipTest("Dinoki sample or production reference not available for parity check")

        with tempfile.TemporaryDirectory() as tmp:
            new_opt_glb = Path(tmp) / "new_pipeline_dinoki.glb"
            optimizer = ModelOptimizer(resolution=1024, texture_format="ktx2", double_sided=True, verbose=False)
            optimizer.optimize(sample_dinoki, new_opt_glb)

            prod_meta = inspect_glb_metadata(prod_reference)
            new_meta = inspect_glb_metadata(new_opt_glb)

            # 1. Triangles 100% match
            self.assertEqual(
                new_meta["tris"],
                prod_meta["tris"],
                f"Face count mismatch: New={new_meta['tris']}, Prod={prod_meta['tris']}"
            )

            # 2. Vertices 100% match
            self.assertEqual(
                new_meta["verts"],
                prod_meta["verts"],
                f"Vertex count mismatch: New={new_meta['verts']}, Prod={prod_meta['verts']}"
            )

            # 3. Extensions match
            self.assertIn("EXT_meshopt_compression", new_meta["exts"])
            self.assertIn("KHR_texture_basisu", new_meta["exts"])

            # 4. File size similarity (< 1% difference)
            size_diff = abs(new_meta["data_len"] - prod_meta["data_len"])
            pct_diff = (size_diff / prod_meta["data_len"]) * 100.0
            self.assertLess(
                pct_diff,
                1.0,
                f"File size disparity exceeds 1%: New={new_meta['data_len']}B, Prod={prod_meta['data_len']}B ({pct_diff:.3f}%)"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
