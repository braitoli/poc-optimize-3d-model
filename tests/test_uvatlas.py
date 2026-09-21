"""
test_uvatlas.py

Unit tests for Microsoft UVAtlas integration:
- Availability checks (Open3D and CLI fallback)
- Zero-decimation 2-manifold repair
- Iso-chart UV unwrapping
- High-density texture baking with UVAtlas
- StepPipeline uv_mode options and backward compatibility
"""

import unittest
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path

from optimizer.core.uvatlas import (
    is_uvatlas_available,
    is_open3d_uvatlas_available,
    ensure_manifold_zero_decimation,
    unwrap_mesh_uvatlas
)
from optimizer.core.uv_baker import (
    rebake_texture_uvatlas,
    rechart_and_bake_high_density
)
from optimizer.step_pipeline import StepPipeline


class TestUVAtlas(unittest.TestCase):
    def setUp(self):
        # Create a simple box mesh
        self.mesh = trimesh.creation.box()
        # Attach sample material with texture
        img = Image.new("RGB", (256, 256), (180, 80, 40))
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img,
            metallicFactor=0.2,
            roughnessFactor=0.7,
            doubleSided=False
        )
        uv = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]
        ], dtype=np.float64)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        self.test_model_path = Path("examples/models/dinoki_raw.glb")

    def test_uvatlas_availability(self):
        avail, backend = is_uvatlas_available()
        self.assertTrue(avail, "Microsoft UVAtlas must be available in environment")
        self.assertIn(backend, ["open3d", "cli"], f"Backend should be open3d or cli, got {backend}")

    def test_ensure_manifold_zero_decimation_on_box(self):
        verts, faces, vmap = ensure_manifold_zero_decimation(self.mesh.vertices, self.mesh.faces)
        self.assertEqual(len(faces), len(self.mesh.faces), "Face count must be strictly preserved")
        self.assertEqual(len(verts), len(self.mesh.vertices), "Box is already manifold, vertices should not change")
        self.assertTrue(np.array_equal(vmap, np.arange(len(verts))))

    def test_ensure_manifold_zero_decimation_on_non_manifold(self):
        # Create synthetic non-manifold mesh: two triangles sharing an edge + third triangle sharing same edge
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        # Edge (0, 1) shared by 3 faces
        faces = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4]
        ], dtype=np.int64)
        v_man, f_man, vmap = ensure_manifold_zero_decimation(verts, faces)
        self.assertEqual(len(f_man), 3, "All 3 faces must be preserved (Zero Decimation)")
        self.assertGreater(len(v_man), len(verts), "Non-manifold edge must be split by duplicating vertices")

    def test_unwrap_mesh_uvatlas_on_box(self):
        v_unwrapped, f_unwrapped, uv_unwrapped, vmap, stats = unwrap_mesh_uvatlas(
            self.mesh,
            target_res=512,
            gutter=2.0
        )
        self.assertEqual(len(f_unwrapped), len(self.mesh.faces), "Face count must match original")
        self.assertEqual(len(v_unwrapped), len(uv_unwrapped), "Vertex count must match UV count")
        self.assertEqual(len(vmap), len(v_unwrapped), "Vmapping length must match vertices")
        self.assertTrue(stats["zero_decimation_faces_preserved"])
        # Check UV range strictly [0, 1]
        self.assertGreaterEqual(float(uv_unwrapped.min()), 0.0)
        self.assertLessEqual(float(uv_unwrapped.max()), 1.0)
        self.assertGreater(stats.get("uvatlas_chart_count", 0), 0)

    def test_rebake_texture_uvatlas(self):
        orig_img = getattr(self.mesh.visual.material, "baseColorTexture")
        orig_uv = self.mesh.visual.uv
        out_mesh, baked_pil = rebake_texture_uvatlas(
            self.mesh,
            source_image=orig_img,
            source_uv=orig_uv,
            target_res=256,
            dilation_padding=8,
            double_sided=True
        )
        self.assertEqual(len(out_mesh.faces), len(self.mesh.faces), "Faces preserved")
        self.assertEqual(baked_pil.size, (256, 256), "Texture resolution correct")
        self.assertTrue(out_mesh.visual.material.doubleSided, "doubleSided must be True")

    def test_rechart_and_bake_high_density_uvatlas_mode(self):
        orig_img = getattr(self.mesh.visual.material, "baseColorTexture")
        orig_uv = self.mesh.visual.uv
        stats = {}
        out_mesh, baked_pil, stats = rechart_and_bake_high_density(
            self.mesh,
            target_res=256,
            source_image=orig_img,
            source_uv=orig_uv,
            dilation_padding=8,
            double_sided=False,
            stats=stats,
            return_stats=True,
            unwrap_method="uvatlas"
        )
        self.assertEqual(len(out_mesh.faces), len(self.mesh.faces))
        self.assertIn("uvatlas_chart_count", stats)
        self.assertIn("uv_coverage_ratio_percent", stats)
        self.assertGreater(stats["uv_coverage_ratio_percent"], 0.0)

    def test_step_pipeline_uv_mode_resolution(self):
        # Default mode -> xatlas
        p1 = StepPipeline()
        self.assertEqual(p1.uv_mode, "xatlas")

        # Explicit uv_mode uvatlas
        p3 = StepPipeline(uv_mode="uvatlas")
        self.assertEqual(p3.uv_mode, "uvatlas")

        # Explicit uv_mode xatlas
        p5 = StepPipeline(uv_mode="xatlas")
        self.assertEqual(p5.uv_mode, "xatlas")

    def test_uvatlas_on_complex_model_if_present(self):
        if not self.test_model_path.exists():
            self.skipTest(f"Test model {self.test_model_path} not found")
        mesh = trimesh.load(self.test_model_path, force="mesh")
        orig_face_count = len(mesh.faces)
        v_unw, f_unw, uv_unw, vmap, stats = unwrap_mesh_uvatlas(mesh, target_res=512)
        self.assertEqual(len(f_unw), orig_face_count, "Zero-decimation: 100% faces preserved on complex model")
        self.assertTrue(stats["zero_decimation_faces_preserved"])
        self.assertGreaterEqual(float(uv_unw.min()), 0.0)
        self.assertLessEqual(float(uv_unw.max()), 1.0)

    def test_uvatlas_gutter_margin_enforcement(self):
        # Passing gutter=1.0 should be clamped to >= 4.0
        v_unw, f_unw, uv_unw, vmap, stats = unwrap_mesh_uvatlas(self.mesh, target_res=256, gutter=1.0)
        self.assertGreaterEqual(stats.get("uvatlas_gutter", 0.0), 4.0)

    def test_uvatlas_canvas_init_and_mandatory_dilation(self):
        # Create a texture with stark pure white background (255, 255, 255)
        # and dark blue model surface (20, 40, 100)
        tex_arr = np.full((256, 256, 3), 255, dtype=np.uint8)
        tex_arr[100:200, 100:200] = [20, 40, 100]
        tex_img = Image.fromarray(tex_arr, mode="RGB")

        # Map mesh UVs to the blue region
        uv = np.array([
            [0.45, 0.30], [0.70, 0.30], [0.70, 0.55], [0.45, 0.55],
            [0.48, 0.35], [0.65, 0.35], [0.65, 0.50], [0.48, 0.50]
        ], dtype=np.float64)
        mesh_box = self.mesh.copy()
        mesh_box.visual.uv = uv

        stats = {}
        # Test with dilation_padding=0: UVAtlas must MANDATORILY apply at least 8px dilation
        baked_mesh, baked_pil, stats = rechart_and_bake_high_density(
            mesh_box,
            target_res=256,
            source_image=tex_img,
            source_uv=uv,
            dilation_padding=0,
            double_sided=False,
            stats=stats,
            return_stats=True,
            unwrap_method="uvatlas",
            uvatlas_gutter=2.0  # Should be clamped to >= 4.0
        )
        self.assertGreaterEqual(stats["dilation_padding"], 8, "UVAtlas must enforce mandatory dilation >= 8px")

        # Verify that the baked texture canvas does NOT have stark white background (255, 255, 255)
        baked_arr = np.asarray(baked_pil)
        # Average color of non-dilated areas should match the dark blue surface, not white (255, 255, 255)
        mean_baked = np.mean(baked_arr, axis=(0, 1))
        # Blue channel should dominate, and total brightness should be much lower than 255
        self.assertLess(float(mean_baked[0]), 50.0, "Red channel must not be white")
        self.assertLess(float(mean_baked[1]), 60.0, "Green channel must not be white")
        self.assertGreater(float(mean_baked[2]), 80.0, "Blue surface color must dominate")
        # Corner background must be the dark blue surface color, not white
        self.assertEqual(list(baked_arr[0, 0]), [20, 40, 100])


if __name__ == "__main__":
    unittest.main()
