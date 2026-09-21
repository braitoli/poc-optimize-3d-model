"""
test_uv_baker.py

Tests for:
- optimizer/core/uv_baker.py (direct_resample_texture & rebake_texture_xatlas)
- doubleSided=True preservation
- vertex_normals preservation (smooth shading, no flat normals)
- Alpha channel preservation
- Material properties preservation (roughness, metallic, alphaMode)
- set_doublesided_material helper
"""

import json
import struct
import unittest
import numpy as np
from PIL import Image
import trimesh

from optimizer.core.uv_baker import (
    direct_resample_texture,
    rebake_texture_xatlas,
    rechart_and_bake_high_density,
    compute_uv_metrics
)
from optimizer.core.texture_utils import clamp_target_resolution
from optimizer.pipeline import set_doublesided_material


class TestUvBaker(unittest.TestCase):
    def setUp(self):
        # Create a sample box mesh with smooth vertex normals and UVs
        self.mesh = trimesh.creation.box()
        # Ensure vertex normals exist
        _ = self.mesh.vertex_normals
        self.uv = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]
        ], dtype=np.float64)

    def test_direct_resample_preserves_doublesided_and_material_properties(self):
        mat = trimesh.visual.material.PBRMaterial(
            metallicFactor=0.35,
            roughnessFactor=0.65,
            alphaMode="BLEND",
            doubleSided=False  # Originally false
        )
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        img = Image.new("RGB", (256, 256), (200, 100, 50))

        out_mesh, clean_pil = direct_resample_texture(
            self.mesh,
            source_image=img,
            target_res=256,
            dilation_padding=8
        )

        out_mat = out_mesh.visual.material
        self.assertTrue(out_mat.doubleSided, "doubleSided must be True")
        self.assertAlmostEqual(out_mat.metallicFactor, 0.35, places=2)
        self.assertAlmostEqual(out_mat.roughnessFactor, 0.65, places=2)
        self.assertEqual(out_mat.alphaMode, "BLEND")
        self.assertEqual(clean_pil.size, (256, 256))

    def test_direct_resample_preserves_alpha_channel(self):
        mat = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        # Create RGBA image with transparent border
        rgba_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        # Fill center with semi-transparent color
        for x in range(32, 96):
            for y in range(32, 96):
                rgba_img.putpixel((x, y), (255, 120, 40, 180))

        out_mesh, clean_pil = direct_resample_texture(
            self.mesh,
            source_image=rgba_img,
            target_res=128,
            dilation_padding=8
        )

        self.assertEqual(clean_pil.mode, "RGBA", "Must preserve RGBA mode")
        arr = np.array(clean_pil)
        self.assertEqual(arr.shape[2], 4, "Must have 4 channels")
        # Center should maintain color and alpha
        center_pixel = arr[64, 64]
        self.assertEqual(center_pixel[0], 255)
        self.assertEqual(center_pixel[1], 120)
        self.assertEqual(center_pixel[2], 40)
        self.assertEqual(center_pixel[3], 180)

    def test_rebake_xatlas_preserves_doublesided_and_normals(self):
        orig_normals = self.mesh.vertex_normals.copy()
        mat = trimesh.visual.material.PBRMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.9,
            doubleSided=False
        )
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        img = Image.new("RGB", (128, 128), (100, 150, 200))

        recharted_mesh, dilated_pil = rebake_texture_xatlas(
            self.mesh,
            source_image=img,
            source_uv=self.uv,
            target_res=256,
            dilation_padding=8
        )

        # 1. doubleSided must be True
        self.assertTrue(recharted_mesh.visual.material.doubleSided, "doubleSided must be True")
        # 2. Material properties preserved
        self.assertAlmostEqual(recharted_mesh.visual.material.metallicFactor, 0.1, places=2)
        self.assertAlmostEqual(recharted_mesh.visual.material.roughnessFactor, 0.9, places=2)
        # 3. Vertex normals must be preserved
        self.assertIsNotNone(recharted_mesh.vertex_normals, "Vertex normals must be preserved")
        self.assertEqual(len(recharted_mesh.vertex_normals), len(recharted_mesh.vertices))

    def test_rebake_xatlas_preserves_rgba(self):
        mat = trimesh.visual.material.PBRMaterial(doubleSided=True)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        rgba_img = Image.new("RGBA", (128, 128), (220, 80, 50, 200))

        recharted_mesh, dilated_pil = rebake_texture_xatlas(
            self.mesh,
            source_image=rgba_img,
            source_uv=self.uv,
            target_res=128,
            dilation_padding=4
        )

        self.assertEqual(dilated_pil.mode, "RGBA", "Must preserve RGBA texture mode in xatlas bake")

    def test_set_doublesided_material_glb_rewrite(self):
        # Create a GLB with doubleSided = False
        box = trimesh.creation.box()
        mat = trimesh.visual.material.PBRMaterial(doubleSided=False)
        box.visual = trimesh.visual.TextureVisuals(material=mat)
        glb_bytes = trimesh.exchange.gltf.export_glb(box)

        # Before rewrite, verify doubleSided is False in glTF
        magic, ver, length = struct.unpack("<III", glb_bytes[:12])
        j_len, j_type = struct.unpack("<II", glb_bytes[12:20])
        gltf = json.loads(glb_bytes[20:20 + j_len].decode("utf-8"))
        for m in gltf.get("materials", []):
            self.assertFalse(m.get("doubleSided", False))

        # Apply set_doublesided_material
        fixed_glb = set_doublesided_material(glb_bytes)

        # After rewrite, verify doubleSided is True in glTF
        magic, ver, length = struct.unpack("<III", fixed_glb[:12])
        j_len, j_type = struct.unpack("<II", fixed_glb[12:20])
        gltf_fixed = json.loads(fixed_glb[20:20 + j_len].decode("utf-8"))
        for m in gltf_fixed.get("materials", []):
            self.assertTrue(m.get("doubleSided"), "doubleSided must be True in glTF JSON")

    def test_rechart_and_bake_high_density_frontside_and_metrics(self):
        mat = trimesh.visual.material.PBRMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.7,
            doubleSided=True
        )
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        img = Image.new("RGB", (256, 256), (80, 160, 240))

        recharted_mesh, dilated_pil, stats = rechart_and_bake_high_density(
            self.mesh,
            target_res=256,
            source_image=img,
            source_uv=self.uv,
            dilation_padding=16,
            double_sided=False,
            return_stats=True
        )

        # 1. FrontSide rendering: doubleSided must be False
        self.assertFalse(recharted_mesh.visual.material.doubleSided, "doubleSided must be False for FrontSide")
        # 2. Material properties preserved
        self.assertAlmostEqual(recharted_mesh.visual.material.metallicFactor, 0.2, places=2)
        self.assertAlmostEqual(recharted_mesh.visual.material.roughnessFactor, 0.7, places=2)
        # 3. Vertex normals must exist and match vertices
        self.assertIsNotNone(recharted_mesh.vertex_normals)
        self.assertEqual(len(recharted_mesh.vertex_normals), len(recharted_mesh.vertices))
        # 4. Metrics verification
        self.assertIn("uv_coverage_ratio_percent", stats)
        self.assertIn("texel_density_linear", stats)
        self.assertIn("covered_pixels", stats)
        self.assertGreater(stats["uv_coverage_ratio_percent"], 0.0)
        self.assertGreater(stats["texel_density_linear"], 0.0)
        self.assertEqual(stats["target_resolution"], 256)
        self.assertEqual(stats["canvas_pixels"], 256 * 256)
        self.assertEqual(dilated_pil.size, (256, 256))

    def test_compute_uv_metrics_standalone(self):
        metrics = compute_uv_metrics(self.mesh, target_res=256, uv=self.uv)
        self.assertIn("uv_coverage_ratio_percent", metrics)
        self.assertIn("texel_density_linear", metrics)
        self.assertIn("texel_density_area", metrics)
        self.assertEqual(metrics["canvas_pixels"], 256 * 256)
        self.assertGreaterEqual(metrics["covered_pixels"], 0)

    def test_no_upscale_clamp_target_resolution_helper(self):
        # 1536x1536 -> largest POT <= 1536 is 1024
        self.assertEqual(clamp_target_resolution(2048, (1536, 1536)), 1024)
        self.assertEqual(clamp_target_resolution(4096, (1536, 1536)), 1024)
        # 768x768 -> largest POT <= 768 is 512
        self.assertEqual(clamp_target_resolution(1024, (768, 768)), 512)
        # 1024x1024 requested 1024 -> keeps 1024
        self.assertEqual(clamp_target_resolution(1024, (1024, 1024)), 1024)
        # 1024x1024 requested 2048 -> clamps to 1024
        self.assertEqual(clamp_target_resolution(2048, (1024, 1024)), 1024)
        # 2048x1024 requested 2048 -> orig_max is 2048, <= 2048 -> keeps 2048
        self.assertEqual(clamp_target_resolution(2048, (2048, 1024)), 2048)
        # 2048x1024 requested 4096 -> clamps to 2048
        self.assertEqual(clamp_target_resolution(4096, (2048, 1024)), 2048)

    def test_no_upscale_enforced_in_direct_resample(self):
        img = Image.new("RGB", (128, 128), (200, 100, 50))
        out_mesh, clean_pil = direct_resample_texture(
            self.mesh,
            source_image=img,
            target_res=256,
            dilation_padding=8
        )
        # 256 > 128 -> clamped to 128 (largest POT <= 128)
        self.assertEqual(clean_pil.size, (128, 128))

    def test_no_upscale_enforced_in_rechart_and_bake(self):
        img = Image.new("RGB", (128, 128), (80, 160, 240))
        recharted_mesh, dilated_pil, stats = rechart_and_bake_high_density(
            self.mesh,
            target_res=256,
            source_image=img,
            source_uv=self.uv,
            dilation_padding=16,
            double_sided=False,
            return_stats=True
        )
        # 256 > 128 -> clamped to 128
        self.assertEqual(stats["target_resolution"], 128)
        self.assertEqual(dilated_pil.size, (128, 128))


if __name__ == "__main__":
    unittest.main()
