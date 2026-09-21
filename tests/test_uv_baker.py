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
    compute_uv_metrics,
    can_downscale_texture as uv_baker_can_downscale,
    maximize_uv_bounds,
    get_adaptive_chart_options
)
from optimizer.core.texture_utils import (
    clamp_target_resolution,
    can_downscale_texture,
    maximize_uv_space
)
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

    def test_clamp_target_resolution_auto(self):
        # 'auto' or 'AUTO' computes largest POT <= orig_max
        self.assertEqual(clamp_target_resolution("auto", (1536, 1536)), 1024)
        self.assertEqual(clamp_target_resolution("AUTO", (4096, 4096)), 4096)
        self.assertEqual(clamp_target_resolution("auto", (2048, 1024)), 2048)
        self.assertEqual(clamp_target_resolution("auto", (512, 512)), 512)

    def test_can_downscale_texture_evaluation(self):
        # 4K texture with box mesh: should downscale to 2048
        img_4k = Image.new("RGB", (4096, 4096), (100, 150, 200))
        can_down, optimal_res, details = can_downscale_texture(
            self.mesh,
            source_image=img_4k,
            uv=self.uv,
            target_res="auto",
            min_texel_density=64.0
        )
        self.assertTrue(can_down, "4K texture should be downscaled for web performance")
        self.assertEqual(optimal_res, 2048)
        self.assertEqual(details["candidate_res"], 2048)
        self.assertEqual(details["base_res"], 4096)

        # 512 texture: should NOT downscale below 512
        img_512 = Image.new("RGB", (512, 512), (100, 150, 200))
        can_down_512, optimal_res_512, details_512 = can_downscale_texture(
            self.mesh,
            source_image=img_512,
            uv=self.uv,
            target_res="auto",
            min_texel_density=200.0
        )
        self.assertFalse(can_down_512, "512 texture should not be downscaled below 512")
        self.assertEqual(optimal_res_512, 512)

    def test_maximize_uv_space_subregion(self):
        # Mesh with UVs only occupying [0.2, 0.2] to [0.6, 0.6]
        sub_uv = np.array([
            [0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6],
            [0.3, 0.3], [0.5, 0.3], [0.5, 0.5], [0.3, 0.5]
        ], dtype=np.float64)
        mat = trimesh.visual.material.PBRMaterial()
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=sub_uv, material=mat)
        img = Image.new("RGB", (256, 256), (150, 100, 50))

        out_mesh, cropped_img, new_uv, info = maximize_uv_space(
            self.mesh,
            source_image=img,
            uv=sub_uv,
            target_res=256
        )

        self.assertTrue(info["adjusted"], "UV space should be adjusted when sub-rectangle has margins")
        self.assertAlmostEqual(new_uv.min(), 0.0, places=1)
        self.assertAlmostEqual(new_uv.max(), 1.0, places=1)
        self.assertEqual(cropped_img.size, (256, 256))

    def test_maximize_uv_space_already_maximized(self):
        # UVs already spanning full [0, 1]
        full_uv = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
        ], dtype=np.float64)
        img = Image.new("RGB", (256, 256), (150, 100, 50))

        _, _, _, info = maximize_uv_space(
            self.mesh,
            source_image=img,
            uv=full_uv,
            target_res=256
        )
    def test_uv_baker_can_downscale_threshold_logic(self):
        # When TD_new_downscaled >= 0.85 * TD_orig: downscale 4096 -> 2048
        # Simulate mesh with surface area 10.0
        # Old UV with small coverage (0.15) on 4096:
        # TD_orig = 0.15 * 4096^2 / 10.0 = 251,658.24
        # New UV with high coverage (0.80) on 2048:
        # TD_downscaled = 0.80 * 2048^2 / 10.0 = 335,544.32 (335544.32 / 251658.24 = 1.33 >= 0.85) -> CÓ THỂ DOWNSCALE
        sparse_uv = np.array([
            [0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3],
            [0.15, 0.15], [0.25, 0.15], [0.25, 0.25], [0.15, 0.25]
        ], dtype=np.float64)
        full_new_uv = np.array([
            [0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
        ], dtype=np.float64)

        can_down, target_res, details = uv_baker_can_downscale(
            self.mesh,
            current_res=4096,
            old_uv=sparse_uv,
            new_uv=full_new_uv,
            min_res=1024,
            td_threshold_ratio=0.85
        )
        self.assertTrue(can_down)
        self.assertEqual(target_res, 2048)
        self.assertEqual(details["originalResolution"], "4096x4096")
        self.assertEqual(details["finalResolution"], "2048x2048")
        self.assertIn("texelDensityDelta", details)
        self.assertIn("uvCoverageRatio", details)

    def test_uv_baker_can_downscale_rejected_when_td_too_low(self):
        # When TD would drop below 0.85 * TD_orig: keep current resolution
        # Old UV already high coverage (0.75) on 2048:
        # Downscaling to 1024 would drop TD to ~1/4 * (0.80/0.75) = 26% of TD_orig < 85%
        packed_uv = np.array([
            [0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
        ], dtype=np.float64)

        can_down, target_res, details = uv_baker_can_downscale(
            self.mesh,
            current_res=2048,
            old_uv=packed_uv,
            new_uv=packed_uv,
            min_res=1024,
            td_threshold_ratio=0.85
        )
        self.assertFalse(can_down)
        self.assertEqual(target_res, 2048)
        self.assertEqual(details["finalResolution"], "2048x2048")

    def test_uv_baker_can_downscale_rejected_for_small_resolutions(self):
        # Resolutions <= 1024 should not downscale further
        full_uv = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
        ], dtype=np.float64)
        can_down, target_res, details = uv_baker_can_downscale(
            self.mesh,
            current_res=1024,
            old_uv=full_uv,
            new_uv=full_uv,
            min_res=1024
        )
        self.assertFalse(can_down)
        self.assertEqual(target_res, 1024)

    def test_maximize_uv_bounds_scaling_and_margin(self):
        # UVs occupying only [0.2, 0.3] to [0.7, 0.8]
        sub_uv = np.array([
            [0.2, 0.3], [0.7, 0.3], [0.7, 0.8], [0.2, 0.8]
        ], dtype=np.float64)
        target_res = 1024
        padding_px = 4
        margin = 4.0 / 1024.0

        maximized = maximize_uv_bounds(sub_uv, target_res=target_res, padding_px=padding_px)

        # Minimum must be >= margin
        self.assertGreaterEqual(float(maximized[:, 0].min()), margin - 1e-6)
        self.assertGreaterEqual(float(maximized[:, 1].min()), margin - 1e-6)
        # Maximum must be <= 1.0 - margin
        self.assertLessEqual(float(maximized[:, 0].max()), 1.0 - margin + 1e-6)
        self.assertLessEqual(float(maximized[:, 1].max()), 1.0 - margin + 1e-6)
        # Bounding box should span wider than original
        self.assertGreater(float(maximized[:, 0].max() - maximized[:, 0].min()), 0.5)

    def test_get_adaptive_chart_options(self):
        # Large mesh (> 100k faces) -> fast single iteration, cost=4.0
        large_opts = get_adaptive_chart_options(150_000)
        self.assertEqual(large_opts.max_iterations, 1)
        self.assertEqual(large_opts.max_cost, 4.0)

        # Medium mesh (50k faces) -> 2 iterations
        med_opts = get_adaptive_chart_options(50_000)
        self.assertEqual(med_opts.max_iterations, 2)
        self.assertEqual(med_opts.max_cost, 3.0)

        # Small mesh (< 40k faces) -> 4 iterations
        small_opts = get_adaptive_chart_options(5_000)
        self.assertEqual(small_opts.max_iterations, 4)
        self.assertEqual(small_opts.max_cost, 2.0)

    def test_rechart_and_bake_high_density_downscale_and_metrics(self):
        # Sparse UVs on 4096 texture: should downscale to 2048 and report metrics
        sparse_uv = np.array([
            [0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3],
            [0.15, 0.15], [0.25, 0.15], [0.25, 0.25], [0.15, 0.25]
        ], dtype=np.float64)
        mat = trimesh.visual.material.PBRMaterial(doubleSided=True)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=sparse_uv, material=mat)
        img_4k = Image.new("RGB", (4096, 4096), (70, 140, 210))

        recharted_mesh, dilated_pil, stats = rechart_and_bake_high_density(
            self.mesh,
            target_res=4096,
            source_image=img_4k,
            source_uv=sparse_uv,
            dilation_padding=16,
            double_sided=False,
            return_stats=True
        )

        # 1. FrontSide rendering: doubleSided must be False
        self.assertFalse(recharted_mesh.visual.material.doubleSided)
        # 2. Adaptive downscale occurred
        self.assertTrue(stats["downscaled"])
        self.assertEqual(stats["originalResolution"], "4096x4096")
        self.assertEqual(stats["finalResolution"], "2048x2048")
        self.assertEqual(dilated_pil.size, (2048, 2048))
        # 3. Required metric fields present
        self.assertIn("uvCoverageRatio", stats)
        self.assertIn("texelDensityDelta", stats)
        self.assertGreater(stats["uvCoverageRatio"], 0.0)


if __name__ == "__main__":
    unittest.main()
