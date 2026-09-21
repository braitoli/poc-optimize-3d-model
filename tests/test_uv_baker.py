"""
test_uv_baker.py

Tests for:
- optimizer/core/uv_baker.py (rechart_and_bake_high_density)
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
    rechart_and_bake_high_density,
    maximize_uv_bounds,
    get_adaptive_chart_options
)
from optimizer.core.glb_utils import set_doublesided_material


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

    def test_rechart_preserves_rgba(self):
        mat = trimesh.visual.material.PBRMaterial(doubleSided=True)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
        rgba_img = Image.new("RGBA", (128, 128), (220, 80, 50, 200))

        recharted_mesh, dilated_pil = rechart_and_bake_high_density(
            self.mesh,
            source_image=rgba_img,
            source_uv=self.uv,
            dilation_padding=4,
            double_sided=True
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
            source_image=img,
            source_uv=self.uv,
            dilation_padding=16,
            double_sided=False,
            return_stats=True
        )
        res = stats["final_resolution"]

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
        self.assertEqual(stats["size_mode"], "exact")
        self.assertEqual(res % 4, 0)
        self.assertEqual(stats["target_resolution"], res)
        self.assertEqual(stats["canvas_pixels"], res * res)
        self.assertEqual(dilated_pil.size, (res, res))
        self.assertGreaterEqual(stats["texel_density_ratio"], 0.97)

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


if __name__ == "__main__":
    unittest.main()
