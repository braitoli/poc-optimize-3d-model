"""
test_uvatlas_wrapper.py

Unit tests for optimizer/core/uvatlas_wrapper.py:
- uvatlas_unwrap on manifold mesh
- is_mesh_manifold diagnostics
- Fallback behavior on non-manifold geometry
- Exact 3D geometry preservation
"""

import unittest
import numpy as np
import trimesh

from optimizer.core.uvatlas_wrapper import uvatlas_unwrap, is_mesh_manifold


class TestUVAtlasWrapper(unittest.TestCase):
    def test_uvatlas_unwrap_sphere(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        v = sphere.vertices
        f = sphere.faces

        vmapping, new_faces, new_uvs, meta = uvatlas_unwrap(v, f, target_res=512)

        self.assertEqual(meta["engine"], "uvatlas")
        self.assertFalse(meta["fallback_used"])
        self.assertEqual(len(new_faces), len(f), "All faces preserved")
        self.assertEqual(len(vmapping), len(new_uvs), "Vertex count matches UV count")

        # Range of UVs strictly [0, 1]
        self.assertGreaterEqual(float(new_uvs.min()), 0.0)
        self.assertLessEqual(float(new_uvs.max()), 1.0)

        # Reconstructed 3D geometry matches original 100%
        reconstructed_verts = v[vmapping]
        diff = np.max(np.abs(reconstructed_verts[new_faces] - v[f]))
        self.assertEqual(diff, 0.0, "3D geometry must match original mesh exactly")

    def test_is_mesh_manifold(self):
        # Manifold box
        box = trimesh.creation.box()
        is_m, diag = is_mesh_manifold(box.vertices, box.faces)
        self.assertTrue(is_m)
        self.assertEqual(diag["non_manifold_edges_count"], 0)
        self.assertEqual(diag["non_manifold_vertices_count"], 0)

        # Synthetic non-manifold mesh (3 faces sharing edge 0-1)
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4]
        ], dtype=np.int64)

        is_m_bad, diag_bad = is_mesh_manifold(verts, faces)
        self.assertFalse(is_m_bad)
        self.assertGreater(diag_bad["non_manifold_edges_count"], 0)

    def test_fallback_on_non_manifold_mesh(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 1, 0], [0, 0, 1]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4]
        ], dtype=np.int64)

        # With fallback_to_xatlas=True, unwrap succeeds via xatlas
        vmap, new_faces, new_uvs, meta = uvatlas_unwrap(
            verts, faces, target_res=256, fallback_to_xatlas=True
        )
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["engine"], "xatlas")
        self.assertEqual(len(new_faces), 3)

        # With fallback_to_xatlas=False, raises RuntimeError
        with self.assertRaises(RuntimeError):
            uvatlas_unwrap(verts, faces, target_res=256, fallback_to_xatlas=False)


if __name__ == "__main__":
    unittest.main()
