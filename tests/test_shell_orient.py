"""
test_shell_orient.py

Unit tests for visibility-based shell orientation (orient_faces_by_visibility).
Verifies:
1. Outward CCW FrontSide winding orientation with 100% faces preserved.
2. Oppositely-wound UV pieces are oriented independently.
3. Per-pixel z-buffer depth.
"""

import unittest
import numpy as np
import trimesh

from optimizer.core.shell_orient import (
    orient_faces_by_visibility,
    _render_single_view_task,
    DEFAULT_VIEWS,
    DEFAULT_RESOLUTION,
)


def create_opposed_uv_pieces_sphere():
    """
    Icosphere split into 8 octant pieces, each with its own copy of its vertices
    (identical positions, different indices, like UV seams). Pieces are wound in a
    checkerboard, so every piece is wound opposite to its neighbours.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    bits = (sphere.triangles_center > 0).astype(np.int64)
    octant = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    verts, faces, offset = [], [], 0
    for g in range(8):
        g_faces = sphere.faces[octant == g]
        used, local = np.unique(g_faces, return_inverse=True)
        local = local.reshape(g_faces.shape)
        if bin(g).count("1") % 2 == 1:
            local = local[:, ::-1]
        verts.append(sphere.vertices[used])
        faces.append(local + offset)
        offset += len(used)
    return np.vstack(verts), np.vstack(faces)


class TestShellOrientAndOcclusionCulling(unittest.TestCase):
    def test_orient_faces_by_visibility_backward_compatibility(self):
        """Existing orient_faces_by_visibility preserves 100% faces."""
        mesh = trimesh.creation.box()
        # Invert box faces
        mesh.faces = mesh.faces[:, ::-1]
        
        stats = {}
        oriented = orient_faces_by_visibility(mesh.vertices, mesh.faces, stats=stats)
        self.assertEqual(len(oriented), len(mesh.faces))
        self.assertGreater(stats["faces_flipped"], 0)
        
        m_oriented = trimesh.Trimesh(vertices=mesh.vertices, faces=oriented, process=False)
        self.assertGreater(m_oriented.volume, 0, "Box should be flipped outward")

    def test_orient_opposed_uv_pieces_independently(self):
        """Oppositely-wound UV pieces sharing seam positions are each oriented outward."""
        vertices, faces = create_opposed_uv_pieces_sphere()
        inward = np.einsum(
            "ij,ij->i",
            np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
            vertices[faces].mean(axis=1),
        ) < 0
        self.assertEqual(int(inward.sum()), len(faces) // 2, "Fixture should start half inverted")

        stats = {}
        oriented = orient_faces_by_visibility(vertices, faces, stats=stats)
        self.assertEqual(len(oriented), len(faces))
        self.assertEqual(stats["faces_after"], len(faces))

        tris = vertices[oriented]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        outward = np.einsum("ij,ij->i", normals, tris.mean(axis=1)) > 0
        self.assertEqual(int((~outward).sum()), 0, "Every face must point outward")

    def test_zbuffer_uses_per_pixel_depth(self):
        """A small triangle behind a tilted large one stays hidden even if its mean depth is closer."""
        # Camera looks from +z. Large triangle lies on plane z = -x (mean depth 0);
        # small triangle at z = 0.3 sits where the large one is at z in [0.4, 0.6].
        vertices = np.array([
            [-1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [0.0, 1.0, 0.0],
            [-0.6, -0.7, 0.3], [-0.4, -0.7, 0.3], [-0.5, -0.5, 0.3],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        ids, _, _ = _render_single_view_task(
            (np.array([0.0, 0.0, 1.0]), vertices, faces, np.zeros(3), 2.0, 64)
        )
        self.assertIn(0, ids)
        self.assertNotIn(1, ids, "Occluded small triangle must not win any pixel")


if __name__ == "__main__":
    unittest.main()
