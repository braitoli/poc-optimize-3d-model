"""
test_density_reference.py

What Step 4 measures the source texture's density against when Step 3 collapsed the edges.

The pre-reduction mesh is the wrong reference: it still holds the hidden interior shell the
removals cut away, and on an AI-generated model that shell owns most of the source atlas (dinoki:
85% of the atlas for 51% of the surface). Sizing the canvas for it means baking the visible faces
at a density the source never had for them - a bigger texture holding no extra detail.

reduce_faces therefore hands back the mesh as the removals left it, source UVs intact.
"""

import unittest

import numpy as np
import trimesh

from optimizer.core.face_reduce import reduce_faces


def subdivided_box(times=4):
    """An over-tessellated box: its faces are coplanar, so a collapse costs almost no deviation
    and the merge search is certain to find one."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    for _ in range(times):
        mesh = mesh.subdivide()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    span = vertices.max(axis=0) - vertices.min(axis=0)
    uv = (vertices[:, :2] - vertices[:, :2].min(axis=0)) / span[:2]
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh


class TestPreCollapseMesh(unittest.TestCase):
    def setUp(self):
        self.mesh = subdivided_box()

    def reduce(self, ops, pre_collapse):
        stats = {}
        result = reduce_faces(
            self.mesh, engine="cgal", ops=ops, quality_budget_percent=5.0,
            isolated_min_faces=1, stats=stats, pre_collapse=pre_collapse
        )
        return result, stats

    def test_a_collapse_hands_back_the_mesh_the_removals_left(self):
        out = {}
        result, stats = self.reduce(["repair", "merge"], out)
        if not stats["uvInvalidated"]:
            self.skipTest("the merge search collapsed nothing on this fixture")
        self.assertIn("mesh", out, "a collapse must hand back its pre-collapse mesh")
        pre = out["mesh"]
        # Source UVs still on it, one per vertex: that is what makes it usable as the reference
        self.assertIsNotNone(getattr(pre.visual, "uv", None))
        self.assertEqual(len(pre.visual.uv), len(pre.vertices))
        # The removals already happened, the collapse has not
        self.assertLessEqual(len(pre.faces), stats["facesBefore"])
        self.assertGreater(len(pre.faces), len(result.faces))

    def test_no_collapse_hands_back_nothing(self):
        out = {}
        _, stats = self.reduce(["repair"], out)
        self.assertFalse(stats["uvInvalidated"])
        self.assertEqual(out, {}, "without a collapse the caller keeps using the reduced mesh")

    def test_the_reference_excludes_what_the_removals_cut(self):
        """The whole point: the reference must not carry the faces that are gone."""
        out = {}
        result, stats = self.reduce(["repair", "hidden", "merge"], out)
        if not stats["uvInvalidated"]:
            self.skipTest("the merge search collapsed nothing on this fixture")
        removed_by_ops = sum(stats["removed"].values())
        self.assertEqual(len(out["mesh"].faces), stats["facesBefore"] - removed_by_ops)


if __name__ == "__main__":
    unittest.main()
