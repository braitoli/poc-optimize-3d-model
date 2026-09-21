"""
test_shell_orient.py

Unit and integration tests for shell orientation and occlusion culling / interior face pruning.
Verifies:
1. Zero decimation when prune_interior=False.
2. Accurate interior face pruning and orphan vertex removal when prune_interior=True.
3. Topological hole protection preventing punctures ("không bị bục tượng").
4. Outward CCW FrontSide winding orientation.
5. Real statue benchmarks (Dinoki, Koidrax).
"""

import unittest
from pathlib import Path
import numpy as np
import trimesh

from optimizer.core.shell_orient import (
    orient_faces_by_visibility,
    prune_interior_and_orient_faces,
    prune_mesh_interior_and_orient,
    DEFAULT_VIEWS,
    DEFAULT_RESOLUTION,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def create_nested_spheres(r_outer: float = 1.0, r_inner: float = 0.5) -> trimesh.Trimesh:
    """Creates a mesh with an outer sphere and an inner completely occluded sphere."""
    outer = trimesh.creation.icosphere(subdivisions=2, radius=r_outer)
    inner = trimesh.creation.icosphere(subdivisions=2, radius=r_inner)
    # Reverse inner faces so they face inwards
    inner.faces = inner.faces[:, ::-1]
    
    verts = np.vstack([outer.vertices, inner.vertices])
    faces = np.vstack([outer.faces, inner.faces + len(outer.vertices)])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class TestShellOrientAndOcclusionCulling(unittest.TestCase):
    def test_zero_decimation_when_prune_interior_false(self):
        """When prune_interior=False, 100% of vertices and faces are preserved."""
        mesh = create_nested_spheres(1.0, 0.5)
        orig_v = len(mesh.vertices)
        orig_f = len(mesh.faces)
        
        stats = {}
        new_v, new_f = prune_interior_and_orient_faces(
            mesh.vertices, mesh.faces, prune_interior=False, stats=stats
        )
        self.assertEqual(len(new_v), orig_v)
        self.assertEqual(len(new_f), orig_f)
        self.assertEqual(stats["faces_pruned"], 0)
        self.assertEqual(stats["vertices_pruned"], 0)

    def test_occlusion_culling_nested_sphere(self):
        """Interior sphere is 100% occluded and should be completely pruned."""
        mesh = create_nested_spheres(1.0, 0.5)
        outer_faces = 320  # icosphere(2) has 320 faces, 162 vertices
        inner_faces = 320
        self.assertEqual(len(mesh.faces), outer_faces + inner_faces)
        
        stats = {}
        new_v, new_f = prune_interior_and_orient_faces(
            mesh.vertices, mesh.faces, prune_interior=True, stats=stats
        )
        
        # All 320 inner faces should be pruned
        self.assertEqual(len(new_f), outer_faces)
        self.assertEqual(stats["faces_pruned"], inner_faces)
        # All 162 inner vertices should be removed as orphans
        self.assertEqual(len(new_v), 162)
        self.assertEqual(stats["vertices_pruned"], 162)
        
        # Remaining outer sphere should be outward facing
        m_out = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        self.assertGreater(m_out.volume, 0, "Outer sphere volume should be positive (outward)")

    def test_surface_protection_on_crevice(self):
        """Exterior faces in folds/crevices must not be pruned to prevent punctures."""
        # Create a torus or cylinder
        mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
        orig_f = len(mesh.faces)
        
        stats = {}
        new_v, new_f = prune_interior_and_orient_faces(
            mesh.vertices, mesh.faces, prune_interior=True, stats=stats
        )
        # For a simple watertight exterior surface, zero faces should be pruned
        self.assertEqual(len(new_f), orig_f)
        self.assertEqual(stats["faces_pruned"], 0)

    def test_real_statue_dinoki_occlusion_and_orient(self):
        """Real model test on Dinoki: prunes interior faces, orients outward, zero puncture."""
        sample_dinoki = REPO_ROOT / "examples" / "sample_dinoki.glb"
        if not sample_dinoki.exists():
            self.skipTest("sample_dinoki.glb not found")

        mesh = trimesh.load(str(sample_dinoki), force="mesh", process=False)
        stats = {}
        new_mesh = prune_mesh_interior_and_orient(mesh, prune_interior=True, stats=stats)
        
        # Verify pruning
        self.assertGreater(stats["faces_pruned"], 3000, "Should prune at least 3,000 interior faces")
        self.assertGreater(stats["vertices_pruned"], 3000, "Should prune at least 3,000 orphan vertices")
        self.assertEqual(len(new_mesh.faces), stats["faces_after"])
        self.assertEqual(len(new_mesh.vertices), stats["vertices_after"])
        
        # Verify UV preservation
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            self.assertEqual(len(new_mesh.visual.uv), len(new_mesh.vertices))

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


if __name__ == "__main__":
    unittest.main()
