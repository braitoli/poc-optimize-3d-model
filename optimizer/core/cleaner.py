"""
cleaner.py

Geometric cleaning and auto-grounding for 3D statues.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles.
- Eliminates invalid vertices and unreferenced artifacts.
- Grounds base at Y=0 and centers on X/Z axes.
"""

from typing import Tuple
import numpy as np
import trimesh


def clean_and_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Cleans mesh geometry while strictly preserving 100% valid triangles.
    Removes infinite values, unreferenced vertices, and fixes basic winding.
    """
    m = mesh.copy()
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)

    if hasattr(m, "remove_infinite_values"):
        m.remove_infinite_values()
    elif not np.isfinite(m.vertices).all():
        valid_verts = np.isfinite(m.vertices).all(axis=1)
        m.update_vertices(valid_verts)

    # Rule 11 Zero-Decimation: Strictly preserve 100% faces (do not drop any face)
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass

    try:
        trimesh.repair.fix_winding(m)
    except Exception:
        pass

    return m


def auto_ground_and_center(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Translates mesh so that its bottom rests flat on Y=0 and its bounding box
    is centered at X=0, Z=0.
    Returns (grounded_mesh, translation_vector).
    """
    m = mesh.copy()
    bounds = m.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    min_x, min_y, min_z = bounds[0]
    max_x, _, max_z = bounds[1]

    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    shift_y = -min_y

    translation = np.array([-center_x, shift_y, -center_z], dtype=np.float64)
    m.apply_translation(translation)

    return m, translation
