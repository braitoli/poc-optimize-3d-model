"""
cleaner.py

Geometric cleaning and auto-grounding for 3D statues.
Strictly adheres to Rule 11 (Zero-Decimation Policy):
- Preserves 100% geometric triangles.
- Rejects non-finite vertices (PipelineAbort) and removes unreferenced artifacts.
- Grounds base at Y=0 and centers on X/Z axes.
"""

from typing import Tuple
import numpy as np
import trimesh

from optimizer.core.errors import PipelineAbort


def _preserve_texture_attributes(src: trimesh.Trimesh, dst: trimesh.Trimesh) -> None:
    """Preserves image formats and fast-save hooks across trimesh mesh copying."""
    if not hasattr(src, "visual") or not hasattr(dst, "visual"):
        return
    src_mat = getattr(src.visual, "material", None)
    dst_mat = getattr(dst.visual, "material", None)
    if not src_mat or not dst_mat:
        return
    for attr in [
        "baseColorTexture",
        "image",
        "metallicRoughnessTexture",
        "normalTexture",
        "emissiveTexture",
        "occlusionTexture"
    ]:
        src_img = getattr(src_mat, attr, None)
        dst_img = getattr(dst_mat, attr, None)
        if src_img is not None and dst_img is not None:
            if hasattr(src_img, "format") and src_img.format:
                dst_img.format = src_img.format
            if hasattr(src_img, "_fast_save_data"):
                dst_img._fast_save_data = src_img._fast_save_data
                dst_img.save = src_img.save


def clean_and_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Cleans mesh geometry while strictly preserving 100% valid triangles.
    Removes unreferenced vertices and fixes basic winding. A non-finite vertex raises PipelineAbort
    before anything changes: dropping its faces would break the zero-decimation rule.
    """
    m = mesh.copy()
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)

    non_finite = int(np.count_nonzero(~np.isfinite(np.asarray(m.vertices)).all(axis=1)))
    if non_finite:
        raise PipelineAbort(f"mesh has {non_finite} non-finite vertex coordinates")

    _preserve_texture_attributes(mesh, m)

    # Rule 11 Zero-Decimation: Strictly preserve 100% faces (do not drop any face)
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_winding(m)

    return m


def auto_ground_and_center(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Translates mesh so that its bottom rests flat on Y=0 and its bounding box
    is centered at X=0, Z=0.
    Returns (grounded_mesh, translation_vector).
    """
    m = mesh.copy()
    _preserve_texture_attributes(mesh, m)
    bounds = m.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    min_x, min_y, min_z = bounds[0]
    max_x, _, max_z = bounds[1]

    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    shift_y = -min_y

    translation = np.array([-center_x, shift_y, -center_z], dtype=np.float64)
    m.apply_translation(translation)

    return m, translation

