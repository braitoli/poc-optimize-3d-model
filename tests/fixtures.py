"""
fixtures.py

Shared test fixtures. Also used by the CI smoke test:
    python3 -c "from tests.fixtures import create_mock_glb; ..."
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def create_mock_glb(path: Path, subdivisions: int = 2) -> int:
    """Writes a textured icosphere GLB (spherical UVs, 256x256 texture) and returns its face count."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)

    img = Image.new("RGB", (256, 256), (100, 180, 70))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 90, 90], fill=(220, 50, 40))
    draw.ellipse([110, 110, 200, 200], fill=(40, 80, 220))

    norm_v = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    u = 0.5 + np.arctan2(norm_v[:, 0], norm_v[:, 2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(norm_v[:, 1]) / np.pi
    uv = np.column_stack([u, v])

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    scene = trimesh.Scene({"Model": mesh})
    Path(path).write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    return len(mesh.faces)


# Source UVs of the sphere grid are squeezed into [GRID_U0, GRID_U0 + GRID_SPAN]^2 of the texture
GRID_U0, GRID_SPAN = 0.30, 0.40


def make_sphere_grid_case(src_res: int):
    """6x6 grid of small icospheres (11,520 faces, ~200 charts so the packer's minimum padding is
    actually reached) with planar-projected source UVs (front and back faces share UV space, like
    mirrored UVs) and a 3-stripe src_res x src_res source texture.
    Returns (mesh, texture, uv)."""
    parts = []
    for i in range(6):
        for j in range(6):
            part = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
            part.apply_translation([i, j, 0.0])
            parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    xy = np.asarray(mesh.vertices)[:, :2]
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    uv = GRID_U0 + xy * GRID_SPAN
    # Red / green / blue vertical stripes across the UV region. Every source texel and every
    # bilinear blend of neighbouring stripes has at least one zero channel, whereas the flat
    # canvas background (mean sampled surface colour) has all three channels > 0.
    img = np.zeros((src_res, src_res, 3), dtype=np.uint8)
    xs = np.arange(src_res) / float(src_res - 1)
    img[:, xs < GRID_U0 + GRID_SPAN / 3.0, 0] = 255
    img[:, (xs >= GRID_U0 + GRID_SPAN / 3.0) & (xs < GRID_U0 + 2.0 * GRID_SPAN / 3.0), 1] = 255
    img[:, xs >= GRID_U0 + 2.0 * GRID_SPAN / 3.0, 2] = 255
    return mesh, Image.fromarray(img, mode="RGB"), uv


def create_sphere_grid_glb(path: Path, src_res: int) -> int:
    """Writes make_sphere_grid_case(src_res) as a textured GLB and returns its face count."""
    mesh, img, uv = make_sphere_grid_case(src_res)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.7)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    scene = trimesh.Scene({"Model": mesh})
    Path(path).write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    return len(mesh.faces)
