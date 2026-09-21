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
