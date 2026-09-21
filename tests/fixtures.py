"""
fixtures.py

Shared test fixtures. Also used by the CI smoke test:
    python3 -c "from tests.fixtures import create_mock_glb; ..."
"""

import io
import json
import struct
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
    texture = Image.fromarray(img, mode="RGB")
    # The baker carries the source glTF PBR material over (it refuses anything else)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=0.7)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return mesh, texture, uv


def create_sphere_grid_glb(path: Path, src_res: int) -> int:
    """Writes make_sphere_grid_case(src_res) as a textured GLB and returns its face count."""
    mesh, img, uv = make_sphere_grid_case(src_res)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.7)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    scene = trimesh.Scene({"Model": mesh})
    Path(path).write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    return len(mesh.faces)


def image_bytes(size=(64, 32), fmt: str = "PNG", mode: str = "RGB") -> bytes:
    """Encoded bytes of a gradient image (so lossy encoders keep some structure)."""
    w, h = size
    arr = np.zeros((h, w, len(mode)), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    if len(mode) > 1:
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    if len(mode) > 2:
        arr[..., 2] = 90
    if len(mode) > 3:
        arr[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(arr if len(mode) > 1 else arr[..., 0], mode=mode).save(buf, format=fmt)
    return buf.getvalue()


def quad(offset=(0.0, 0.0, 0.0), uv: bool = True, material=None) -> dict:
    """One unit-quad primitive (4 vertices, 2 triangles) for build_glb."""
    prim = {
        "positions": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        + np.asarray(offset, dtype=np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    }
    if uv:
        prim["uv"] = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    if material is not None:
        prim["material"] = material
    return prim


def textured_material(texture: int = 0) -> dict:
    return {"pbrMetallicRoughness": {"baseColorTexture": {"index": texture}, "metallicFactor": 0.0, "roughnessFactor": 0.7}}


def build_glb(path: Path, meshes=(), images=(), materials=(), nodes=None, extensions_used=(), extensions_required=()) -> None:
    """Writes a minimal GLB by hand, for inputs trimesh would refuse to export (NaN vertices, no UVs,
    several primitives, arbitrary image bytes / MIME types).
    meshes: list of meshes, each a list of primitives {"positions", "indices", "uv"?, "material"?}.
    images: (encoded bytes, MIME type) pairs; texture i samples image i.
    nodes: mesh index per scene node (default: one node per mesh).
    POSITION min/max cover the finite rows only, so a NaN vertex still gives valid JSON."""
    chunks, views, accessors = [], [], []

    def view(data: bytes, target=None) -> int:
        entry = {"buffer": 0, "byteOffset": sum(len(c) for c in chunks), "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        chunks.append(data + b"\0" * (-len(data) % 4))
        views.append(entry)
        return len(views) - 1

    def accessor(arr: np.ndarray, type_: str, component_type: int, target: int, bounds: bool = False) -> int:
        entry = {"bufferView": view(arr.tobytes(), target), "componentType": component_type, "count": len(arr), "type": type_}
        if bounds:
            finite = arr[np.isfinite(arr).all(axis=1)]
            entry["min"], entry["max"] = finite.min(axis=0).tolist(), finite.max(axis=0).tolist()
        accessors.append(entry)
        return len(accessors) - 1

    gltf_meshes = []
    for prims in meshes:
        out = []
        for p in prims:
            attrs = {"POSITION": accessor(np.asarray(p["positions"], dtype=np.float32), "VEC3", 5126, 34962, bounds=True)}
            if p.get("uv") is not None:
                attrs["TEXCOORD_0"] = accessor(np.asarray(p["uv"], dtype=np.float32), "VEC2", 5126, 34962)
            entry = {"attributes": attrs, "indices": accessor(np.asarray(p["indices"], dtype=np.uint32), "SCALAR", 5125, 34963)}
            if p.get("material") is not None:
                entry["material"] = p["material"]
            out.append(entry)
        gltf_meshes.append({"primitives": out})

    node_meshes = list(range(len(gltf_meshes))) if nodes is None else list(nodes)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(node_meshes)))}],
        "nodes": [{"mesh": m} for m in node_meshes],
    }
    if gltf_meshes:
        gltf["meshes"] = gltf_meshes
    if images:
        gltf["images"] = [{"bufferView": view(data), "mimeType": mime} for data, mime in images]
        gltf["textures"] = [{"source": i} for i in range(len(images))]
    if materials:
        gltf["materials"] = list(materials)
    if extensions_used:
        gltf["extensionsUsed"] = list(extensions_used)
    if extensions_required:
        gltf["extensionsRequired"] = list(extensions_required)
    gltf["accessors"] = accessors
    gltf["bufferViews"] = views
    bin_chunk = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    if not accessors and not views:
        del gltf["accessors"], gltf["bufferViews"], gltf["buffers"]

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    out = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    if bin_chunk:
        out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(out)) + out)
