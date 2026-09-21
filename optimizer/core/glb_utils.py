"""
glb_utils.py

Binary GLB helpers: material doubleSided rewriting/detection and
EXT_meshopt_compression decompression for trimesh compatibility.
"""

import json
import subprocess
from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_frontside_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is false for all materials in a binary GLB."""
    import struct
    if len(glb_bytes) < 20:
        return glb_bytes

    magic, ver, length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF":
        return glb_bytes

    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        return glb_bytes

    json_bytes = glb_bytes[20:20 + chunk_len]
    gltf = json.loads(json_bytes.decode("utf-8"))

    modified = False
    if "materials" in gltf:
        for mat in gltf["materials"]:
            if mat.get("doubleSided") is not False:
                mat["doubleSided"] = False
                modified = True

    if not modified:
        return glb_bytes

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + chunk_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    out = bytearray()
    out.extend(struct.pack("<4sII", magic, ver, new_total_len))
    out.extend(struct.pack("<I4s", len(new_json_bytes), b"JSON"))
    out.extend(new_json_bytes)
    out.extend(bin_chunk)
    return bytes(out)


def _decompress_meshopt_if_needed(input_path: Path, tmp_dir: Path) -> Path:
    """If input GLB has EXT_meshopt_compression, decompress it using Node.js for trimesh compatibility."""
    import struct
    try:
        with open(input_path, "rb") as f:
            header = f.read(12)
            if len(header) == 12:
                magic, ver, length = struct.unpack("<4sII", header)
                if magic == b"glTF":
                    chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
                    if chunk_type == b"JSON":
                        gltf = json.loads(f.read(chunk_len))
                        exts = gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", [])
                        if "EXT_meshopt_compression" in exts:
                            uncompressed_path = tmp_dir / f"unpacked_{input_path.name}"
                            node_script = f"""
import {{ NodeIO }} from '@gltf-transform/core';
import {{ ALL_EXTENSIONS }} from '@gltf-transform/extensions';
import {{ MeshoptDecoder }} from 'meshoptimizer';
import fs from 'fs';

async function decompress() {{
    await MeshoptDecoder.ready;
    const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({{ 'meshopt.decoder': MeshoptDecoder }});
    const doc = await io.read({json.dumps(str(input_path.resolve()))});
    const ext = doc.getRoot().listExtensionsUsed().find(e => e.extensionName === 'EXT_meshopt_compression');
    if (ext) ext.dispose();
    const glb = await io.writeBinary(doc);
    fs.writeFileSync({json.dumps(str(uncompressed_path.resolve()))}, glb);
}}
decompress();
"""
                            subprocess.run(
                                ["node", "-e", node_script],
                                check=True,
                                cwd=str(REPO_ROOT),
                                capture_output=True
                            )
                            if uncompressed_path.exists():
                                return uncompressed_path
    except Exception:
        pass
    return input_path


def set_doublesided_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is true for all materials in a binary GLB."""
    import struct
    if len(glb_bytes) < 20:
        return glb_bytes

    magic, ver, length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF":
        return glb_bytes

    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        return glb_bytes

    json_bytes = glb_bytes[20:20 + chunk_len]
    gltf = json.loads(json_bytes.decode("utf-8"))

    modified = False
    if "materials" in gltf and gltf["materials"]:
        for mat in gltf["materials"]:
            if mat.get("doubleSided") is not True:
                mat["doubleSided"] = True
                modified = True
    else:
        gltf["materials"] = [{"name": "default_material", "doubleSided": True}]
        for mesh in gltf.get("meshes", []):
            for prim in mesh.get("primitives", []):
                if "material" not in prim:
                    prim["material"] = 0
        modified = True

    if not modified:
        return glb_bytes

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + chunk_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    out = bytearray()
    out.extend(struct.pack("<4sII", magic, ver, new_total_len))
    out.extend(struct.pack("<I4s", len(new_json_bytes), b"JSON"))
    out.extend(new_json_bytes)
    out.extend(bin_chunk)
    return bytes(out)


def check_glb_double_sided(glb_input: Union[bytes, Path, str]) -> bool:
    """Checks if any material in a GLB has doubleSided=True."""
    import struct
    try:
        if isinstance(glb_input, (str, Path)):
            with open(glb_input, "rb") as f:
                header = f.read(20)
                if len(header) < 20:
                    return False
                magic, ver, length = struct.unpack("<4sII", header[:12])
                if magic != b"glTF":
                    return False
                chunk_len, chunk_type = struct.unpack("<I4s", header[12:20])
                if chunk_type != b"JSON":
                    return False
                json_bytes = f.read(chunk_len)
        else:
            if len(glb_input) < 20:
                return False
            magic, ver, length = struct.unpack("<4sII", glb_input[:12])
            if magic != b"glTF":
                return False
            chunk_len, chunk_type = struct.unpack("<I4s", glb_input[12:20])
            if chunk_type != b"JSON":
                return False
            json_bytes = glb_input[20:20 + chunk_len]

        gltf = json.loads(json_bytes.decode("utf-8"))
        for mat in gltf.get("materials", []):
            if mat.get("doubleSided") is True:
                return True
    except Exception:
        pass
    return False
