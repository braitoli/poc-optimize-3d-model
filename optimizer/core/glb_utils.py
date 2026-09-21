"""
glb_utils.py

Binary GLB helpers: reading (JSON + BIN chunks), material doubleSided rewriting/detection and
EXT_meshopt_compression decompression for trimesh compatibility.
"""

import json
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from optimizer.core.errors import PipelineAbort

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_glb_json(glb_bytes: bytes, source: str) -> Tuple[int, int, Dict[str, Any]]:
    """(version, JSON chunk length, glTF JSON) of a binary GLB; raises PipelineAbort when the bytes
    are not a GLB whose first chunk is a complete, valid JSON chunk."""
    if len(glb_bytes) < 20:
        raise PipelineAbort(f"{source} is not a GLB: {len(glb_bytes)} bytes, shorter than the 20-byte GLB header")
    magic, ver, _length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF":
        raise PipelineAbort(f"{source} is not a GLB: magic is {magic!r}, expected b'glTF'")
    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        raise PipelineAbort(f"{source} is not a valid GLB: first chunk type is {chunk_type!r}, expected b'JSON'")
    if 20 + chunk_len > len(glb_bytes):
        raise PipelineAbort(
            f"{source} is truncated: JSON chunk declares {chunk_len} bytes, only {len(glb_bytes) - 20} present"
        )
    try:
        gltf = json.loads(glb_bytes[20:20 + chunk_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PipelineAbort(f"{source} has an invalid JSON chunk: {e}") from e
    if not isinstance(gltf, dict):
        raise PipelineAbort(f"{source} has an invalid JSON chunk: top level is {type(gltf).__name__}, expected an object")
    return ver, chunk_len, gltf


def read_glb(path: Union[Path, str]) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """(glTF JSON, BIN chunk bytes or None) of a GLB file; raises PipelineAbort when the file is not a
    GLB with a valid JSON chunk, or its second chunk is not a complete BIN chunk."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise PipelineAbort(f"Cannot read {path.name}: {e}") from e
    _, chunk_len, gltf = _parse_glb_json(data, path.name)
    offset = 20 + chunk_len
    if offset == len(data):
        return gltf, None
    if offset + 8 > len(data):
        raise PipelineAbort(f"{path.name} is truncated after its JSON chunk")
    bin_len, bin_type = struct.unpack("<I4s", data[offset:offset + 8])
    if bin_type != b"BIN\x00":
        raise PipelineAbort(f"{path.name} is not a valid GLB: second chunk type is {bin_type!r}, expected b'BIN\\x00'")
    if offset + 8 + bin_len > len(data):
        raise PipelineAbort(f"{path.name} is truncated: BIN chunk declares {bin_len} bytes, only {len(data) - offset - 8} present")
    return gltf, data[offset + 8:offset + 8 + bin_len]


def set_frontside_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is false for all materials in a binary GLB."""
    ver, chunk_len, gltf = _parse_glb_json(glb_bytes, "set_frontside_material input")

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
    out.extend(struct.pack("<4sII", b"glTF", ver, new_total_len))
    out.extend(struct.pack("<I4s", len(new_json_bytes), b"JSON"))
    out.extend(new_json_bytes)
    out.extend(bin_chunk)
    return bytes(out)


def _decompress_meshopt_if_needed(input_path: Path, tmp_dir: Path) -> Path:
    """If input GLB has EXT_meshopt_compression, decompress it using Node.js for trimesh compatibility.
    Raises PipelineAbort when the input is not a GLB, Node fails, or Node writes no output."""
    _, _, gltf = _parse_glb_json(Path(input_path).read_bytes(), str(input_path))
    exts = gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", [])
    if "EXT_meshopt_compression" not in exts:
        return input_path

    uncompressed_path = tmp_dir / f"unpacked_{input_path.name}"
    uncompressed_path.unlink(missing_ok=True)  # a stale file must not pass for Node's output
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
    proc = subprocess.run(
        ["node", "-e", node_script],
        cwd=str(REPO_ROOT),
        capture_output=True
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise PipelineAbort(
            f"Meshopt decompression of {input_path} failed (node exit {proc.returncode}): {stderr}"
        )
    if not uncompressed_path.exists():
        raise PipelineAbort(
            f"Meshopt decompression of {input_path} wrote no output ({uncompressed_path} missing)"
        )
    return uncompressed_path


def set_doublesided_material(glb_bytes: bytes) -> bytes:
    """Ensures doubleSided is true for all materials in a binary GLB (which must have materials)."""
    ver, chunk_len, gltf = _parse_glb_json(glb_bytes, "set_doublesided_material input")

    if not gltf.get("materials"):
        raise PipelineAbort("set_doublesided_material input has no materials to mark doubleSided")

    modified = False
    for mat in gltf["materials"]:
        if mat.get("doubleSided") is not True:
            mat["doubleSided"] = True
            modified = True

    if not modified:
        return glb_bytes

    new_json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * pad

    bin_chunk = glb_bytes[20 + chunk_len:]
    new_total_len = 12 + 8 + len(new_json_bytes) + len(bin_chunk)

    out = bytearray()
    out.extend(struct.pack("<4sII", b"glTF", ver, new_total_len))
    out.extend(struct.pack("<I4s", len(new_json_bytes), b"JSON"))
    out.extend(new_json_bytes)
    out.extend(bin_chunk)
    return bytes(out)


def check_glb_double_sided(glb_input: Union[bytes, Path, str]) -> bool:
    """Checks if any material in a GLB has doubleSided=True. Raises on unreadable or non-GLB input."""
    if isinstance(glb_input, (str, Path)):
        _, _, gltf = _parse_glb_json(Path(glb_input).read_bytes(), str(glb_input))
    else:
        _, _, gltf = _parse_glb_json(glb_input, "check_glb_double_sided input")
    return any(mat.get("doubleSided") is True for mat in gltf.get("materials", []))
