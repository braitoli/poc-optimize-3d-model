import os
import struct
import json
import hashlib
import io
from PIL import Image

def get_image_info(img_bytes, mime_type):
    if img_bytes.startswith(b"\xabKTX 20\xbb"):
        vkFormat, typeSize, pixelWidth, pixelHeight, pixelDepth = struct.unpack_from("<IIIII", img_bytes, 12)
        # KTX2 UASTC typically 1 byte per pixel for UASTC 4x4 or ETC1S, uncompressed is 4 bytes/pixel
        uncompressed = pixelWidth * pixelHeight * 4
        # KTX2 GPU VRAM is ~ width * height * 4 / 3 (with mipmaps compressed)
        gpu_vram = int(pixelWidth * pixelHeight * 1.3333333333333333) # 4 bpp or 1 byte/pixel * 1.33
        return {
            "format": "KTX2",
            "compression": "UASTC/Basis",
            "resolution": [pixelWidth, pixelHeight],
            "gpu_vram": int(pixelWidth * pixelHeight * 4 / 3),
            "uncompressed_vram": uncompressed
        }
    else:
        try:
            im = Image.open(io.BytesIO(img_bytes))
            w, h = im.size
            fmt = im.format
            mode = im.mode
            channels = len(im.getbands())
            uncompressed = w * h * (4 if "A" in mode else 3)
            return {
                "format": fmt,
                "compression": "None",
                "resolution": [w, h],
                "gpu_vram": uncompressed,
                "uncompressed_vram": uncompressed,
                "mode": mode,
                "channels": channels
            }
        except Exception as e:
            return {
                "format": "Unknown",
                "error": str(e)
            }

def extract_gltf_details(glb_path):
    if not os.path.exists(glb_path):
        return None
    file_size = os.path.getsize(glb_path)
    with open(glb_path, "rb") as f:
        data = f.read()
    if len(data) < 12:
        return None
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        return None
    offset = 12
    json_data = None
    bin_data = b""
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_type = chunk_type.decode("latin1")
        chunk_bytes = data[offset:offset+chunk_len]
        offset += chunk_len
        if chunk_type == "JSON":
            json_data = json.loads(chunk_bytes.decode("utf-8"))
        elif chunk_type in ("BIN\x00", "BIN"):
            bin_data = chunk_bytes

    if not json_data:
        return None

    h = hashlib.md5(data).hexdigest()

    meshes = json_data.get("meshes", [])
    nodes = json_data.get("nodes", [])
    scenes = json_data.get("scenes", [])
    accessors = json_data.get("accessors", [])
    buffer_views = json_data.get("bufferViews", [])
    materials = json_data.get("materials", [])
    textures = json_data.get("textures", [])
    images = json_data.get("images", [])
    animations = json_data.get("animations", [])
    skins = json_data.get("skins", [])
    extensions_used = json_data.get("extensionsUsed", [])
    extensions_required = json_data.get("extensionsRequired", [])
    extras = json_data.get("extras", {})

    total_faces = 0
    total_verts = 0
    total_prims = 0
    attributes_present = set()
    morph_targets = 0

    for m in meshes:
        for p in m.get("primitives", []):
            total_prims += 1
            if "indices" in p and p["indices"] < len(accessors):
                total_faces += accessors[p["indices"]].get("count", 0) // 3
            attrs = p.get("attributes", {})
            for k in attrs.keys():
                attributes_present.add(k)
            if "POSITION" in attrs and attrs["POSITION"] < len(accessors):
                total_verts += accessors[attrs["POSITION"]].get("count", 0)
            if "targets" in p:
                morph_targets += len(p["targets"])

    uv_channels = []
    for attr in sorted(attributes_present):
        if attr.startswith("TEXCOORD"):
            for m in meshes:
                for p in m.get("primitives", []):
                    if attr in p.get("attributes", {}):
                        acc_idx = p["attributes"][attr]
                        acc = accessors[acc_idx]
                        uv_channels.append({
                            "channel": attr,
                            "count": acc.get("count"),
                            "min": acc.get("min"),
                            "max": acc.get("max"),
                            "componentType": acc.get("componentType"),
                            "type": acc.get("type")
                        })
                        break
                if uv_channels:
                    break

    tex_details = []
    for img_idx, img in enumerate(images):
        mime = img.get("mimeType", "unknown")
        name = img.get("name", f"img_{img_idx}")
        bv_idx = img.get("bufferView")
        img_len = 0
        img_info = {}
        if bv_idx is not None and bv_idx < len(buffer_views):
            bv = buffer_views[bv_idx]
            bv_offset = bv.get("byteOffset", 0)
            bv_len = bv.get("byteLength", 0)
            img_len = bv_len
            img_bytes = bin_data[bv_offset:bv_offset+bv_len]
            img_info = get_image_info(img_bytes, mime)
        
        tex_details.append({
            "index": img_idx,
            "name": name,
            "mimeType": mime,
            "bytes": img_len,
            "info": img_info
        })

    return {
        "file": glb_path,
        "size_bytes": file_size,
        "md5": h,
        "nodes_count": len(nodes),
        "scenes_count": len(scenes),
        "meshes_count": len(meshes),
        "primitives_count": total_prims,
        "faces_count": total_faces,
        "vertices_count": total_verts,
        "attributes": sorted(list(attributes_present)),
        "uv_channels": uv_channels,
        "materials_count": len(materials),
        "materials": [
            {
                "name": m.get("name"),
                "alphaMode": m.get("alphaMode", "OPAQUE"),
                "doubleSided": m.get("doubleSided", False),
                "roughness": m.get("pbrMetallicRoughness", {}).get("roughnessFactor"),
                "metallic": m.get("pbrMetallicRoughness", {}).get("metallicFactor"),
                "hasBaseColor": "baseColorTexture" in m.get("pbrMetallicRoughness", {})
            }
            for m in materials
        ],
        "textures_count": len(textures),
        "images_count": len(images),
        "images": tex_details,
        "animations_count": len(animations),
        "skins_count": len(skins),
        "morph_targets_count": morph_targets,
        "extensionsUsed": extensions_used,
        "extensionsRequired": extensions_required,
        "extras": extras
    }

if __name__ == "__main__":
    target_files = [
        "examples/models/dinoki_raw.glb",
        "examples/sample_dinoki.glb",
        "examples/sample_dinoki_opt.glb",
        "workspaces/benchmark_dinoki/dinoki/step_00_raw.glb",
        "workspaces/benchmark_dinoki/dinoki/step_01_cleaned_grounded.glb",
        "workspaces/benchmark_dinoki/dinoki/step_02_oriented.glb",
        "workspaces/benchmark_dinoki/dinoki/step_03_texture_baked.glb",
        "workspaces/benchmark_dinoki/dinoki/step_04_palette_tagged.glb",
        "workspaces/benchmark_dinoki/dinoki/step_05_meshopt.glb",
        "workspaces/benchmark_dinoki/dinoki/step_06_final.glb",
        "workspaces/exp_dinoki_res1024/step_06_final.glb",
        "workspaces/exp_dinoki_res2048/step_06_final.glb",
        "workspaces/test_job_webp/step_06_final.glb",
        "examples/jobs/default_sample/steps/step6_final.glb",
    ]

    all_res = []
    for tf in target_files:
        if os.path.exists(tf):
            res = extract_gltf_details(tf)
            all_res.append(res)

    print(json.dumps(all_res, indent=2))
