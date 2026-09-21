import struct
import json
import os
import io
import glob
from PIL import Image

def inspect_glb(glb_path):
    if not os.path.exists(glb_path):
        print(f"File not found: {glb_path}")
        return None

    size = os.path.getsize(glb_path)
    result = {
        "path": glb_path,
        "size_bytes": size,
        "size_formatted": f"{size / (1024*1024):.2f} MB ({size:,} bytes)",
        "textures": [],
        "meshes_count": 0,
        "vertices_count": 0,
        "faces_count": 0,
    }
    
    with open(glb_path, "rb") as f:
        data = f.read()
        
    if len(data) < 12:
        print("Invalid file: too short")
        return None

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    magic = magic.decode("latin1")
    if magic != "glTF":
        print(f"Not a GLB file! Magic={magic}")
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
        print("No JSON chunk found!")
        return None

    images = json_data.get("images", [])
    textures = json_data.get("textures", [])
    materials = json_data.get("materials", [])
    meshes = json_data.get("meshes", [])
    accessors = json_data.get("accessors", [])
    buffer_views = json_data.get("bufferViews", [])

    total_faces = 0
    total_verts = 0
    for m in meshes:
        for p in m.get("primitives", []):
            if "indices" in p and p["indices"] < len(accessors):
                idx_acc = accessors[p["indices"]]
                total_faces += idx_acc.get("count", 0) // 3
            if "attributes" in p and "POSITION" in p["attributes"] and p["attributes"]["POSITION"] < len(accessors):
                pos_acc = accessors[p["attributes"]["POSITION"]]
                total_verts += pos_acc.get("count", 0)

    result["meshes_count"] = len(meshes)
    result["vertices_count"] = total_verts
    result["faces_count"] = total_faces
    
    # Map texture usages
    tex_usage = {}
    for mat_idx, mat in enumerate(materials):
        mat_name = mat.get("name", f"mat_{mat_idx}")
        pbr = mat.get("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr:
            tex_usage.setdefault(pbr["baseColorTexture"].get("index"), []).append(f"{mat_name} (baseColor)")
        if "metallicRoughnessTexture" in pbr:
            tex_usage.setdefault(pbr["metallicRoughnessTexture"].get("index"), []).append(f"{mat_name} (metallicRoughness)")
        if "normalTexture" in mat:
            tex_usage.setdefault(mat["normalTexture"].get("index"), []).append(f"{mat_name} (normal)")
        if "occlusionTexture" in mat:
            tex_usage.setdefault(mat["occlusionTexture"].get("index"), []).append(f"{mat_name} (occlusion)")
        if "emissiveTexture" in mat:
            tex_usage.setdefault(mat["emissiveTexture"].get("index"), []).append(f"{mat_name} (emissive)")

    print(f"\n================================================================================")
    print(f"FILE: {glb_path}")
    print(f"  Size: {result['size_formatted']}")
    print(f"  Geometry: {total_verts:,} vertices, {total_faces:,} faces, {len(meshes)} meshes")
    print(f"  Images: {len(images)}, Textures: {len(textures)}, Materials: {len(materials)}")
    
    for img_idx, img in enumerate(images):
        img_name = img.get("name", f"image_{img_idx}")
        mime_type = img.get("mimeType", "unknown")
        
        img_info = {
            "index": img_idx,
            "name": img_name,
            "mime_type": mime_type,
            "compressed_bytes": 0,
            "resolution": None,
            "format": None,
            "uncompressed_bytes": 0,
            "usage": []
        }
        
        if "bufferView" in img:
            bv_idx = img["bufferView"]
            bv = buffer_views[bv_idx]
            bv_offset = bv.get("byteOffset", 0)
            bv_len = bv.get("byteLength", 0)
            img_bytes = bin_data[bv_offset:bv_offset+bv_len]
            img_info["compressed_bytes"] = len(img_bytes)
        elif "uri" in img:
            print(f"  Image [{img_idx}]: URI reference: {img['uri']}")
            continue
        else:
            print(f"  Image [{img_idx}]: No bufferView or URI")
            continue

        # Check format
        if img_bytes.startswith(b"\xabKTX 20\xbb"):
            # KTX2 file
            vkFormat, typeSize, pixelWidth, pixelHeight, pixelDepth = struct.unpack_from("<IIIII", img_bytes, 12)
            img_info["format"] = "KTX2"
            img_info["resolution"] = (pixelWidth, pixelHeight)
            # Estimate uncompressed RGBA size
            uncompressed = pixelWidth * pixelHeight * 4
            img_info["uncompressed_bytes"] = uncompressed
            print(f"  Texture [{img_idx}] Name: '{img_name}'")
            print(f"    - Resolution: {pixelWidth} x {pixelHeight}")
            print(f"    - Container / Format: KTX2 (Basis Universal / UASTC/ETC1S)")
            print(f"    - Compressed Size in GLB: {len(img_bytes):,} bytes ({len(img_bytes)/1024:.2f} KB)")
            print(f"    - Uncompressed GPU VRAM: {uncompressed:,} bytes ({uncompressed / (1024*1024):.2f} MB)")
        else:
            try:
                im = Image.open(io.BytesIO(img_bytes))
                w, h = im.size
                fmt = im.format
                mode = im.mode
                channels = len(im.getbands())
                uncompressed = w * h * (4 if "A" in mode else 3)
                img_info["resolution"] = (w, h)
                img_info["format"] = fmt
                img_info["uncompressed_bytes"] = uncompressed
                print(f"  Texture [{img_idx}] Name: '{img_name}'")
                print(f"    - Resolution: {w} x {h}")
                print(f"    - Format: {fmt} (MIME: {mime_type}, Mode: {mode}, Channels: {channels})")
                print(f"    - Compressed Size in GLB: {len(img_bytes):,} bytes ({len(img_bytes)/1024:.2f} KB)")
                print(f"    - Uncompressed RAM/VRAM: {uncompressed:,} bytes ({uncompressed / (1024*1024):.2f} MB)")
            except Exception as e:
                print(f"  Texture [{img_idx}] Name: '{img_name}' (PIL Parse Error: {e})")
                print(f"    - MIME: {mime_type}, Size: {len(img_bytes):,} bytes")

        # Map to textures
        pointing_textures = [t_idx for t_idx, t in enumerate(textures) if t.get("source") == img_idx or t.get("extensions", {}).get("KHR_texture_basisu", {}).get("source") == img_idx]
        usages = []
        for pt in pointing_textures:
            u = tex_usage.get(pt, ["unknown"])
            usages.extend(u)
        img_info["usage"] = usages
        print(f"    - Texture Slots: {usages if usages else 'Unreferenced'}")
        result["textures"].append(img_info)

    return result

if __name__ == "__main__":
    files = [
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
        "workspaces/default_sample/step_03_texture_baked.glb",
        "workspaces/default_sample/step_06_final.glb",
        "workspaces/test_job_webp/step_03_texture_baked.glb",
        "workspaces/test_job_webp/step_06_final.glb",
        "workspaces/job_1789994093548_nci589/step3_uv_bake.glb",
        "workspaces/job_1789994093548_nci589/step6_final.glb",
        "workspaces/job_1790003735318_bt9qo1/step_03_texture_baked.glb",
        "workspaces/job_1790003735318_bt9qo1/step_06_final.glb",
    ]
    for f in files:
        inspect_glb(f)
