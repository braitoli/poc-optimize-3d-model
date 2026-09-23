#!/usr/bin/env python3
"""
scripts/inspect_koidrax_deep.py

High-performance 3D Geometry and glTF/GLB Inspector for Koidrax models.
Handles EXT_meshopt_compression via Node.js on-the-fly decompression.
"""

import sys
import os
import struct
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh

MODULE_ROOT = Path(__file__).resolve().parent.parent

def parse_glb_json(glb_path: str) -> Dict[str, Any]:
    """Parse glTF JSON chunk directly from binary GLB."""
    with open(glb_path, "rb") as f:
        glb_bytes = f.read()

    if len(glb_bytes) < 20:
        raise ValueError(f"File too short: {glb_path}")

    magic, ver, length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF":
        raise ValueError(f"Not a valid GLB: {magic}")

    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        raise ValueError(f"First chunk is not JSON: {chunk_type}")

    json_bytes = glb_bytes[20:20 + chunk_len]
    return json.loads(json_bytes.decode("utf-8"))


def decompress_meshopt_if_needed(glb_path: str, tmp_dir: Path) -> str:
    """If GLB has EXT_meshopt_compression, decompress with @gltf-transform Node.js."""
    try:
        gltf = parse_glb_json(glb_path)
        exts = gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", [])
        if "EXT_meshopt_compression" not in exts:
            return glb_path
    except Exception:
        return glb_path

    out_name = f"unpacked_{Path(glb_path).name}"
    unpacked_path = tmp_dir / out_name
    if unpacked_path.exists():
        return str(unpacked_path)

    node_script = f"""
import {{ NodeIO }} from '@gltf-transform/core';
import {{ ALL_EXTENSIONS }} from '@gltf-transform/extensions';
import {{ MeshoptDecoder }} from 'meshoptimizer';
import fs from 'fs';

async function decompress() {{
    await MeshoptDecoder.ready;
    const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({{ 'meshopt.decoder': MeshoptDecoder }});
    const doc = await io.read({json.dumps(str(Path(glb_path).resolve()))});
    const ext = doc.getRoot().listExtensionsUsed().find(e => e.extensionName === 'EXT_meshopt_compression');
    if (ext) ext.dispose();
    const glb = await io.writeBinary(doc);
    fs.writeFileSync({json.dumps(str(unpacked_path.resolve()))}, glb);
}}
decompress().catch(e => {{
    console.error(e);
    process.exit(1);
}});
"""
    subprocess.run(["node", "-e", node_script], check=True, cwd=str(MODULE_ROOT), capture_output=True)
    if unpacked_path.exists():
        return str(unpacked_path)
    return glb_path


def analyze_gltf_header(glb_path: str) -> Dict[str, Any]:
    """Analyze glTF JSON header: materials, meshes, primitives, doubleSided."""
    gltf = parse_glb_json(glb_path)
    
    meshes = gltf.get("meshes", [])
    materials = gltf.get("materials", [])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    extensions_used = gltf.get("extensionsUsed", [])
    extensions_required = gltf.get("extensionsRequired", [])
    extras = gltf.get("extras", {})

    primitives_count = sum(len(m.get("primitives", [])) for m in meshes)

    mat_details = []
    for i, mat in enumerate(materials):
        double_sided = mat.get("doubleSided", "UNDEFINED (defaults to false in spec)")
        mat_details.append({
            "index": i,
            "name": mat.get("name", f"material_{i}"),
            "doubleSided": double_sided,
            "alphaMode": mat.get("alphaMode", "OPAQUE"),
            "alphaCutoff": mat.get("alphaCutoff", 0.5),
            "pbrMetallicRoughness": mat.get("pbrMetallicRoughness", {})
        })

    return {
        "file_size": os.path.getsize(glb_path),
        "mesh_count": len(meshes),
        "primitives_count": primitives_count,
        "material_count": len(materials),
        "materials": mat_details,
        "texture_count": len(textures),
        "image_count": len(images),
        "extensions_used": extensions_used,
        "extensions_required": extensions_required,
        "extras": extras
    }


def analyze_geometry_topology_fast(glb_path: str, tmp_dir: Path) -> Dict[str, Any]:
    """
    High-speed geometric topology analysis using numpy and scipy graph theory.
    """
    t0 = time.time()
    loadable_path = decompress_meshopt_if_needed(glb_path, tmp_dir)

    mesh = trimesh.load(loadable_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(verts)
    n_faces = len(faces)

    # 1. Edge analysis
    # Directed half-edges
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges_directed = np.vstack([e0, e1, e2])  # (3 * n_faces, 2)
    
    # Sort vertices in each edge to find unique undirected edges
    edges_sorted = np.sort(all_edges_directed, axis=1)
    
    # Pack edges into int64
    packed_edges = edges_sorted[:, 0].astype(np.int64) * (n_verts + 1) + edges_sorted[:, 1].astype(np.int64)
    unique_packed, counts = np.unique(packed_edges, return_counts=True)
    
    n_unique_edges = len(unique_packed)
    boundary_mask = (counts == 1)
    manifold_mask = (counts == 2)
    non_manifold_mask = (counts > 2)
    
    n_boundary_edges = int(np.sum(boundary_mask))
    n_manifold_edges = int(np.sum(manifold_mask))
    n_non_manifold_edges = int(np.sum(non_manifold_mask))

    is_watertight = (n_boundary_edges == 0) and (n_non_manifold_edges == 0)

    # 2. Winding order consistency on manifold edges
    # For each manifold edge (count == 2), check if directed edges oppose each other:
    # Face A has (u, v) and Face B has (v, u).
    # Pack directed edges: u * (n_verts+1) + v
    packed_directed = all_edges_directed[:, 0].astype(np.int64) * (n_verts + 1) + all_edges_directed[:, 1].astype(np.int64)
    # Find duplicate directed edges: if (u, v) appears twice, then both faces have identical winding (inconsistent/flipped)!
    _, dir_counts = np.unique(packed_directed, return_counts=True)
    inconsistent_directed_edges = int(np.sum(dir_counts > 1))

    # 3. Connected components via face adjacency
    adj = mesh.face_adjacency
    if len(adj) > 0:
        row = adj[:, 0]
        col = adj[:, 1]
        data = np.ones(len(adj), dtype=bool)
        graph = sp.csr_matrix((data, (row, col)), shape=(n_faces, n_faces))
        n_components, labels = csgraph.connected_components(graph, directed=False)
    else:
        n_components = n_faces
        labels = np.arange(n_faces)

    comp_sizes = np.bincount(labels, minlength=n_components)
    
    # Boundary edges per component
    face_ids = np.tile(np.arange(n_faces), 3)
    boundary_set = set(unique_packed[boundary_mask])
    is_boundary_half_edge = np.fromiter((p in boundary_set for p in packed_edges), dtype=bool, count=len(packed_edges))
    boundary_face_ids = face_ids[is_boundary_half_edge]
    boundary_comp_ids = labels[boundary_face_ids]
    comp_boundary_counts = np.bincount(boundary_comp_ids, minlength=n_components)

    open_components = int(np.sum(comp_boundary_counts > 0))
    closed_components = n_components - open_components

    # Component details for top 5
    top_indices = np.argsort(comp_sizes)[::-1][:5]
    component_details = []
    
    mesh_center = np.mean(verts, axis=0)
    for c_idx in top_indices:
        c_faces_mask = (labels == c_idx)
        c_face_count = int(comp_sizes[c_idx])
        c_boundary = int(comp_boundary_counts[c_idx])
        
        c_vert_indices = np.unique(faces[c_faces_mask])
        c_verts = verts[c_vert_indices]
        c_min = np.min(c_verts, axis=0)
        c_max = np.max(c_verts, axis=0)
        extents = c_max - c_min
        min_extent = float(np.min(extents))
        max_extent = float(np.max(extents))
        flatness_ratio = min_extent / (max_extent + 1e-8)
        
        component_details.append({
            "component_id": int(c_idx),
            "faces": c_face_count,
            "vertices": len(c_vert_indices),
            "boundary_edges": c_boundary,
            "is_closed": (c_boundary == 0),
            "extents": [round(float(x), 4) for x in extents],
            "flatness_ratio": round(flatness_ratio, 5),
            "center": [round(float(x), 4) for x in np.mean(c_verts, axis=0)],
            "is_open_sheet": (c_boundary > 0)
        })

    # 4. Normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross_vecs = np.cross(v1 - v0, v2 - v0)
    cross_norms = np.linalg.norm(cross_vecs, axis=1)
    zero_normals = int(np.sum(cross_norms < 1e-12))
    
    safe_norms = np.where(cross_norms < 1e-12, 1.0, cross_norms)
    face_normals = cross_vecs / safe_norms[:, None]
    
    face_centroids = (v0 + v1 + v2) / 3.0
    vec_from_center = face_centroids - mesh_center
    vec_lens = np.linalg.norm(vec_from_center, axis=1, keepdims=True)
    vec_lens = np.where(vec_lens < 1e-6, 1.0, vec_lens)
    unit_center_dirs = vec_from_center / vec_lens
    
    dots = np.sum(face_normals * unit_center_dirs, axis=1)
    outward_faces = int(np.sum(dots > 0))
    outward_pct = round(outward_faces / max(1, n_faces) * 100, 2)

    t_analysis = time.time() - t0

    return {
        "vertices_count": n_verts,
        "faces_count": n_faces,
        "is_watertight": is_watertight,
        "unique_edges_count": n_unique_edges,
        "boundary_edges_count": n_boundary_edges,
        "boundary_edges_pct": round(n_boundary_edges / max(1, n_unique_edges) * 100, 2),
        "manifold_edges_count": n_manifold_edges,
        "non_manifold_edges_count": n_non_manifold_edges,
        "inconsistent_winding_edges": inconsistent_directed_edges,
        "submesh_count": n_components,
        "closed_components": closed_components,
        "open_components": open_components,
        "zero_normals_count": zero_normals,
        "outward_faces_pct": outward_pct,
        "analysis_time_sec": round(t_analysis, 3),
        "top_components": component_details
    }


def inspect_model(label: str, path: str, tmp_dir: Path) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"🔍 INSPECTING: {label}")
    print(f"Path: {path}")
    print(f"=======================================================")
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return {"label": label, "path": path, "error": "File not found"}

    hdr = analyze_gltf_header(path)
    topo = analyze_geometry_topology_fast(path, tmp_dir)

    print(f"📦 File Size: {hdr['file_size'] / (1024*1024):.2f} MB ({hdr['file_size']:,} bytes)")
    print(f"🧱 Meshes: {hdr['mesh_count']}, Primitives: {hdr['primitives_count']}, Materials: {hdr['material_count']}")
    
    for mat in hdr["materials"]:
        ds_status = mat['doubleSided']
        flag_str = "✅ TRUE (Double-sided)" if ds_status is True else ("❌ FALSE (FrontSide only - Backface Culled)" if ds_status is False else f"⚠️ {ds_status}")
        print(f"   Material #{mat['index']} '{mat['name']}': doubleSided = {flag_str}")
        print(f"      alphaMode: {mat['alphaMode']}, roughness: {mat['pbrMetallicRoughness'].get('roughnessFactor', 'default')}")

    print(f"🔺 Triangles (Faces): {topo['faces_count']:,}")
    print(f"⚪ Vertices: {topo['vertices_count']:,}")
    print(f"🔒 Watertight: {topo['is_watertight']} (Boundary edges = {topo['boundary_edges_count']:,})")
    print(f"✂️ Boundary Edges: {topo['boundary_edges_count']:,} / {topo['unique_edges_count']:,} unique edges ({topo['boundary_edges_pct']}%)")
    print(f"⚠️ Non-manifold Edges: {topo['non_manifold_edges_count']:,}")
    print(f"🔄 Inconsistent Winding Edges: {topo['inconsistent_winding_edges']:,}")
    print(f"🧩 Submesh Components: {topo['submesh_count']} total ({topo['closed_components']} closed, {topo['open_components']} open shells)")
    print(f"📐 Outward normal facing ratio: {topo['outward_faces_pct']}% (Zero normals: {topo['zero_normals_count']})")
    print(f"⏱️ Analysis speed: {topo['analysis_time_sec']}s")

    print("\n   Top Connected Components Breakdown:")
    for comp in topo["top_components"][:4]:
        kind = "OPEN SHEET / MESH HỞ" if comp["is_open_sheet"] else "CLOSED SHELL / KHỐI KÍN"
        print(f"     - Shell #{comp['component_id']}: {comp['faces']:,} faces, {comp['vertices']:,} verts, boundary_edges={comp['boundary_edges']:,} [{kind}]")
        print(f"       extents=[{comp['extents'][0]}, {comp['extents'][1]}, {comp['extents'][2]}], flatness_ratio={comp['flatness_ratio']}")

    return {
        "label": label,
        "path": path,
        "header": hdr,
        "topology": topo
    }


def compare_models(raw_res: Dict[str, Any], opt_res: Dict[str, Any]) -> Dict[str, Any]:
    """Compare raw vs optimized model."""
    r_hdr = raw_res["header"]
    r_topo = raw_res["topology"]
    o_hdr = opt_res["header"]
    o_topo = opt_res["topology"]

    r_ds = r_hdr["materials"][0]["doubleSided"] if r_hdr["materials"] else None
    o_ds = o_hdr["materials"][0]["doubleSided"] if o_hdr["materials"] else None

    diff = {
        "faces_match": (r_topo["faces_count"] == o_topo["faces_count"]),
        "raw_faces": r_topo["faces_count"],
        "opt_faces": o_topo["faces_count"],
        "raw_double_sided": r_ds,
        "opt_double_sided": o_ds,
        "double_sided_changed": (r_ds != o_ds),
        "raw_boundary_edges": r_topo["boundary_edges_count"],
        "opt_boundary_edges": o_topo["boundary_edges_count"],
        "raw_components": r_topo["submesh_count"],
        "opt_components": o_topo["submesh_count"],
        "size_reduction_pct": round((1 - o_hdr["file_size"] / r_hdr["file_size"]) * 100, 2)
    }
    return diff


def main():
    tmp_dir = Path("scratch/unpacked_models")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    target_files = [
        ("Koidrax Raw (Gốc AI)", "examples/models/koidrax_raw.glb"),
        ("Koidrax 2K Production", "examples/models/koidrax_opt_2k.glb"),
        ("Koidrax Restored", "examples/models/koidrax_restored.glb"),
        ("Direct Master Final (1024 KTX2)", "workspaces/job_1790014982889_v7g533/step_06_final.glb"),
        ("Direct Master Step 02 (Oriented)", "workspaces/job_1790014982889_v7g533/step_02_oriented.glb"),
        ("Direct Master Step 03 (Baked)", "workspaces/job_1790014982889_v7g533/step_03_texture_baked.glb"),
        ("Direct Master Step 05 (Meshopt)", "workspaces/job_1790014982889_v7g533/step_05_meshopt.glb"),
        ("xatlas Rechart Final (1024 KTX2)", "workspaces/job_1790015026781_nrgbiv/step_06_final.glb")
    ]

    all_results = []
    for label, path in target_files:
        res = inspect_model(label, path, tmp_dir)
        all_results.append(res)

    # Run comparisons against Raw
    raw_res = all_results[0]
    comparisons = {}
    for res in all_results[1:]:
        if "error" not in res:
            comparisons[res["label"]] = compare_models(raw_res, res)

    print("\n=======================================================")
    print("📊 COMPARISON SUMMARY VS RAW KOIDRAX:")
    print("=======================================================")
    for label, comp in comparisons.items():
        ds_flag = "⚠️ CHANGED TO FALSE (CULLING ACTIVE)" if comp["double_sided_changed"] and comp["opt_double_sided"] is False else ("✅ PRESERVED TRUE" if comp["opt_double_sided"] is True else "UNCHANGED")
        zero_dec = "✅ 100% PRESERVED" if comp["faces_match"] else f"❌ MISMATCH ({comp['opt_faces']} vs {comp['raw_faces']})"
        print(f"🔹 {label}:")
        print(f"   - Rule 11 (Zero Decimation): {zero_dec} ({comp['opt_faces']:,} faces)")
        print(f"   - Material doubleSided: Raw={comp['raw_double_sided']} -> Opt={comp['opt_double_sided']} [{ds_flag}]")
        print(f"   - Boundary Edges: Raw={comp['raw_boundary_edges']:,} -> Opt={comp['opt_boundary_edges']:,}")
        print(f"   - File Size Saved: {comp['size_reduction_pct']}%")

    out_payload = {
        "models": all_results,
        "comparisons": comparisons
    }

    out_json = "scratch/koidrax_deep_inspection_summary.json"
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n🎉 Full Inspection Report saved to: {out_json}")


if __name__ == "__main__":
    main()
