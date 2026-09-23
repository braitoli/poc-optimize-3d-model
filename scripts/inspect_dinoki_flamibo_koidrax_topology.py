#!/usr/bin/env python3
"""
inspect_dinoki_flamibo_koidrax_topology.py

Detailed 3D geometry topology, glTF material header, and rendering comparative analysis
between Dinoki, Flamibo, and Koidrax.
"""

import sys
import os
import json
import struct
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import csgraph, csr_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimizer.core.glb_utils import _decompress_meshopt_if_needed


def inspect_gltf_header(path: Path) -> dict:
    """Reads raw glTF JSON chunk without loading heavy meshes."""
    with open(path, "rb") as f:
        magic, ver, length = struct.unpack("<4sII", f.read(12))
        chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
        chunk_data = f.read(chunk_len)
        gltf = json.loads(chunk_data.decode("utf-8"))

    materials = gltf.get("materials", [])
    mats_info = []
    for i, m in enumerate(materials):
        mats_info.append({
            "index": i,
            "name": m.get("name", "unnamed"),
            "doubleSided": m.get("doubleSided", False),
            "alphaMode": m.get("alphaMode", "OPAQUE")
        })

    return {
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "generator": gltf.get("asset", {}).get("generator", "unknown"),
        "extensionsUsed": gltf.get("extensionsUsed", []),
        "extensionsRequired": gltf.get("extensionsRequired", []),
        "materials_count": len(materials),
        "materials": mats_info,
        "is_double_sided": any(m.get("doubleSided", False) for m in materials)
    }


def analyze_geometry_topology(path: Path, tol: float = 1e-5) -> dict:
    """
    Analyzes mesh topology, welded vertices, boundary edges, watertightness,
    connected components, and volume/thickness characteristics.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        decomp = _decompress_meshopt_if_needed(path, Path(tmp_dir))
        scene = trimesh.load(decomp, force="scene")
        mesh = trimesh.util.concatenate(scene.dump()) if isinstance(scene, trimesh.Scene) else scene

    raw_verts = len(mesh.vertices)
    raw_faces = len(mesh.faces)

    # Topological spatial welding to detect true boundary edges across UV/normal splits
    v_rounded, inv = np.unique(np.round(mesh.vertices, 5), axis=0, return_inverse=True)
    w_faces = inv[mesh.faces]
    welded_verts = len(v_rounded)

    # Edge analysis
    edges = np.sort(
        np.concatenate([w_faces[:, [0, 1]], w_faces[:, [1, 2]], w_faces[:, [2, 0]]], axis=0),
        axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    b_edges = unique_edges[counts == 1]
    m_edges = unique_edges[counts == 2]
    nm_edges = unique_edges[counts > 2]

    num_unique = len(unique_edges)
    num_boundary = len(b_edges)
    num_manifold = len(m_edges)
    num_non_manifold = len(nm_edges)

    # Connected components decomposition
    n_faces = len(w_faces)
    fids = np.repeat(np.arange(n_faces), 3)
    order = np.lexsort((fids, edges[:, 1], edges[:, 0]))
    s_edges = edges[order]
    s_fids = fids[order]
    same = (s_edges[1:] == s_edges[:-1]).all(axis=1)
    f1 = s_fids[:-1][same]
    f2 = s_fids[1:][same]
    adj = csr_matrix((np.ones(len(f1), dtype=bool), (f1, f2)), shape=(n_faces, n_faces))
    n_comp, labels = csgraph.connected_components(adj, directed=False)

    comp_sizes = np.bincount(labels)

    # Map boundary edges to components and faces
    b_edge_set = set(map(tuple, b_edges))
    comp_boundary_counts = np.zeros(n_comp, dtype=int)
    face_b_counts = np.zeros(n_faces, dtype=int)

    for f_idx, f in enumerate(w_faces):
        e1 = tuple(sorted([f[0], f[1]]))
        e2 = tuple(sorted([f[1], f[2]]))
        e3 = tuple(sorted([f[2], f[0]]))
        cnt = 0
        if e1 in b_edge_set: cnt += 1
        if e2 in b_edge_set: cnt += 1
        if e3 in b_edge_set: cnt += 1
        face_b_counts[f_idx] = cnt
        comp_boundary_counts[labels[f_idx]] += cnt

    closed_comps = int(np.sum(comp_boundary_counts == 0))
    open_comps = int(np.sum(comp_boundary_counts > 0))

    faces_in_closed = int(np.sum(comp_sizes[comp_boundary_counts == 0]))
    faces_in_open = int(np.sum(comp_sizes[comp_boundary_counts > 0]))

    faces_touching_boundary = int(np.sum(face_b_counts > 0))

    # Bounding box & dimensions
    extents = (v_rounded.max(axis=0) - v_rounded.min(axis=0)).tolist()

    # Major components overview
    top_indices = np.argsort(comp_sizes)[::-1][:5]
    major_components = []
    for rank, idx in enumerate(top_indices):
        major_components.append({
            "rank": rank + 1,
            "component_id": int(idx),
            "faces": int(comp_sizes[idx]),
            "faces_percent": round(float(comp_sizes[idx] / raw_faces * 100), 2),
            "boundary_edges": int(comp_boundary_counts[idx]),
            "is_closed_watertight": bool(comp_boundary_counts[idx] == 0)
        })

    return {
        "raw_vertices": raw_verts,
        "welded_vertices": welded_verts,
        "raw_faces": raw_faces,
        "unique_edges": num_unique,
        "boundary_edges": num_boundary,
        "boundary_edge_percent": round(num_boundary / num_unique * 100, 4) if num_unique > 0 else 0,
        "manifold_edges": num_manifold,
        "manifold_edge_percent": round(num_manifold / num_unique * 100, 2) if num_unique > 0 else 0,
        "non_manifold_edges": num_non_manifold,
        "non_manifold_edge_percent": round(num_non_manifold / num_unique * 100, 4) if num_unique > 0 else 0,
        "total_connected_components": int(n_comp),
        "closed_watertight_components": closed_comps,
        "closed_components_percent": round(closed_comps / n_comp * 100, 2) if n_comp > 0 else 0,
        "open_components": open_comps,
        "open_components_percent": round(open_comps / n_comp * 100, 2) if n_comp > 0 else 0,
        "faces_in_closed_shells": faces_in_closed,
        "faces_in_closed_percent": round(faces_in_closed / raw_faces * 100, 2) if raw_faces > 0 else 0,
        "faces_in_open_sheets": faces_in_open,
        "faces_in_open_percent": round(faces_in_open / raw_faces * 100, 2) if raw_faces > 0 else 0,
        "faces_touching_boundary": faces_touching_boundary,
        "faces_touching_boundary_percent": round(faces_touching_boundary / raw_faces * 100, 2) if raw_faces > 0 else 0,
        "extents": extents,
        "is_mesh_overall_watertight": bool(num_boundary == 0 and num_non_manifold == 0),
        "major_components": major_components
    }


def main():
    models_to_check = [
        ("Dinoki Raw", REPO_ROOT / "examples/models/dinoki_raw.glb"),
        ("Dinoki Opt 1024", REPO_ROOT / "output/qa_step3_report/dinoki/dinoki_step3_fixed_1024.glb"),
        ("Flamibo Raw", REPO_ROOT / "examples/models/flamibo_raw.glb"),
        ("Flamibo Baseline", REPO_ROOT / "examples/models/flamibo_baseline.glb"),
        ("Flamibo Opt 2048", REPO_ROOT / "output/qa_step3_report/flamibo/flamibo_step3_fixed_2048.glb"),
        ("Koidrax Raw", REPO_ROOT / "examples/models/koidrax_raw.glb"),
        ("Koidrax Opt 2K", REPO_ROOT / "examples/models/koidrax_opt_2k.glb"),
        ("Koidrax Restored", REPO_ROOT / "examples/models/koidrax_restored.glb"),
    ]

    results = {}
    print("=" * 80)
    print("  DINOKI, FLAMIBO & KOIDRAX 3D GEOMETRIC & TOPOLOGICAL INSPECTION")
    print("=" * 80)

    for name, p in models_to_check:
        if not p.exists():
            print(f"Skipping {name} (file not found: {p})")
            continue
        print(f"\nAnalyzing: {name} ({p.name})...")
        hdr = inspect_gltf_header(p)
        topo = analyze_geometry_topology(p)
        results[name] = {
            "path": str(p),
            "header": hdr,
            "topology": topo
        }

        print(f"  File size: {hdr['file_size_mb']} MB")
        print(f"  Material doubleSided: {hdr['is_double_sided']} ({hdr['materials'][0]['doubleSided'] if hdr['materials'] else 'N/A'})")
        print(f"  Faces: {topo['raw_faces']:,} | Vertices: {topo['raw_vertices']:,} (Welded: {topo['welded_vertices']:,})")
        print(f"  Unique Edges: {topo['unique_edges']:,}")
        print(f"  Boundary Edges: {topo['boundary_edges']:,} ({topo['boundary_edge_percent']}%)")
        print(f"  Closed Watertight Shells: {topo['closed_watertight_components']} / {topo['total_connected_components']} ({topo['faces_in_closed_percent']}% of faces)")
        print(f"  Open Sheet Components: {topo['open_components']} ({topo['faces_in_open_percent']}% of faces, {topo['faces_touching_boundary']:,} boundary faces)")

    # Save to JSON
    output_json = REPO_ROOT / "output" / "dinoki_flamibo_koidrax_comparison.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary JSON saved to: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
