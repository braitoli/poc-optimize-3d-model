"""
blender_bake_export.py

Production-ready Blender Python module and script for:
1. Scanning scene mesh objects and all material slots before join/bake.
2. Collecting and analyzing backface culling / double-sided states.
3. Consolidating multi-material states using an intelligent strategy (default: 'any').
4. Creating a new baked atlas material with exact use_backface_culling mapping.
5. Exporting cleanly to glTF 2.0 / GLB via bpy.ops.export_scene.gltf while preserving doubleSided.
6. Verifying the exported glTF/GLB JSON structure post-export.

Can be run:
- Inside Blender Python (interactive or background script)
- Via command line: blender --background --python blender_bake_export.py -- --input in.glb --output out.glb --strategy auto
"""

import sys
import os
import json
import struct
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False


# ==============================================================================
# 1. State Collection & Multi-Material Analysis
# ==============================================================================

def is_material_double_sided(mat: Any, empty_slot_fallback: bool = True) -> bool:
    """
    Determines if a Blender Material represents a double-sided surface in glTF 2.0.

    CRITICAL BLENDER <-> GLTF MAPPING:
      - Blender Material.use_backface_culling = False (culling disabled -> both sides rendered)
        ==> glTF 2.0 material.doubleSided = True
      - Blender Material.use_backface_culling = True (culling enabled -> only front face rendered)
        ==> glTF 2.0 material.doubleSided = False

    Formula:
      glTF.doubleSided = not (material.use_backface_culling)
    """
    if mat is None:
        # Mesh slot without material: safe fallback is True (prevent thin ribbons/leaves disappearing)
        return empty_slot_fallback

    # Check for direct Blender use_backface_culling attribute
    if hasattr(mat, "use_backface_culling"):
        return not bool(mat.use_backface_culling)

    # Fallback for custom properties if present
    if hasattr(mat, "get") and mat.get("doubleSided") is not None:
        return bool(mat.get("doubleSided"))

    # Default fallback if unknown
    return empty_slot_fallback


def collect_scene_material_states(
    mesh_objects: Optional[List[Any]] = None,
    empty_slot_fallback: bool = True
) -> Dict[str, Any]:
    """
    Scans all provided mesh objects (or all scene mesh objects if None)
    and collects material slots, backface culling states, and double-sided booleans.

    Returns a comprehensive state dictionary:
      {
        "objects_scanned": int,
        "total_slots": int,
        "unique_materials": int,
        "materials": {
          mat_name: {
            "use_backface_culling": bool,
            "double_sided": bool,
            "used_by_objects": [obj_name, ...]
          }
        },
        "has_empty_slots": bool,
        "has_any_double_sided": bool,
        "has_all_double_sided": bool
      }
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) is required to run collect_scene_material_states.")

    if mesh_objects is None:
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    materials_info: Dict[str, Dict[str, Any]] = {}
    total_slots = 0
    has_empty_slots = False

    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue

        slots = obj.material_slots
        if len(slots) == 0:
            # Object has mesh data but no material slots
            has_empty_slots = True
            placeholder_name = f"<empty_slot:{obj.name}>"
            if placeholder_name not in materials_info:
                ds = is_material_double_sided(None, empty_slot_fallback)
                materials_info[placeholder_name] = {
                    "use_backface_culling": not ds,
                    "double_sided": ds,
                    "used_by_objects": [obj.name]
                }
            else:
                materials_info[placeholder_name]["used_by_objects"].append(obj.name)
            continue

        for slot in slots:
            total_slots += 1
            mat = slot.material
            if mat is None:
                has_empty_slots = True
                mat_name = f"<none:{obj.name}>"
                ds = is_material_double_sided(None, empty_slot_fallback)
                culling = not ds
            else:
                mat_name = mat.name
                ds = is_material_double_sided(mat, empty_slot_fallback)
                culling = getattr(mat, "use_backface_culling", not ds)

            if mat_name not in materials_info:
                materials_info[mat_name] = {
                    "use_backface_culling": culling,
                    "double_sided": ds,
                    "used_by_objects": [obj.name]
                }
            else:
                if obj.name not in materials_info[mat_name]["used_by_objects"]:
                    materials_info[mat_name]["used_by_objects"].append(obj.name)

    ds_values = [info["double_sided"] for info in materials_info.values()]
    has_any = any(ds_values) if ds_values else empty_slot_fallback
    has_all = all(ds_values) if ds_values else empty_slot_fallback

    return {
        "objects_scanned": len(mesh_objects),
        "total_slots": total_slots,
        "unique_materials": len(materials_info),
        "materials": materials_info,
        "has_empty_slots": has_empty_slots,
        "has_any_double_sided": has_any,
        "has_all_double_sided": has_all
    }


# ==============================================================================
# 2. Multi-Material Consolidation Strategy
# ==============================================================================

def resolve_consolidated_double_sided(
    scene_state: Dict[str, Any],
    strategy: str = "auto"
) -> bool:
    """
    Resolves the single consolidated double-sided boolean for a baked mesh/material atlas.

    Strategies:
      - 'auto' (Default, Golden Standard):
          Applies 'any()'. If at least one source material is double-sided, the baked
          material becomes double-sided (use_backface_culling = False).
          Why: Solid meshes do NOT suffer visual corruption when double-sided (GPU depth
          buffer / Z-buffer correctly resolves occluded backfaces), but thin planar elements
          (leaves, ribbons, hair cards, cape cloth) will completely vanish (fatal culling bug)
          if single-sided.
      - 'any':
          Explicit any(double_sided_list).
      - 'all':
          Strict all(double_sided_list). Only double-sided if all source materials are double-sided.
      - 'force_double' / 'double':
          Always True (use_backface_culling = False).
      - 'force_single' / 'single':
          Always False (use_backface_culling = True).
    """
    strat = strategy.lower().strip()

    if strat in ("force_double", "double", "true"):
        return True
    if strat in ("force_single", "single", "false"):
        return False

    materials = scene_state.get("materials", {})
    if not materials:
        return True

    ds_values = [m["double_sided"] for m in materials.values()]

    if strat == "all":
        return all(ds_values)

    # Default 'auto' or 'any'
    return any(ds_values)


# ==============================================================================
# 3. Baked Material Creation & Shader Setup
# ==============================================================================

def create_baked_material(
    name: str = "M_Baked_Atlas",
    is_double_sided: bool = True,
    texture_image: Optional[Any] = None,
    texture_path: Optional[str] = None
) -> Any:
    """
    Creates a new Blender Material configured for glTF 2.0 export with baked texture.

    Crucially assigns:
      baked_mat.use_backface_culling = not is_double_sided

    When is_double_sided is True:
      use_backface_culling = False -> io_scene_gltf2 exports doubleSided: true
    When is_double_sided is False:
      use_backface_culling = True  -> io_scene_gltf2 exports doubleSided: false
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) is required to run create_baked_material.")

    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # 1. Backface culling configuration for glTF doubleSided
    mat.use_backface_culling = not is_double_sided

    # 2. Shader node setup (Principled BSDF)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 0)

    # Principled BSDF
    node_principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    node_principled.location = (100, 0)
    links.new(node_principled.outputs["BSDF"], node_output.inputs["Surface"])

    # Base Color Texture (if image supplied)
    img = texture_image
    if img is None and texture_path and os.path.exists(texture_path):
        img = bpy.data.images.load(texture_path, check_existing=True)

    if img is not None:
        node_tex = nodes.new(type="ShaderNodeTexImage")
        node_tex.location = (-250, 0)
        node_tex.image = img
        # Ensure correct sRGB color space for base color
        if hasattr(img, "colorspace_settings"):
            img.colorspace_settings.name = "sRGB"
        links.new(node_tex.outputs["Color"], node_principled.inputs["Base Color"])
        # If image has alpha channel, link to Alpha input
        if getattr(img, "channels", 3) == 4 and "Alpha" in node_tex.outputs:
            links.new(node_tex.outputs["Alpha"], node_principled.inputs["Alpha"])
            mat.blend_method = "CLIP"  # or HASHED/BLEND depending on needs

    return mat


# ==============================================================================
# 4. Mesh Preparation, Material Assignment & Joining
# ==============================================================================

def assign_baked_material_to_mesh(
    mesh_obj: Any,
    baked_material: Any
) -> None:
    """
    Clears all material slots from mesh_obj and assigns baked_material as slot 0.
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) is required.")

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(baked_material)


def join_and_bake_meshes(
    mesh_objects: List[Any],
    baked_material_name: str = "M_Baked_Atlas",
    strategy: str = "auto",
    texture_path: Optional[str] = None
) -> Tuple[Any, Any, bool]:
    """
    High-level consolidation routine:
    1. Gathers material states from all mesh objects.
    2. Resolves consolidated double_sided state using strategy (default: 'any').
    3. Creates new baked material with use_backface_culling = not is_double_sided.
    4. Duplicates, joins meshes into a single consolidated object, and applies material.

    Returns:
      (joined_object, baked_material, is_double_sided)
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) is required.")

    # 1. State collection
    state = collect_scene_material_states(mesh_objects)

    # 2. Resolve consolidated double-sided status
    is_double_sided = resolve_consolidated_double_sided(state, strategy=strategy)

    # 3. Create baked material
    baked_mat = create_baked_material(
        name=baked_material_name,
        is_double_sided=is_double_sided,
        texture_path=texture_path
    )

    # 4. Duplicate meshes for joining
    bpy.ops.object.select_all(action='DESELECT')
    cloned_objects = []
    for obj in mesh_objects:
        obj_copy = obj.copy()
        obj_copy.data = obj.data.copy()
        bpy.context.collection.objects.link(obj_copy)
        obj_copy.select_set(True)
        cloned_objects.append(obj_copy)

    bpy.context.view_layer.objects.active = cloned_objects[0]

    # Join if multiple objects
    if len(cloned_objects) > 1:
        bpy.ops.object.join()
        joined_obj = bpy.context.view_layer.objects.active
    else:
        joined_obj = cloned_objects[0]

    joined_obj.name = "Consolidated_Baked_Mesh"

    # Assign single baked material
    assign_baked_material_to_mesh(joined_obj, baked_mat)

    return joined_obj, baked_mat, is_double_sided


# ==============================================================================
# 5. Export glTF / GLB with Strict Setting Validation
# ==============================================================================

def export_scene_gltf(
    filepath: str,
    export_format: str = "GLB",
    use_selection: bool = False,
    export_materials: str = "EXPORT"
) -> bool:
    """
    Invokes Blender glTF 2.0 Exporter (io_scene_gltf2) with production settings.

    Ensures:
      - export_materials='EXPORT' so materials and doubleSided are preserved.
      - Correct format handling ('GLB', 'GLTF_SEPARATE', 'GLTF_EMBEDDED').
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender (bpy) is required to call bpy.ops.export_scene.gltf.")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Ensure io_scene_gltf2 addon is enabled
    if "io_scene_gltf2" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")

    # Call glTF exporter
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format=export_format,
        use_selection=use_selection,
        export_materials=export_materials,
        export_colors=True,
        export_normals=True,
        export_tangents=False,
        export_apply=True
    )

    return os.path.exists(filepath)


# ==============================================================================
# 6. Post-Export glTF/GLB Binary Verification
# ==============================================================================

def verify_gltf_double_sided(
    filepath: str,
    expected_double_sided: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Verifies the exported glTF/GLB file directly from disk by decoding the JSON chunk.
    Works independently of Blender (pure Python).

    Returns:
      {
        "valid": bool,
        "format": "GLB" | "glTF",
        "num_materials": int,
        "materials": [
          {"name": str, "doubleSided": bool}
        ],
        "matches_expected": bool
      }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        head = f.read(20)

    gltf_data = None
    is_glb = False

    if head[:4] == b"glTF":
        is_glb = True
        magic, ver, total_len = struct.unpack("<4sII", head[:12])
        chunk_len, chunk_type = struct.unpack("<I4s", head[12:20])
        if chunk_type == b"JSON":
            with open(filepath, "rb") as f:
                f.seek(20)
                json_bytes = f.read(chunk_len)
                gltf_data = json.loads(json_bytes.decode("utf-8"))
    else:
        # Try plain text glTF
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                gltf_data = json.load(f)
        except Exception:
            pass

    if gltf_data is None:
        return {
            "valid": False,
            "error": "Failed to parse glTF JSON header",
            "matches_expected": False
        }

    raw_materials = gltf_data.get("materials", [])
    verified_materials = []
    all_match = True

    for mat in raw_materials:
        # glTF specification: default for doubleSided is false
        ds = mat.get("doubleSided", False)
        verified_materials.append({
            "name": mat.get("name", "unnamed"),
            "doubleSided": ds
        })
        if expected_double_sided is not None and ds != expected_double_sided:
            all_match = False

    return {
        "valid": True,
        "format": "GLB" if is_glb else "glTF",
        "num_materials": len(verified_materials),
        "materials": verified_materials,
        "matches_expected": all_match if expected_double_sided is not None else True
    }


# ==============================================================================
# 7. Standalone CLI Entry Point (Blender Headless Mode)
# ==============================================================================

def main():
    """CLI runner when invoked via `blender -b -P blender_bake_export.py -- [args]`."""
    argv = sys.argv
    if "--" in argv:
        args_to_parse = argv[argv.index("--") + 1:]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser(description="Blender Double-Sided Mesh Consolidation & glTF Export")
    parser.add_argument("--input", type=str, help="Input 3D file (.glb/.gltf/.obj/.fbx) to import")
    parser.add_argument("--output", type=str, required=True, help="Output .glb path")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "any", "all", "force_double", "force_single"], help="Multi-material aggregation strategy")
    parser.add_argument("--texture", type=str, default=None, help="Path to baked texture image")
    parser.add_argument("--format", type=str, default="GLB", choices=["GLB", "GLTF_SEPARATE", "GLTF_EMBEDDED"])

    args = parser.parse_args(args_to_parse)

    if not BLENDER_AVAILABLE:
        print("[ERROR] Blender Python environment (bpy) not detected. Exiting.")
        sys.exit(1)

    print("=" * 60)
    print("Blender glTF Double-Sided Baker & Exporter")
    print(f"Strategy: {args.strategy.upper()}")
    print("=" * 60)

    # 1. Clear existing scene if importing input
    if args.input:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        in_lower = args.input.lower()
        if in_lower.endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=args.input)
        elif in_lower.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=args.input) if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj(filepath=args.input)
        elif in_lower.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=args.input)

    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not mesh_objs:
        print("[ERROR] No mesh objects found in scene.")
        sys.exit(1)

    print(f"[*] Found {len(mesh_objs)} mesh objects in scene.")

    # 2. Join, bake material, configure use_backface_culling
    joined_obj, baked_mat, is_ds = join_and_bake_meshes(
        mesh_objects=mesh_objs,
        strategy=args.strategy,
        texture_path=args.texture
    )

    print(f"[*] Consolidated doubleSided result: {is_ds}")
    print(f"[*] Material use_backface_culling set to: {baked_mat.use_backface_culling}")

    # 3. Export to glTF
    print(f"[*] Exporting to: {args.output}")
    export_scene_gltf(
        filepath=args.output,
        export_format=args.format,
        use_selection=False,
        export_materials="EXPORT"
    )

    # 4. Verify exported binary
    verification = verify_gltf_double_sided(args.output, expected_double_sided=is_ds)
    print(f"[*] Export verification: {verification}")
    if verification.get("matches_expected"):
        print("✅ SUCCESS: Exported glTF matches expected doubleSided configuration perfectly.")
    else:
        print("❌ WARNING: Exported glTF doubleSided mismatch.")


if __name__ == "__main__" and BLENDER_AVAILABLE and "--" in sys.argv:
    main()
