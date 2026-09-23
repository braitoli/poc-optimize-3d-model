"""
regenerate_default_sample.py

Regenerates the Dinoki default showcase the viewer loads before any job has run:
one full pipeline run into workspaces/default_sample/ (every step enabled, so the stepper shows
what a complete run looks like), copied to examples/jobs/default_sample/steps/.

The step GLBs keep the pipeline's own file names (step_00_raw.glb ... step_07_final.glb); the
viewer reads metrics.json for their URLs.
"""

import json
import shutil
from pathlib import Path

from optimizer.step_pipeline import StepPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_MODEL = REPO_ROOT / "examples" / "models" / "dinoki_raw.glb"
WORKSPACE_DIR = REPO_ROOT / "workspaces" / "default_sample"
EXAMPLES_STEPS_DIR = REPO_ROOT / "examples" / "jobs" / "default_sample" / "steps"

# Viewer-facing labels, one per pipeline step
STEP_LABELS = [
    ("Raw Input", "Original raw unoptimized 3D asset (Dinoki)"),
    ("Clean & Auto-Ground", "Base grounded at Y=0, unreferenced vertices dropped, winding fixed"),
    ("Shell Orienting", "Z-Buffer visibility raycast, flipped faces to CCW FrontSide"),
    ("Face Repair & Reduction", "MeshLab/CGAL repair, isolated & hidden faces cut, edges collapsed within the quality budget"),
    ("UV & Texture Bake", "UV re-chart at 1:1 texel density with 16px boundary dilation"),
    ("Palette Extraction", "10 dominant surface colors extracted via KMeans"),
    ("Meshopt Compression", "Weld, quantize, and EXT_meshopt_compression"),
    ("KTX2 GPU Compression", "Basis UASTC L2 GPU mipmaps, frontSide, final extras")
]


def main():
    print(f"🔄 Regenerating default_sample from {INPUT_MODEL}...")
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_STEPS_DIR.mkdir(parents=True, exist_ok=True)

    # The run below writes every step file itself, so clear the directory of whatever an earlier
    # layout left there (including the symlinks the old two-naming-schemes sample used)
    for previous in list(WORKSPACE_DIR.glob("step*.glb")) + list(EXAMPLES_STEPS_DIR.glob("step*.glb")):
        previous.unlink()

    pipeline = StepPipeline(
        texture_format="ktx2",
        uv_mode="xatlas",
        downscale=True,
        size_mode="exact",
        double_sided=False,
        preserve_textures=True,
        verbose=True,
        stream_events=False
    )
    result = pipeline.run(INPUT_MODEL, WORKSPACE_DIR)

    step_metrics_map = {}
    for step_entry in result["steps"]:
        index = step_entry["step"]
        name, description = STEP_LABELS[index]
        metrics = step_entry["metrics"]
        if step_entry["skipped"]:
            step_metrics_map[str(index)] = {
                "step": index, "stepName": name, "name": name, "description": description,
                "file": None, "glbUrl": None, "skipped": True
            }
            continue

        textures = metrics.get("textures") or [{}]
        step_metrics_map[str(index)] = {
            "step": index,
            "stepName": name,
            "name": name,
            "description": description,
            "file": step_entry["file"],
            "glbUrl": f"/workspaces/default_sample/{step_entry['file']}",
            "skipped": False,
            "fileSize": metrics.get("fileSizeBytes", 0),
            "fileSizeFormatted": metrics.get("fileSizeFormatted", ""),
            "faces": metrics.get("faces", 0),
            "vertices": metrics.get("vertices", 0),
            "drawCalls": metrics.get("drawCalls", 1),
            "meshes": metrics.get("meshes", 1),
            "primitives": metrics.get("primitives", 1),
            "bbox": metrics.get("boundingBox", {}).get("dimensions", []),
            # null for a step whose GLB carries no texture (Step 3 after a collapse): the viewer
            # then shows the texture that step still carries in, instead of an invented one
            "textureFormat": textures[0].get("format"),
            "textureRes": textures[0].get("resolutionFormatted"),
            "gpuVramMb": round(metrics.get("totalGpuVramBytes", 0) / (1024 * 1024), 2),
            "palette": metrics.get("palette", []),
            "paletteDetails": metrics.get("paletteDetails", []),
            "primaryColor": metrics.get("primaryColor"),
            "materials": metrics.get("materials", []),
            "details": step_entry.get("details", {})
        }
        shutil.copy2(WORKSPACE_DIR / step_entry["file"], EXAMPLES_STEPS_DIR / step_entry["file"])

    metrics_payload = {
        "jobId": "default_sample",
        "status": "completed",
        "modelName": "Dinoki Sample Showcase",
        "totalSteps": StepPipeline.TOTAL_STEPS,
        "currentStep": StepPipeline.TOTAL_STEPS - 1,
        "skippedSteps": result["summary"]["skippedSteps"],
        "config": {
            "format": "ktx2",
            "uvMode": "xatlas",
            "downscale": "on",
            "sizeMode": "exact",
            "reduceEngine": pipeline.reduce_engine,
            "reduceOps": list(pipeline.reduce_ops),
            "reduceQualityBudget": pipeline.reduce_quality_budget,
            "reduceNormalBudget": pipeline.reduce_normal_budget
        },
        "steps": step_metrics_map,
        "summary": result["summary"]
    }
    (WORKSPACE_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    reduction = result["summary"]["faceReduction"]
    print("✅ default_sample regeneration complete!")
    print(f"   Workspace: {WORKSPACE_DIR}")
    print(f"   Examples: {EXAMPLES_STEPS_DIR}")
    print(
        f"   Step 3 ({reduction['engine']}): {reduction['facesBefore']:,} -> {reduction['facesAfter']:,} faces "
        f"(-{reduction['percent']}%) at {reduction['deviationPercent']}% deviation"
    )


if __name__ == "__main__":
    main()
