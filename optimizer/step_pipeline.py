"""
step_pipeline.py

Step-by-Step 3D Model Optimization Pipeline with granular GLB stage exports & metrics.
- Step 0: step_00_raw.glb (Raw input mesh ingested & inspected)
- Step 1: step_01_cleaned_grounded.glb (Cleaner & auto_ground_and_center Y=0)
- Step 2: step_02_oriented.glb (Shell orient z-buffer visibility outward CCW FrontSide)
- Step 3: step_03_face_reduced.glb (CGAL / MeshLab face repair & reduction, quality budget)
- Step 4: step_04_texture_baked.glb (uv_baker Lanczos resample & 16px dilation)
- Step 5: step_05_palette_tagged.glb (Palette k-means 10 dominant colors embedded)
- Step 6: step_06_meshopt.glb (Node smooth_normals, weld, quantize, meshopt geometry)
- Step 7: step_07_final.glb (Basisu KTX2/WebP GPU compression, frontSide, extras)

Steps 1-6 are optional (--skip-steps): a skipped step writes no GLB, and nothing downstream reads
what it would have produced. Step 3 is the only step that removes triangles, and only within its
quality budget; every other step still preserves 100% of the faces it is given (Rule 11).

Emits real-time NDJSON events to stdout:
{"event": "step_complete", "step": X, "stepName": "...", "file": "step_XX_....glb", "metrics": {...}}
{"event": "step_skipped", "step": X, "stepName": "...", "file": null}
then {"event": "pipeline_complete", "summary": {...}}, and continuously updates <output_dir>/metrics.json.
On failure the CLI prints {"event": "pipeline_error", "error": "<reason>", "errorType": "<class>", "step": <n|null>}
as its last stdout line (full traceback on stderr) and exits 1.
"""

import io
import os
import sys
import time
import json
import base64
import binascii
import shutil
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
from PIL import Image
import trimesh

from optimizer.core.cleaner import clean_and_repair_mesh, auto_ground_and_center
from optimizer.core.shell_orient import orient_faces_by_visibility, DEFAULT_VIEWS, DEFAULT_RESOLUTION
from optimizer.core.uv_baker import SIZE_MODES, plan_uv_canvas, bake_uv_plan
from optimizer.core.face_reduce import (
    DEFAULT_ISOLATED_MIN_FACES,
    DEFAULT_NORMAL_BUDGET_DEGREES,
    DEFAULT_OPS as DEFAULT_REDUCE_OPS,
    DEFAULT_QUALITY_BUDGET_PERCENT,
    ENGINES as REDUCE_ENGINES,
    OPS as REDUCE_OPS,
    SourceUVProjector,
    engine_available as reduce_engine_available,
    reduce_faces,
    restore_hard_edges,
    align_faces_outward,
    _within_budget as within_budget,
    validate_normal_budget,
    validate_ops as validate_reduce_ops
)
from optimizer.core.uvatlas import is_uvatlas_available, UVATLAS_UNAVAILABLE
from optimizer.core.palette import extract_palette, embed_gltf_extras
from optimizer.core.texture_utils import (
    SUPPORTED_TEXTURE_FORMATS,
    extract_original_texture_info,
    preserve_mesh_textures,
    optimize_mesh_texture_for_export
)
from optimizer.core.glb_utils import (
    read_glb,
    set_frontside_material,
    set_doublesided_material,
    check_glb_double_sided
)
from optimizer.core.errors import PipelineAbort, describe_failure

MODULE_ROOT = Path(__file__).resolve().parent
INSPECT_SCRIPT = MODULE_ROOT / "inspect_metrics.mjs"
NODE_OPT_SCRIPT = MODULE_ROOT / "node" / "optimize_meshopt.mjs"

TEXTURE_FORMATS = ("ktx2", "webp", "original", "passthrough", "raw")
# An input carrying one of these is an already optimized output; there is no decompression step
COMPRESSED_INPUT_EXTENSIONS = ("EXT_meshopt_compression", "KHR_draco_mesh_compression", "KHR_texture_basisu")
# First stderr line of a failed Node script (optimize_meshopt.mjs / inspect_metrics.mjs): the reason
NODE_ERROR_PREFIXES = ("Fatal optimization error:", "Failed to inspect GLB metrics:")


def node_step_failure(what: str, proc: subprocess.CompletedProcess) -> PipelineAbort:
    """PipelineAbort for a failed Node sub-step: `what`, then the one-line reason the script printed.
    The full output is chained as the cause, so it stays in the traceback on stderr only."""
    output = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    reason = next((line.split(":", 1)[1].strip() for line in lines if line.startswith(NODE_ERROR_PREFIXES)), None)
    if reason is None:
        reason = next((line for line in lines if "error" in line.lower()), None)
    if reason is None:
        reason = lines[0] if lines else f"node exited with code {proc.returncode} without output"
    err = PipelineAbort(f"{what}: {reason}")
    err.__cause__ = RuntimeError(f"node exited with code {proc.returncode}:\n{output}")
    return err


def _node_json(proc: subprocess.CompletedProcess, what: str) -> Dict[str, Any]:
    """The JSON object a Node script printed as its (last) result line on stdout."""
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError as err:
                raise PipelineAbort(f"{what}: invalid JSON result from Node: {err}") from err
    raise PipelineAbort(f"{what}: the Node script printed no JSON result")


def encoded_texture_summary(proc: subprocess.CompletedProcess, fmt: str) -> Dict[str, Any]:
    """The `fmt` ('ktx2' / 'webp') entry of the Step 7 Node summary; PipelineAbort unless it encoded
    at least one texture (and was not skipped)."""
    result = _node_json(proc, f"Step 7: {fmt.upper()} texture compression").get(fmt)
    if not isinstance(result, dict) or result.get("skipped") or not result.get("count", 0) > 0:
        raise PipelineAbort(f"Step 7: {fmt.upper()} compression encoded no texture (Node summary: {json.dumps(result)})")
    return result


def inspect_glb_metrics(glb_path: Path) -> Dict[str, Any]:
    """Invokes inspect_metrics.mjs to extract comprehensive 3D metrics from a GLB."""
    cmd = ["node", str(INSPECT_SCRIPT), str(glb_path), "--compact"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise node_step_failure(f"Cannot inspect {glb_path.name}", proc)
    return _node_json(proc, f"Cannot inspect {glb_path.name}")


def preflight_tools(
    texture_format: str,
    uv_mode: str,
    downscale: bool,
    uv_step: bool = True,
    reduce_engine: Optional[str] = None
) -> None:
    """Raises PipelineAbort before Step 0 when a tool this run needs is missing. A tool only a
    skipped step would have used is not required: `uv_step` is False when Step 4 is skipped, and
    `reduce_engine` is None when Step 3 is."""
    if shutil.which("node") is None:
        raise PipelineAbort("Node.js ('node') is not on PATH: the metrics, meshopt and texture steps run Node scripts")
    # Both Node scripts load sharp (via @gltf-transform/functions -> ndarray-pixels), and
    # optimize_meshopt.mjs reads, converts and resizes textures with it: needed for every format
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", "await import('sharp')"],
        cwd=str(NODE_OPT_SCRIPT.parent), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise node_step_failure("The Node package 'sharp' cannot be loaded (run npm install)", proc)
    if texture_format == "ktx2" and shutil.which("basisu") is None:
        raise PipelineAbort("Texture format KTX2 needs the basisu CLI (Basis Universal), which is not on PATH")
    if uv_step and uv_mode == "uvatlas" and downscale and not is_uvatlas_available()[0]:
        raise PipelineAbort(UVATLAS_UNAVAILABLE)
    if reduce_engine is not None:
        available, reason = reduce_engine_available(reduce_engine)
        if not available:
            raise PipelineAbort(f"Step 3 face reduction cannot run: {reason}")


def _gltf_item(gltf: Dict[str, Any], key: str, index: Any, what: str) -> Dict[str, Any]:
    items = gltf.get(key, [])
    if not isinstance(index, int) or not 0 <= index < len(items):
        raise PipelineAbort(f"The {what} refers to {key}[{index}], which does not exist ({len(items)} {key})")
    return items[index]


def _image_bytes(gltf: Dict[str, Any], bin_chunk: Optional[bytes], image: Dict[str, Any]) -> bytes:
    """Encoded bytes of a glTF image stored in the GLB's BIN chunk or in a data: URI."""
    if "bufferView" in image:
        view = _gltf_item(gltf, "bufferViews", image["bufferView"], "baseColorTexture image")
        start, length = view.get("byteOffset", 0), view.get("byteLength", 0)
        if view.get("buffer") != 0 or bin_chunk is None or start + length > len(bin_chunk):
            raise PipelineAbort("The baseColorTexture image data is not inside the GLB's BIN chunk")
        return bin_chunk[start:start + length]
    uri = image.get("uri", "")
    if uri.startswith("data:") and ";base64," in uri:
        try:
            return base64.b64decode(uri.split(";base64,", 1)[1], validate=True)
        except binascii.Error as err:
            raise PipelineAbort(f"The baseColorTexture data: URI is not valid base64: {err}") from err
    raise PipelineAbort(f"The baseColorTexture image is an external file ({uri or 'no uri'}): only self-contained GLBs are supported")


def validate_input_glb(path: Path) -> None:
    """
    Checks, from the input GLB's JSON (and the base colour image bytes), that it is a model the
    pipeline supports; raises PipelineAbort with the specific reason otherwise:
    - not an already optimized/compressed output (COMPRESSED_INPUT_EXTENSIONS);
    - exactly one mesh primitive (across all meshes), placed by one node, with a material;
    - that material has a baseColorTexture read through TEXCOORD_0, whose JPEG / PNG / WebP image decodes;
    - the primitive has POSITION and TEXCOORD_0 with one UV per vertex.
    Non-finite vertex positions are refused by the cleaner (Step 1).
    """
    gltf, bin_chunk = read_glb(path)
    listed = [*gltf.get("extensionsRequired", []), *gltf.get("extensionsUsed", [])]
    compressed = [ext for ext in COMPRESSED_INPUT_EXTENSIONS if ext in listed]
    if compressed:
        raise PipelineAbort(
            f"Input is already optimized/compressed ({', '.join(compressed)}); upload the original uncompressed model."
        )

    meshes = gltf.get("meshes", [])
    primitives = [prim for mesh in meshes for prim in mesh.get("primitives", [])]
    if not primitives:
        raise PipelineAbort("input has no mesh primitives")
    mesh_nodes = [node for node in gltf.get("nodes", []) if "mesh" in node]
    if len(primitives) != 1 or len(mesh_nodes) != 1:
        raise PipelineAbort(
            f"input has {len(primitives)} mesh primitive(s) in {len(meshes)} mesh(es) placed by {len(mesh_nodes)} "
            f"node(s): only single-mesh, single-material models are supported"
        )

    prim = primitives[0]
    if "material" not in prim:
        raise PipelineAbort("The mesh primitive has no material: only textured models are supported")
    material = _gltf_item(gltf, "materials", prim["material"], "mesh primitive")
    base = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if base is None:
        raise PipelineAbort("input has no baseColorTexture: only textured models are supported")
    if base.get("texCoord", 0) != 0:
        raise PipelineAbort(f"The baseColorTexture uses TEXCOORD_{base['texCoord']}: only TEXCOORD_0 is supported")

    attributes = prim.get("attributes", {})
    if "POSITION" not in attributes:
        raise PipelineAbort("The mesh primitive has no POSITION attribute")
    if "TEXCOORD_0" not in attributes:
        raise PipelineAbort("The mesh primitive has no TEXCOORD_0 (UVs): only UV-mapped models are supported")
    n_vertices = _gltf_item(gltf, "accessors", attributes["POSITION"], "POSITION attribute").get("count")
    n_uvs = _gltf_item(gltf, "accessors", attributes["TEXCOORD_0"], "TEXCOORD_0 attribute").get("count")
    if n_uvs != n_vertices:
        raise PipelineAbort(f"TEXCOORD_0 has {n_uvs} UVs for {n_vertices} vertices (POSITION)")

    texture = _gltf_item(gltf, "textures", base.get("index"), "baseColorTexture")
    source = texture.get("source", texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
    image = _gltf_item(gltf, "images", source, "baseColorTexture")
    data = _image_bytes(gltf, bin_chunk, image)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img_format = img.format
            img.load()
    except Exception as err:
        raise PipelineAbort(
            f"The baseColorTexture image ({image.get('mimeType', 'no MIME type')}, {len(data)} bytes) "
            f"cannot be decoded: {err}"
        ) from err
    if img_format not in SUPPORTED_TEXTURE_FORMATS:
        raise PipelineAbort(
            f"The baseColorTexture image is {img_format}: supported formats are {', '.join(SUPPORTED_TEXTURE_FORMATS)}"
        )


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a friendly human-readable string (e.g. '54ms', '0.24s', '1.53s')."""
    if seconds < 0.1:
        ms = round(seconds * 1000)
        return f"{ms}ms" if ms > 0 else "<1ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


class StepPipeline:
    """
    Orchestrates the 8-step optimization pipeline, exporting intermediate GLBs
    and recording 3D metrics at each discrete step. Steps 1-6 can be switched off (skip_steps).
    """

    STEP_DEFINITIONS = [
        {"step": 0, "name": "raw", "file": "step_00_raw.glb", "desc": "Raw input model ingested & analyzed", "optional": False},
        {"step": 1, "name": "cleaned_grounded", "file": "step_01_cleaned_grounded.glb", "desc": "Cleaned geometry & grounded at Y=0", "optional": True},
        {"step": 2, "name": "oriented", "file": "step_02_oriented.glb", "desc": "Visibility-based shell orientation (outward CCW)", "optional": True},
        {"step": 3, "name": "face_reduced", "file": "step_03_face_reduced.glb", "desc": "CGAL / MeshLab face repair & reduction within the quality budget", "optional": True},
        {"step": 4, "name": "texture_baked", "file": "step_04_texture_baked.glb", "desc": "Texture resampled with Lanczos + 16px dilation", "optional": True},
        {"step": 5, "name": "palette_tagged", "file": "step_05_palette_tagged.glb", "desc": "10-color dominant palette extracted & embedded", "optional": True},
        {"step": 6, "name": "meshopt", "file": "step_06_meshopt.glb", "desc": "Smooth normals, weld, quantize, and EXT_meshopt_compression", "optional": True},
        {"step": 7, "name": "final", "file": "step_07_final.glb", "desc": "GPU texture compression (KTX2/WebP), frontSide, final extras", "optional": False}
    ]
    TOTAL_STEPS = len(STEP_DEFINITIONS)
    # Every step but the raw ingest and the final model can be switched off from the UI / CLI
    OPTIONAL_STEPS = tuple(d["step"] for d in STEP_DEFINITIONS if d["optional"])

    def __init__(
        self,
        texture_format: str = "ktx2",
        uv_mode: str = "xatlas",
        downscale: bool = True,
        size_mode: str = "exact",
        merge_uv_islands: bool = True,
        smooth_normals: Optional[bool] = None,
        double_sided: bool = False,
        preserve_textures: bool = True,
        verbose: bool = True,
        stream_events: bool = True,
        ktx2_min_vram_mb: float = 20.0,
        skip_steps: Sequence[int] = (),
        reduce_engine: str = "cgal",
        reduce_ops: Sequence[str] = DEFAULT_REDUCE_OPS,
        reduce_quality_budget: float = DEFAULT_QUALITY_BUDGET_PERCENT,
        reduce_normal_budget: float = DEFAULT_NORMAL_BUDGET_DEGREES,
        reduce_isolated_min_faces: int = DEFAULT_ISOLATED_MIN_FACES,
        max_faces: Optional[int] = None
    ):
        if max_faces is not None:
            if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces <= 0:
                raise ValueError(f"max_faces must be a positive integer, got {max_faces!r}")
        self.max_faces = max_faces
        self.texture_format = texture_format.lower()
        if self.texture_format not in TEXTURE_FORMATS:
            raise ValueError(f"Unsupported texture_format '{texture_format}' (expected one of {', '.join(TEXTURE_FORMATS)})")
        self.uv_mode = uv_mode.lower()
        if self.uv_mode not in ("xatlas", "uvatlas"):
            raise ValueError(f"Unsupported uv_mode '{uv_mode}' (expected 'xatlas' or 'uvatlas')")
        if not isinstance(downscale, bool):
            raise TypeError(f"downscale must be a bool, got {downscale!r}")
        self.downscale = downscale
        if size_mode not in SIZE_MODES:
            raise ValueError(f"Unsupported size_mode '{size_mode}' (expected one of {', '.join(SIZE_MODES)})")
        self.size_mode = size_mode
        if not isinstance(merge_uv_islands, bool):
            raise TypeError(f"merge_uv_islands must be a bool, got {merge_uv_islands!r}")
        self.merge_uv_islands = merge_uv_islands
        # Step 6 KTX2 runs only when the Step 5 texture VRAM estimate reaches this many MB (0 = always)
        if isinstance(ktx2_min_vram_mb, bool) or not isinstance(ktx2_min_vram_mb, (int, float)):
            raise TypeError(f"ktx2_min_vram_mb must be a number, got {ktx2_min_vram_mb!r}")
        if not ktx2_min_vram_mb >= 0:
            raise ValueError(f"ktx2_min_vram_mb must be a non-negative number, got {ktx2_min_vram_mb!r}")
        self.ktx2_min_vram_mb = float(ktx2_min_vram_mb)

        if smooth_normals is None:
            # On: the smoothing heals the shading seam a UV split leaves behind and drops the
            # duplicate vertices with it. It only welds vertices whose normals already agree
            # (see optimizer/node/smooth_normals.mjs), so the creases Step 3 rebuilds survive it.
            self.smooth_normals = True
        else:
            self.smooth_normals = smooth_normals

        # Steps switched off by the caller: they produce no GLB and nothing downstream uses them
        skipped = set()
        for step in skip_steps:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError(f"skip_steps must contain step indices as integers, got {step!r}")
            if step not in self.OPTIONAL_STEPS:
                raise ValueError(
                    f"Step {step} cannot be skipped (optional steps: "
                    f"{', '.join(str(n) for n in self.OPTIONAL_STEPS)})"
                )
            skipped.add(step)
        self.skipped_steps = frozenset(skipped)

        if reduce_engine not in REDUCE_ENGINES:
            raise ValueError(
                f"Unsupported reduce_engine '{reduce_engine}' (expected one of {', '.join(REDUCE_ENGINES)})"
            )
        self.reduce_engine = reduce_engine
        self.reduce_ops = validate_reduce_ops(reduce_ops)
        if isinstance(reduce_quality_budget, bool) or not isinstance(reduce_quality_budget, (int, float)):
            raise TypeError(f"reduce_quality_budget must be a number, got {reduce_quality_budget!r}")
        if not reduce_quality_budget > 0:
            raise ValueError(f"reduce_quality_budget must be a positive number, got {reduce_quality_budget!r}")
        self.reduce_quality_budget = float(reduce_quality_budget)
        try:
            self.reduce_normal_budget = validate_normal_budget(reduce_normal_budget)
        except ValueError as err:
            raise ValueError(str(err).replace("normal_budget_degrees", "reduce_normal_budget")) from None
        if isinstance(reduce_isolated_min_faces, bool) or not isinstance(reduce_isolated_min_faces, int):
            raise TypeError(f"reduce_isolated_min_faces must be an integer, got {reduce_isolated_min_faces!r}")
        if reduce_isolated_min_faces < 0:
            raise ValueError(f"reduce_isolated_min_faces must not be negative, got {reduce_isolated_min_faces!r}")
        self.reduce_isolated_min_faces = reduce_isolated_min_faces
        # Collapsing edges throws the model's UVs away, and only the Step 4 re-chart can make new
        # ones: with Step 4 off the run could only end on an untextured model
        if self.enabled(3) and "merge" in self.reduce_ops and not self.enabled(4):
            raise ValueError(
                "Step 3's 'merge' operation rewrites the mesh topology and invalidates its UVs, so "
                "Step 4 (UV re-chart & texture bake) cannot be skipped: either keep Step 4 or drop "
                "'merge' from the face reduction operations"
            )

        self.double_sided = double_sided
        self.preserve_textures = preserve_textures
        self.verbose = verbose
        self.stream_events = stream_events

    def enabled(self, step: int) -> bool:
        """True when `step` runs in this pipeline (every non-optional step always does)."""
        return step not in self.skipped_steps

    def log(self, msg: str):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", file=sys.stderr, flush=True)

    def _emit_step_event(
        self,
        step_idx: int,
        step_name: str,
        filename: str,
        metrics: Dict[str, Any],
        extra_data: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        total_duration_seconds: Optional[float] = None
    ):
        event_payload = {
            "event": "step_complete",
            "step": step_idx,
            "stepName": step_name,
            "file": filename,
            "metrics": metrics
        }
        if duration_seconds is not None:
            event_payload["durationSeconds"] = round(duration_seconds, 3)
            event_payload["durationFormatted"] = format_duration(duration_seconds)
        if total_duration_seconds is not None:
            event_payload["totalDurationSeconds"] = round(total_duration_seconds, 3)
        if extra_data:
            event_payload.update(extra_data)

        if self.stream_events:
            print(json.dumps(event_payload), flush=True)

    def run(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Runs the 7 steps. Any exception that escapes carries the failing step index as `.step`
        (kept if the raiser already set it, e.g. PipelineAbort(reason, step=...))."""
        self._current_step = 0
        try:
            return self._run_steps(input_path, output_dir)
        except Exception as e:
            if getattr(e, "step", None) is None:
                e.step = self._current_step
            raise

    def _run_steps(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        t0 = time.time()
        # Before Step 0: every tool the steps this run actually executes need, then a supported input
        preflight_tools(
            self.texture_format,
            self.uv_mode,
            self.downscale,
            uv_step=self.enabled(4),
            reduce_engine=self.reduce_engine if self.enabled(3) else None
        )
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        validate_input_glb(input_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = output_dir / "metrics.json"

        steps_record: List[Dict[str, Any]] = []
        skipped_steps = sorted(self.skipped_steps)
        texture_resolution: Optional[str] = None  # "WxH" of the texture the final model carries
        texture_format_label = self.texture_format.upper()  # real final format once Step 7 skips KTX2
        last_step = self.TOTAL_STEPS - 1

        def metrics_payload(last_completed: int) -> Dict[str, Any]:
            return {
                "success": True,
                "model": input_path.name,
                "outputDir": str(output_dir),
                "resolution": texture_resolution,
                "downscale": self.downscale,
                "sizeMode": self.size_mode,
                "uvMode": self.uv_mode,
                "textureFormat": texture_format_label,
                "skippedSteps": skipped_steps,
                "steps": steps_record,
                "lastCompletedStep": last_completed
            }

        def save_and_record_metrics(
            step_idx: int,
            glb_file: Path,
            extra_info: Optional[Dict[str, Any]] = None,
            t_step_start: Optional[float] = None,
            explicit_duration: Optional[float] = None,
            duration_seconds: Optional[float] = None,
            **kwargs
        ) -> Dict[str, Any]:
            step_def = self.STEP_DEFINITIONS[step_idx]
            metrics = inspect_glb_metrics(glb_file)

            if duration_seconds is not None:
                step_duration = duration_seconds
            elif explicit_duration is not None:
                step_duration = explicit_duration
            elif t_step_start is not None:
                step_duration = time.perf_counter() - t_step_start
            else:
                step_duration = 0.0
            total_duration = time.perf_counter() - t_total_start

            dur_sec = round(step_duration, 3)
            dur_fmt = format_duration(step_duration)
            tot_sec = round(total_duration, 3)

            # Enrich metrics dictionary with precise timing indicators
            metrics["durationSeconds"] = dur_sec
            metrics["durationFormatted"] = dur_fmt
            metrics["totalDurationSeconds"] = tot_sec
            metrics["durationMs"] = round(step_duration * 1000, 1)

            if extra_info:
                metrics.update(extra_info)

            step_entry = {
                "step": step_idx,
                "stepName": step_def["name"],
                "file": step_def["file"],
                "description": step_def["desc"],
                "skipped": False,
                "durationSeconds": dur_sec,
                "durationFormatted": dur_fmt,
                "totalDurationSeconds": tot_sec,
                "durationMs": round(step_duration * 1000, 1),
                "metrics": metrics
            }
            if extra_info:
                step_entry["details"] = extra_info
            steps_record.append(step_entry)

            metrics_json_path.write_text(json.dumps(metrics_payload(step_idx), indent=2))

            # Emit streaming NDJSON event
            self._emit_step_event(
                step_idx,
                step_def["name"],
                step_def["file"],
                metrics,
                extra_data=extra_info,
                duration_seconds=dur_sec,
                total_duration_seconds=tot_sec
            )
            return metrics

        def skip_step(step_idx: int) -> None:
            """Records a step the caller switched off: no GLB, no metrics, and nothing downstream
            reads its results."""
            step_def = self.STEP_DEFINITIONS[step_idx]
            steps_record.append({
                "step": step_idx,
                "stepName": step_def["name"],
                "file": None,
                "description": step_def["desc"],
                "skipped": True,
                "durationSeconds": 0.0,
                "durationFormatted": "skipped",
                "metrics": None
            })
            metrics_json_path.write_text(json.dumps(metrics_payload(step_idx), indent=2))
            self.log(f"⏭️  [Step {step_idx}/{last_step}] {step_def['desc']}: skipped")
            if self.stream_events:
                print(json.dumps({
                    "event": "step_skipped",
                    "step": step_idx,
                    "stepName": step_def["name"],
                    "file": None
                }), flush=True)

        self.log("=" * 68)
        self.log(f"🚀 STEP-BY-STEP 3D OPTIMIZATION PIPELINE: {input_path.name}")
        self.log(f"   Target Directory: {output_dir}")
        self.log(
            f"   Downscale: {'ON' if self.downscale else 'OFF'} | Size Mode: {self.size_mode} | "
            f"Format: {self.texture_format.upper()} | UV Mode: {self.uv_mode.upper()} | "
            f"Merge UV islands: {'ON' if self.merge_uv_islands else 'OFF'} | "
            f"Smooth normals: {'ON' if self.smooth_normals else 'OFF'}"
        )
        if self.enabled(3):
            self.log(
                f"   Face reduction: {self.reduce_engine.upper()} | ops: {', '.join(self.reduce_ops)} | "
                f"quality budget: {self.reduce_quality_budget:g}% / {self.reduce_normal_budget:g}deg"
            )
        if skipped_steps:
            self.log(f"   Skipped steps: {', '.join(str(n) for n in skipped_steps)}")
        self.log("=" * 68)

        # =====================================================================
        # STEP 0: Raw Model Ingestion & Baseline Metrics (always runs)
        # =====================================================================
        self.log(f"▶️ [Step 0/{last_step}] Ingesting Raw GLB Model...")
        t_s0 = time.perf_counter()
        step0_file = output_dir / "step_00_raw.glb"
        if input_path.resolve() != step0_file.resolve():
            shutil.copy2(input_path, step0_file)
        m0 = save_and_record_metrics(0, step0_file, t_step_start=t_s0)
        initial_faces = m0["faces"]
        initial_verts = m0["vertices"]
        initial_bytes = m0["fileSizeBytes"]
        orig_tex_info = extract_original_texture_info(step0_file)
        self.log(f"   ✓ Step 0 complete ({m0['durationFormatted']}): {initial_faces:,} faces, {initial_verts:,} verts, {m0['fileSizeFormatted']} (texture: {orig_tex_info.get('default_format')})")

        # The GLB the next step reads, and the metrics of the last step that produced one
        current_file = step0_file
        last_metrics = m0

        # Auto-detect doubleSided from input materials (or CLI flag)
        if check_glb_double_sided(input_path) and not self.double_sided:
            self.log("   ℹ️ Auto-detected doubleSided=True from input materials (preserving thin shells & armor)")
            self.double_sided = True

        # Steps 1-5 work on the mesh itself; loaded once, then carried and transformed in place
        mesh_steps = (1, 2, 3, 4, 5)
        mesh = (
            trimesh.load(str(input_path), force="mesh", process=False)
            if any(self.enabled(n) for n in mesh_steps) else None
        )

        # =====================================================================
        # STEP 1: Cleaner & Auto Grounding (Y=0, X/Z Centered)
        # =====================================================================
        if self.enabled(1):
            self._current_step = 1
            self.log(f"▶️ [Step 1/{last_step}] Cleaning Geometry & Auto-Grounding at Y=0...")
            t_s1 = time.perf_counter()
            cleaned_mesh = clean_and_repair_mesh(mesh)
            mesh, translation = auto_ground_and_center(cleaned_mesh)
            if self.preserve_textures:
                preserve_mesh_textures(mesh, orig_tex_info)

            step1_file = output_dir / self.STEP_DEFINITIONS[1]["file"]
            step1_file.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": mesh}), include_normals=True))
            m1 = save_and_record_metrics(1, step1_file, {"translationApplied": translation.tolist()}, t_step_start=t_s1)
            current_file, last_metrics = step1_file, m1
            self.log(f"   ✓ Step 1 complete ({m1['durationFormatted']}): Grounded at Y=0 (shift: {np.round(translation, 3).tolist()})")
        else:
            skip_step(1)

        # =====================================================================
        # STEP 2: Visibility Z-Buffer Shell Orient (Outward CCW Winding)
        # =====================================================================
        if self.enabled(2):
            self._current_step = 2
            self.log(f"▶️ [Step 2/{last_step}] Orienting Shells via Visibility Z-Buffer (CCW)...")
            t_s2 = time.perf_counter()
            orient_stats: Dict[str, Any] = {}
            mesh.faces = orient_faces_by_visibility(
                mesh.vertices,
                mesh.faces,
                views=DEFAULT_VIEWS,
                resolution=DEFAULT_RESOLUTION,
                stats=orient_stats
            )
            if self.preserve_textures:
                preserve_mesh_textures(mesh, orig_tex_info)

            step2_file = output_dir / self.STEP_DEFINITIONS[2]["file"]
            step2_file.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": mesh}), include_normals=True))
            m2 = save_and_record_metrics(2, step2_file, orient_stats, t_step_start=t_s2)
            current_file, last_metrics = step2_file, m2
            self.log(f"   ✓ Step 2 complete ({m2['durationFormatted']}): Flipped {orient_stats.get('faces_flipped', 0)} faces to outward CCW")
        else:
            skip_step(2)

        # =====================================================================
        # STEP 3: Face Repair & Reduction (CGAL / MeshLab) within the quality budget
        # =====================================================================
        # Set once Step 3 collapsed edges: the mesh then has no UV of its own, and Step 4 reads the
        # original texture through this projector instead
        uv_projector: Optional[SourceUVProjector] = None
        textured_source: Optional[trimesh.Trimesh] = None  # the pre-reduction mesh, kept for Step 4
        # The surface Step 4 measures the source texture's density against: what the removals left,
        # with the source UVs still on it. The pre-reduction mesh would include the hidden shell the
        # removals cut away, and on an AI model that shell holds most of the source atlas - sizing
        # the canvas for it asks for texels no visible face will ever sample.
        density_source: Optional[trimesh.Trimesh] = None
        reduce_stats: Dict[str, Any] = {}
        faces_before_reduction = initial_faces
        if self.enabled(3):
            self._current_step = 3
            self.log(
                f"▶️ [Step 3/{last_step}] {self.reduce_engine.upper()} face repair & reduction "
                f"({', '.join(self.reduce_ops)}, budget {self.reduce_quality_budget:g}%)..."
            )
            t_s3 = time.perf_counter()
            faces_before_reduction = len(mesh.faces)
            textured_source = mesh.copy()  # the UV-carrying mesh a collapse would invalidate
            pre_collapse: Dict[str, Any] = {}  # filled by reduce_faces when it collapses edges
            mesh = reduce_faces(
                mesh,
                engine=self.reduce_engine,
                ops=self.reduce_ops,
                quality_budget_percent=self.reduce_quality_budget,
                normal_budget_degrees=self.reduce_normal_budget,
                isolated_min_faces=self.reduce_isolated_min_faces,
                stats=reduce_stats,
                pre_collapse=pre_collapse
            )
            if reduce_stats["uvInvalidated"]:
                density_source = pre_collapse.get("mesh", textured_source)
                uv_projector = SourceUVProjector(textured_source)
                # Collapsing edges returns bare positions and faces, so the model's hard edges -
                # which a glTF mesh carries as split vertex normals, not as geometry - would be
                # averaged away and every crease would light up as if the surface were smooth
                mesh = restore_hard_edges(mesh, uv_projector)
                reduce_stats["hardEdgeVertices"] = int(len(mesh.vertices))
            elif self.preserve_textures:
                preserve_mesh_textures(mesh, orig_tex_info)
            # Single-sided rendering culls a face wound against its own normals, and the hole it
            # leaves reads as a black triangle stuck on the model
            reduce_stats["outwardFix"] = align_faces_outward(mesh)

            # Enforce max_faces policy by level if requested and faces still exceed threshold
            if self.max_faces and len(mesh.faces) > self.max_faces:
                self.log(f"   Enforcing level policy: capping faces {len(mesh.faces):,} -> {self.max_faces:,}")
                if self.reduce_engine == "meshlab":
                    from optimizer.core.face_reduce import _meshlab_merge
                    new_v, new_f = _meshlab_merge(mesh.vertices, mesh.faces, target_faces=self.max_faces)
                else:
                    from optimizer.core.face_reduce import _cgal_merge
                    new_v, new_f = _cgal_merge(mesh.vertices, mesh.faces, target_faces=self.max_faces)
                mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                reduce_stats["facesAfter"] = len(mesh.faces)
                reduce_stats["faceReductionPercent"] = round(((faces_before_reduction - len(mesh.faces)) / faces_before_reduction) * 100, 2)

            step3_file = output_dir / self.STEP_DEFINITIONS[3]["file"]
            # A collapsed mesh has no UV, so its GLB shows the bare geometry the step produced
            step3_file.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": mesh}), include_normals=True))
            if uv_projector is not None:
                # Step 4 bakes into the source model's own glTF PBR material; only its UVs are gone
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=None, material=textured_source.visual.material.copy()
                )
            m3 = save_and_record_metrics(3, step3_file, reduce_stats, t_step_start=t_s3)
            current_file, last_metrics = step3_file, m3
            self.log(
                f"   ✓ Step 3 complete ({m3['durationFormatted']}): "
                f"{reduce_stats['facesBefore']:,} -> {reduce_stats['facesAfter']:,} faces "
                f"(-{reduce_stats['faceReductionPercent']}%), visible-surface deviation "
                f"{reduce_stats['deviation']['maxPercent']}% of the bbox diagonal, shading "
                f"{reduce_stats['deviation']['normalDeviationDegrees']}deg (budget "
                f"{self.reduce_quality_budget:g}% / {self.reduce_normal_budget:g}deg)"
            )
        else:
            skip_step(3)

        # =====================================================================
        # STEP 4: Texture Baking / Resampling & 16px Dilation
        # =====================================================================
        raw_tex_img = orig_tex_info["base_image"]
        orig_w, orig_h = raw_tex_img.size
        original_resolution = f"{orig_w}x{orig_h}"
        if self.enabled(4):
            self._current_step = 4
            t_s4 = time.perf_counter()
            if uv_projector is None:
                # UVs of the mesh as the previous steps left it (the cleaner keeps them aligned
                # with its vertices; the raw mesh's may not be)
                source_uv = getattr(mesh.visual, "uv", None)
                n_uv = 0 if source_uv is None else len(source_uv)
                if n_uv != len(mesh.vertices):
                    raise PipelineAbort(f"The mesh entering Step 4 has {n_uv} UVs for {len(mesh.vertices)} vertices")
            else:
                # Step 3 collapsed edges: every UV comes from the textured mesh it replaced
                source_uv = uv_projector.vertex_uv(mesh)

            # Downscale on: size the re-chart canvas at the source's 1:1 texel density (size_mode), but
            # keep the original UVs & texture when that canvas is not smaller than the original texture.
            # A Step 3 collapse leaves no original UV to keep, so the re-chart is not optional then.
            plan: Optional[Dict[str, Any]] = None
            if self.downscale or uv_projector is not None:
                self.log(
                    f"▶️ [Step 4/{last_step}] Sizing {self.uv_mode.upper()} re-chart canvas at 1:1 texel density "
                    f"(size mode: {self.size_mode})..."
                )
                plan = plan_uv_canvas(
                    mesh,
                    source_image=raw_tex_img,
                    source_uv=source_uv,
                    size_mode=self.size_mode,
                    unwrap_method=self.uv_mode,
                    # After a collapse the texel density of the source texture is the one the
                    # pre-reduction mesh carried, not what the projected UVs of the new faces say
                    density_mesh=density_source if uv_projector is not None else None,
                    density_uv=density_source.visual.uv if uv_projector is not None else None,
                    merge_islands=self.merge_uv_islands
                )
                canvas = plan["final_resolution"]
                if uv_projector is not None:
                    rechart = True
                    decision = (
                        f"rechart (forced by the Step 3 merge): {self.size_mode} {canvas}x{canvas} "
                        f"vs original {original_resolution} (fit {plan['fit_resolution']})"
                    )
                else:
                    rechart = canvas * canvas < orig_w * orig_h
                    decision = (
                        f"{'rechart' if rechart else 'kept_original'}: {self.size_mode} {canvas}x{canvas} "
                        f"{'<' if rechart else '>='} original {original_resolution} (fit {plan['fit_resolution']})"
                    )
            else:
                self.log(f"▶️ [Step 4/{last_step}] Downscale off: keeping original UVs & texture...")
                rechart = False
                decision = "kept_original: downscale off"
            self.log(f"   {decision}")

            if rechart:
                baked_mesh, dilated_pil, uv_stats = bake_uv_plan(
                    mesh,
                    plan,
                    source_image=raw_tex_img,
                    source_uv=source_uv,
                    dilation_padding=16,
                    double_sided=self.double_sided,
                    source_uv_sampler=uv_projector
                )
                pref_fmt = "PNG"  # PNG Lossless export for Step 4
                texel_density_ratio = uv_stats["texel_density_ratio"]
            else:
                # Previous step's mesh as is (original UVs & faces) with the original texture bitstream
                baked_mesh = mesh
                preserve_mesh_textures(baked_mesh, orig_tex_info)
                dilated_pil = raw_tex_img
                pref_fmt = "ORIGINAL"
                uv_stats = {}
                texel_density_ratio = 1.0
            mesh = baked_mesh

            # FrontSide rendering (doubleSided=False by default)
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                mesh.visual.material.doubleSided = self.double_sided

            # Optimize texture before export (PNG Lossless or Original Bitstream Passthrough)
            dilated_pil = optimize_mesh_texture_for_export(mesh, preferred_format=pref_fmt)

            step4_bytes = trimesh.exchange.gltf.export_glb(trimesh.Scene({"Model": mesh}), include_normals=True)
            if self.double_sided:
                step4_bytes = set_doublesided_material(step4_bytes)
            else:
                step4_bytes = set_frontside_material(step4_bytes)
            step4_file = output_dir / self.STEP_DEFINITIONS[4]["file"]
            step4_file.write_bytes(step4_bytes)
            tex_fmt = getattr(dilated_pil, "format", "PNG")
            texture_resolution = f"{dilated_pil.size[0]}x{dilated_pil.size[1]}"

            step4_extra = {
                "uvMode": self.uv_mode,
                "downscale": self.downscale,
                "sizeMode": self.size_mode,
                "fitResolution": plan["fit_resolution"] if plan is not None else None,
                "finalResolution": texture_resolution,
                "originalResolution": original_resolution,
                "downscaled": rechart,
                "decision": decision,
                "texelDensityRatio": texel_density_ratio,
                "textureResolution": texture_resolution,
                "textureFormat": tex_fmt,
                "dilationPadding": uv_stats["dilation_padding"] if rechart else None,
                "uvCoverageRatio": uv_stats["uvCoverageRatio"] if rechart else None,
                "doubleSided": self.double_sided,
                "mergeUvIslands": self.merge_uv_islands,
                "islandMerge": plan["island_merge"] if plan is not None else None,
                "uvMetrics": uv_stats
            }
            m4 = save_and_record_metrics(4, step4_file, step4_extra, t_step_start=t_s4)
            current_file, last_metrics = step4_file, m4
            self.log(
                f"   ✓ Step 4 complete ({m4['durationFormatted']}): Texture {texture_resolution} ({tex_fmt}) | "
                f"Downscaled: {rechart} ({original_resolution} -> {texture_resolution}) | "
                f"Texel density ratio: {texel_density_ratio:.3f}"
            )
            if plan is not None:
                islands = plan["island_merge"]
                self.log(
                    f"   UV islands: {islands['islandsAfter']} (merge level {islands['chosenLevel']} of "
                    f"{len(islands['levels'])} tried), island border {islands['boundaryTexelsAfter']:,.0f} texels "
                    f"({islands['boundaryReductionPercent']:+.2f}% vs unmerged)"
                )
        else:
            skip_step(4)
            # No bake: the model keeps the texture it came in with
            dilated_pil = raw_tex_img
            texture_resolution = original_resolution

        # Steps 6-7 never resize: the max dimension is the current texture's own
        texture_max_dim = max(dilated_pil.size)

        # =====================================================================
        # STEP 5: Palette Tagging (10 Dominant Colors via K-Means)
        # =====================================================================
        palette_data: Optional[Dict[str, Any]] = None
        if self.enabled(5):
            self._current_step = 5
            self.log(f"▶️ [Step 5/{last_step}] Extracting 10 Dominant Colors & Embedding Extras...")
            t_s5 = time.perf_counter()
            palette_uv = getattr(mesh.visual, "uv", None)
            if palette_uv is None or len(palette_uv) != len(mesh.vertices):
                raise PipelineAbort(
                    f"Step 5 needs one UV per vertex to sample the model's surface colours, but the "
                    f"mesh has {0 if palette_uv is None else len(palette_uv)} UVs for {len(mesh.vertices)} vertices"
                )
            img_rgb = np.asarray(dilated_pil)
            v_uv = np.asarray(palette_uv) % 1.0
            px = np.clip((v_uv[:, 0] * (img_rgb.shape[1] - 1)).astype(int), 0, img_rgb.shape[1] - 1)
            py = np.clip(((1.0 - v_uv[:, 1]) * (img_rgb.shape[0] - 1)).astype(int), 0, img_rgb.shape[0] - 1)
            surface_pixels = img_rgb[py, px]

            palette_data = extract_palette(image=dilated_pil, sample_pixels=surface_pixels, n_colors=10)
            step5_bytes = embed_gltf_extras(current_file.read_bytes(), {
                "palette": palette_data["palette"],
                "primaryColor": palette_data["primaryColor"],
                "paletteDetails": palette_data["paletteDetails"]
            })
            step5_file = output_dir / self.STEP_DEFINITIONS[5]["file"]
            step5_file.write_bytes(step5_bytes)
            m5 = save_and_record_metrics(5, step5_file, {
                "primaryColor": palette_data["primaryColor"],
                "paletteCount": len(palette_data["palette"])
            }, t_step_start=t_s5)
            current_file, last_metrics = step5_file, m5
            self.log(f"   ✓ Step 5 complete ({m5['durationFormatted']}): Primary color: {palette_data['primaryColor']} | Palette: {palette_data['palette']}")
        else:
            skip_step(5)

        # =====================================================================
        # STEP 6: Smooth Normals & EXT_meshopt_compression Geometry
        # =====================================================================
        if self.enabled(6):
            self._current_step = 6
            self.log(f"▶️ [Step 6/{last_step}] Node.js Smooth Normals + Weld + Quantize + Meshopt...")
            t_s6 = time.perf_counter()
            step6_file = output_dir / self.STEP_DEFINITIONS[6]["file"]
            node_cmd_step6 = [
                "node", str(NODE_OPT_SCRIPT),
                str(current_file),
                str(step6_file),
                "--no-ktx2",
                "--pos-bits", "14",
                "--weld", "0.0001",
                "--reorder",
                "--meshopt",
                "--texture-max-dim", str(texture_max_dim),
                "--json"
            ]
            if self.smooth_normals:
                node_cmd_step6.append("--smooth-normals")
            else:
                node_cmd_step6.append("--no-smooth-normals")

            if not self.double_sided:
                node_cmd_step6.append("--single-sided")
            else:
                node_cmd_step6.append("--keep-double-sided")

            proc6 = subprocess.run(node_cmd_step6, capture_output=True, text=True)
            if proc6.returncode != 0:
                raise node_step_failure("Step 6: meshopt geometry compression failed", proc6)
            if not step6_file.exists():
                raise PipelineAbort(f"Step 6: meshopt geometry compression wrote no {step6_file.name}")

            m6 = save_and_record_metrics(
                6, step6_file, {"smoothNormals": self.smooth_normals}, t_step_start=t_s6
            )
            current_file, last_metrics = step6_file, m6
            self.log(
                f"   ✓ Step 6 complete ({m6['durationFormatted']}): Geometry compressed "
                f"({m6['faces']:,} faces, {m6['fileSizeFormatted']}), "
                f"smooth normals {'ON' if self.smooth_normals else 'OFF'}"
            )
        else:
            skip_step(6)

        # Small textures: KTX2 UASTC grows the file while the uncompressed VRAM is already small,
        # so Step 7 keeps the incoming textures when the texture VRAM estimate is below the threshold.
        texture_vram_bytes = last_metrics["totalGpuVramBytes"]
        texture_vram_mb = texture_vram_bytes / (1024 * 1024)
        skip_ktx2 = self.texture_format == "ktx2" and texture_vram_mb < self.ktx2_min_vram_mb
        vram_vs_threshold = f"{texture_vram_mb:.2f} MB {'<' if skip_ktx2 else '>='} {self.ktx2_min_vram_mb:g} MB"
        ktx2_reason = f"texture VRAM {vram_vs_threshold}"
        if self.texture_format == "ktx2":
            self.log(
                f"   Texture VRAM estimate: {vram_vs_threshold} -> "
                f"{'Step 7 KTX2 will be skipped' if skip_ktx2 else 'Step 7 KTX2 UASTC'}"
            )

        # =====================================================================
        # STEP 7: KTX2 / WebP / Original GPU Compression, FrontSide Material, Extras
        # =====================================================================
        self._current_step = 7
        t_s7 = time.perf_counter()
        is_original_format = self.texture_format in ("original", "passthrough", "raw")
        if is_original_format:
            self.log(f"▶️ [Step 7/{last_step}] Finalizing Model (Original Texture Bitstream Pass-through 100% Lossless)...")
            final_bytes = current_file.read_bytes()
        elif skip_ktx2:
            self.log(
                f"▶️ [Step 7/{last_step}] KTX2 GPU Texture Compression SKIPPED ({ktx2_reason}): "
                f"final model keeps the incoming textures..."
            )
            final_bytes = current_file.read_bytes()
        else:
            self.log(f"▶️ [Step 7/{last_step}] Basis Universal GPU Texture Compression ({self.texture_format.upper()})...")
            step7_temp_file = output_dir / "step_07_temp.glb"
            node_cmd_step7 = [
                "node", str(NODE_OPT_SCRIPT),
                str(current_file),
                str(step7_temp_file),
                "--textures-only",
                "--meshopt",
                "--texture-max-dim", str(texture_max_dim),
                "--json"
            ]

            if self.texture_format == "webp":
                node_cmd_step7.extend(["--webp", "--webp-quality", "85"])
            else:
                cpu_threads = str(os.cpu_count() or 4)
                node_cmd_step7.extend([
                    "--ktx2",
                    "--ktx2-mode", "uastc",
                    "--ktx2-level", "2",
                    "--ktx2-rdo", "1.0",
                    "--ktx2-rdo-d", "2048",
                    "--ktx2-threads", cpu_threads
                ])

            if not self.double_sided:
                node_cmd_step7.append("--single-sided")
            else:
                node_cmd_step7.append("--keep-double-sided")

            step7_fmt = "webp" if self.texture_format == "webp" else "ktx2"
            proc7 = subprocess.run(node_cmd_step7, capture_output=True, text=True)
            if proc7.returncode != 0:
                raise node_step_failure(f"Step 7: {step7_fmt.upper()} texture compression failed", proc7)
            if not step7_temp_file.exists():
                raise PipelineAbort(f"Step 7: {step7_fmt.upper()} texture compression wrote no {step7_temp_file.name}")
            encoded_texture_summary(proc7, step7_fmt)

            final_bytes = step7_temp_file.read_bytes()
            step7_temp_file.unlink(missing_ok=True)

        if not self.double_sided:
            final_bytes = set_frontside_material(final_bytes)

        final_extras = {
            "resolution": texture_resolution,
            "texture_format": self.texture_format.upper(),
            "policy": "STRICT 0-DECIMATION (--ratio 1.0)" if not self.enabled(3)
                      else f"FACE REDUCTION (Step 3 {self.reduce_engine}, quality budget {self.reduce_quality_budget:g}%)",
            "tool": "poc-optimize-3d-model v1.0.0"
        }
        # Only a Step 5 that ran has a palette to carry into the final model
        if palette_data is not None:
            final_extras["palette"] = palette_data["palette"]
            final_extras["primaryColor"] = palette_data["primaryColor"]
            final_extras["paletteDetails"] = palette_data["paletteDetails"]
        step7_extra = None
        if self.texture_format == "ktx2":
            if skip_ktx2:
                # The final file keeps the incoming textures: record their real format(s), never "KTX2"
                final_extras["texture_format"] = "+".join(dict.fromkeys(t["format"] for t in last_metrics["textures"]))
                final_extras["gpu_compression"] = f"skipped: {ktx2_reason}"
                texture_format_label = final_extras["texture_format"]
            else:
                final_extras["gpu_compression"] = "ktx2 uastc"
            step7_extra = {
                "gpuCompressionSkipped": skip_ktx2,
                "gpuCompressionReason": ktx2_reason,
                "textureVramEstimateBytes": texture_vram_bytes
            }
        final_bytes = embed_gltf_extras(final_bytes, final_extras)
        step7_file = output_dir / self.STEP_DEFINITIONS[7]["file"]
        step7_file.write_bytes(final_bytes)

        m7 = save_and_record_metrics(7, step7_file, step7_extra, t_step_start=t_s7)
        self.log(f"   ✓ Step 7 complete ({m7['durationFormatted']}): Final GLB ready ({m7['fileSizeFormatted']}, GPU VRAM: {m7['totalGpuVramFormatted']})")

        # =====================================================================
        # Pipeline Summary & Finalization
        # =====================================================================
        total_perf_elapsed = time.perf_counter() - t_total_start
        elapsed = time.time() - t0
        final_bytes_count = m7["fileSizeBytes"]
        saved_bytes = initial_bytes - final_bytes_count
        saved_pct = round((saved_bytes / initial_bytes) * 100, 2)
        initial_vram = m0.get("totalGpuVramBytes", 0)
        final_vram = m7.get("totalGpuVramBytes", 0)
        vram_saved_pct = round((1 - final_vram / initial_vram) * 100, 2) if initial_vram > 0 else 0.0
        # Faces the pipeline is allowed to lose: only Step 3 removes any, and only within its
        # quality budget. Every later step still preserves 100% of what Step 3 left.
        reduced_faces = reduce_stats.get("facesAfter", faces_before_reduction)

        summary = {
            "model": input_path.name,
            "outputDir": str(output_dir),
            "elapsedSeconds": round(total_perf_elapsed, 3),
            "elapsedFormatted": format_duration(total_perf_elapsed),
            "elapsedMs": round(total_perf_elapsed * 1000, 1),
            "totalDurationSeconds": round(total_perf_elapsed, 3),
            "totalDurationFormatted": format_duration(total_perf_elapsed),
            "stepDurations": {
                s["stepName"]: {
                    "step": s["step"],
                    "file": s["file"],
                    "skipped": s["skipped"],
                    "durationSeconds": s.get("durationSeconds", 0.0),
                    "durationFormatted": s.get("durationFormatted", "0s"),
                    "totalDurationSeconds": s.get("totalDurationSeconds", 0.0),
                    "seconds": s.get("durationSeconds", 0.0),
                    "ms": s.get("durationMs", 0.0)
                }
                for s in steps_record
            },
            "skippedSteps": skipped_steps,
            "initialSizeBytes": initial_bytes,
            "finalSizeBytes": final_bytes_count,
            "savedBytes": saved_bytes,
            "savedPercent": saved_pct,
            "initialFaces": initial_faces,
            "finalFaces": m7["faces"],
            "faceReduction": {
                "enabled": self.enabled(3),
                "engine": self.reduce_engine if self.enabled(3) else None,
                "ops": list(self.reduce_ops) if self.enabled(3) else [],
                "facesBefore": faces_before_reduction,
                "facesAfter": reduced_faces,
                "facesRemoved": faces_before_reduction - reduced_faces,
                "percent": reduce_stats.get("faceReductionPercent", 0.0),
                "qualityBudgetPercent": self.reduce_quality_budget if self.enabled(3) else None,
                "normalBudgetDegrees": self.reduce_normal_budget if self.enabled(3) else None,
                # The budget holds at the percentile the step measures at; the worst point
                # anywhere is reported next to it rather than being what decides
                "deviationPercent": reduce_stats.get("deviation", {}).get("percentilePercent"),
                "deviationMaxPercent": reduce_stats.get("deviation", {}).get("maxPercent"),
                "normalDeviationDegrees": reduce_stats.get("deviation", {}).get("normalDeviationDegrees"),
                "withinQualityBudget": (
                    within_budget(
                        reduce_stats["deviation"], self.reduce_quality_budget, self.reduce_normal_budget
                    ) if reduce_stats.get("deviation") else None
                )
            },
            # Rule 11 after the face reduction step: no later step may drop a single triangle
            "facesPreservedPercent": round((m7["faces"] / reduced_faces) * 100, 2) if reduced_faces else 0.0,
            "zeroDecimationVerified": m7["faces"] == reduced_faces,
            "initialVertices": initial_verts,
            "finalVertices": m7["vertices"],
            "initialGpuVramBytes": initial_vram,
            "finalGpuVramBytes": final_vram,
            "gpuVramSavedPercent": vram_saved_pct,
            "primaryColor": palette_data["primaryColor"] if palette_data else None,
            "palette": palette_data["palette"] if palette_data else [],
            "files": [step["file"] for step in self.STEP_DEFINITIONS if self.enabled(step["step"])]
        }

        final_payload = {
            "success": True,
            "summary": summary,
            "steps": steps_record
        }
        metrics_json_path.write_text(json.dumps(final_payload, indent=2))

        # Emit pipeline_completed event
        if self.stream_events:
            print(json.dumps({"event": "pipeline_complete", "summary": summary}), flush=True)

        self.log("=" * 68)
        self.log(f"🎉 PIPELINE COMPLETED IN {total_perf_elapsed:.2f}s!")
        self.log(f"   Size: {m0['fileSizeFormatted']} -> {m7['fileSizeFormatted']} (Saved {saved_pct}%)")
        self.log(f"   GPU VRAM: {m0['totalGpuVramFormatted']} -> {m7['totalGpuVramFormatted']} (Saved {vram_saved_pct}%)")
        if self.enabled(3):
            self.log(
                f"   Triangles: {initial_faces:,} -> {reduced_faces:,} at Step 3 "
                f"(-{reduce_stats.get('faceReductionPercent', 0.0)}%), then preserved 100%: {summary['zeroDecimationVerified']}"
            )
        else:
            self.log(f"   Triangles: {m7['faces']:,} / {initial_faces:,} (100% Zero-Decimation Verified: {summary['zeroDecimationVerified']})")
        self.log(f"   Metrics saved to: {metrics_json_path}")
        self.log("=" * 68)

        return final_payload


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Decimation Step-by-Step 3D Model Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Path to raw source .glb model")
    parser.add_argument("--output-dir", "-o", required=True, help="Destination directory for the GLB step files and metrics.json")
    parser.add_argument("--format", "-f", choices=["ktx2", "webp", "original", "passthrough"], default="ktx2", help="GPU texture compression format ('original' for 100%% bit-for-bit lossless pass-through)")
    parser.add_argument(
        "--uv-mode",
        choices=["xatlas", "uvatlas"],
        default="xatlas",
        help="UV unwrapping and layout mode: 'xatlas' (re-chart with xatlas), 'uvatlas' (Microsoft UVAtlas isochart unwrap)"
    )
    parser.add_argument(
        "--downscale",
        choices=["on", "off"],
        default="on",
        help="'on': re-chart & pack UV islands on a canvas sized by --size-mode (original kept when that canvas "
             "is not smaller than the original texture); 'off': keep original UVs, texture and resolution"
    )
    parser.add_argument(
        "--size-mode",
        choices=list(SIZE_MODES),
        default="exact",
        help="Re-chart canvas at the source's 1:1 texel density: 'exact' (smallest square, multiple of 4), "
             "'pot-up' (power of two, 1:1 or better), 'pot-down' (power of two, islands scaled down)"
    )
    parser.add_argument(
        "--merge-uv-islands",
        choices=["on", "off"],
        default="on",
        help="Step 4: chart the mesh at several island-merge levels and keep the one whose islands "
             "need the smallest canvas. Fewer islands mean less chart border, hence less of the "
             "texture spent on gutter padding ('off' charts once, with the default segmentation)"
    )
    parser.add_argument("--smooth-normals", dest="smooth_normals", action="store_true", default=None, help="Force angle-weighted normal smoothing across seams")
    parser.add_argument("--no-smooth-normals", dest="smooth_normals", action="store_false", default=None, help="Disable angle-weighted normal smoothing across seams")
    parser.add_argument("--double-sided", action="store_true", help="Keep double-sided materials instead of forcing single-sided FrontSide")
    parser.add_argument("--no-preserve-textures", action="store_true", help="Disable texture preservation in Steps 1 & 2")
    parser.add_argument(
        "--skip-steps",
        default="",
        help="Comma-separated indices of the steps to skip, e.g. '3,5' (optional steps: "
             f"{', '.join(str(n) for n in StepPipeline.OPTIONAL_STEPS)}; Step 0 and the final step always run)"
    )
    parser.add_argument(
        "--reduce-engine",
        choices=list(REDUCE_ENGINES),
        default="cgal",
        help="Step 3 face repair & reduction engine: 'cgal' (the optimizer/cgal mesh_repair helper, "
             "Polygon Mesh Processing) or 'meshlab' (pymeshlab filters, which reduces far less at "
             "a tight quality budget)"
    )
    parser.add_argument(
        "--reduce-ops",
        default=",".join(DEFAULT_REDUCE_OPS),
        help=f"Comma-separated Step 3 operations, in any order (available: {', '.join(REDUCE_OPS)})"
    )
    parser.add_argument(
        "--reduce-quality-budget",
        type=float,
        default=DEFAULT_QUALITY_BUDGET_PERCENT,
        help="Step 3 quality budget: how far the visible surface may move, as a percentage of the "
             "model's bounding box diagonal"
    )
    parser.add_argument(
        "--reduce-normal-budget",
        type=float,
        default=DEFAULT_NORMAL_BUDGET_DEGREES,
        help="Step 3 shading budget: how far the surface may turn away from the original, in "
             "degrees at the 99th percentile. A collapse can round a crease away while barely "
             "moving the surface, so this is what keeps narrow slots and sharp edges"
    )
    parser.add_argument(
        "--reduce-isolated-min-faces",
        type=int,
        default=DEFAULT_ISOLATED_MIN_FACES,
        help="Step 3 'isolated' operation: connected components with fewer faces than this are removed"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Step 3 maximum face cap (enforces level-based budget e.g. 50k for L1/L2, 100k for L3, 300k for L4/L5)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress stderr logs and only stream NDJSON events")

    args = parser.parse_args()

    def parse_list(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    try:
        skip_steps = []
        for item in parse_list(args.skip_steps):
            if not item.lstrip("-").isdigit():
                raise ValueError(f"--skip-steps expects step indices, got '{item}'")
            skip_steps.append(int(item))

        pipeline = StepPipeline(
            texture_format=args.format,
            uv_mode=args.uv_mode,
            downscale=(args.downscale == "on"),
            size_mode=args.size_mode,
            merge_uv_islands=(args.merge_uv_islands == "on"),
            smooth_normals=args.smooth_normals,
            double_sided=args.double_sided,
            preserve_textures=not args.no_preserve_textures,
            verbose=not args.quiet,
            stream_events=True,
            skip_steps=skip_steps,
            reduce_engine=args.reduce_engine,
            reduce_ops=parse_list(args.reduce_ops),
            reduce_quality_budget=args.reduce_quality_budget,
            reduce_normal_budget=args.reduce_normal_budget,
            reduce_isolated_min_faces=args.reduce_isolated_min_faces,
            max_faces=args.max_faces
        )
        pipeline.run(Path(args.input), Path(args.output_dir))
    except Exception as e:
        # Full traceback for debugging on stderr; one parseable event (reason, type, step) on stdout
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"event": "pipeline_error", **describe_failure(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
