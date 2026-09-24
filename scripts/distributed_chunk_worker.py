#!/usr/bin/env python3
"""
distributed_chunk_worker.py — Distributed Batch Worker for 3D Model Optimization

Executes a chunk of 3D models using the zero-decimation optimization pipeline
(bin/optimize-3d / optimizer.step_pipeline).

Usage:
    python distributed_chunk_worker.py --chunk <chunk.json> [options]

Chunk JSON format:
    {
      "batch_id": "chunk_gx10_01",
      "items": [
        {
          "id": "model_001",
          "input": "/path/to/input.glb",
          "output": "/path/to/output.glb",
          "options": ["--format", "ktx2", "--downscale", "on"]  # optional
        },
        ...
      ]
    }
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZE_CLI = REPO_ROOT / "bin/optimize-3d"
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"

def optimize_single_model(item: dict, default_options: list) -> dict:
    model_id = item.get("id", Path(item["input"]).stem)
    input_path = Path(item["input"]).resolve()
    output_path = Path(item["output"]).resolve()
    options = item.get("options", default_options)
    
    t0 = time.time()
    result = {
        "id": model_id,
        "input": str(input_path),
        "output": str(output_path),
        "status": "pending",
        "duration_seconds": 0.0,
        "error": None,
        "input_size_bytes": 0,
        "output_size_bytes": 0,
    }
    
    if not input_path.exists():
        result["status"] = "failed"
        result["error"] = f"Input file not found: {input_path}"
        return result
        
    result["input_size_bytes"] = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup environment
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["PATH"] = f"/Users/braitoli/.nvm/versions/node/v24.18.0/bin:/opt/homebrew/bin:{env.get('PATH', '')}"
    else:
        env["PATH"] = f"/usr/local/bin:{env.get('PATH', '')}"
        
    cmd = [str(OPTIMIZE_CLI), str(input_path), str(output_path)] + options
    
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=600  # 10 min max per model
        )
        duration = round(time.time() - t0, 2)
        result["duration_seconds"] = duration
        
        if proc.returncode == 0 and output_path.exists():
            result["status"] = "completed"
            result["output_size_bytes"] = output_path.stat().st_size
        else:
            result["status"] = "failed"
            # Get last 10 lines of output for error diagnostics
            tail_lines = proc.stdout.strip().splitlines()[-10:] if proc.stdout else []
            result["error"] = "\n".join(tail_lines)
            
        # Immediate cleanup of stale temp files to protect disk space (crucial for M5 80% load)
        clean_tmp_cmd = (
            "find /private/var/folders /var/folders /tmp -maxdepth 4 -name 'optimize-3d.*' -mmin +2 -exec rm -rf {} + 2>/dev/null"
            if sys.platform == "darwin" else
            "find /tmp -maxdepth 1 -name 'optimize-3d.*' -mmin +2 -exec rm -rf {} + 2>/dev/null"
        )
        try:
            subprocess.run(clean_tmp_cmd, shell=True, timeout=5)
        except Exception:
            pass
    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error"] = "Process timed out after 600 seconds"
        result["duration_seconds"] = round(time.time() - t0, 2)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["duration_seconds"] = round(time.time() - t0, 2)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Distributed Chunk Worker for 3D Optimization")
    parser.add_argument("--chunk", required=True, help="Path to chunk JSON file")
    parser.add_argument("--progress", default=None, help="Path to progress JSON file")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--format", default="ktx2", choices=["ktx2", "webp", "original"], help="Default texture format")
    parser.add_argument("--downscale", default="on", choices=["on", "off"], help="Texture downscale")
    args = parser.parse_args()
    
    chunk_path = Path(args.chunk).resolve()
    if not chunk_path.exists():
        print(f"❌ Chunk file not found: {chunk_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
        
    batch_id = chunk_data.get("batch_id", chunk_path.stem)
    items = chunk_data.get("items", [])
    
    progress_path = Path(args.progress).resolve() if args.progress else chunk_path.parent / f"{batch_id}_progress.json"
    
    # Load or initialize progress
    state = {
        "batch_id": batch_id,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_items": len(items),
        "completed": 0,
        "failed": 0,
        "items": {}
    }
    
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass
            
    default_options = ["--format", args.format, "--downscale", args.downscale]
    
    remaining_items = [
        item for item in items
        if state["items"].get(item.get("id", Path(item["input"]).stem), {}).get("status") != "completed"
    ]
    
    print(f"==================================================================")
    print(f"🚀 DISTRIBUTED WORKER: {batch_id}")
    print(f"   Total items: {len(items)} | Remaining: {len(remaining_items)}")
    print(f"   Concurrency: {args.concurrency} parallel workers")
    print(f"   Progress file: {progress_path}")
    print(f"==================================================================")
    
    if not remaining_items:
        print("✅ All items in chunk are already completed!")
        sys.exit(0)
        
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(optimize_single_model, item, default_options): item
            for item in remaining_items
        }
        
        for future in as_completed(future_map):
            res = future.result()
            m_id = res["id"]
            state["items"][m_id] = res
            
            if res["status"] == "completed":
                state["completed"] = sum(1 for v in state["items"].values() if v.get("status") == "completed")
                saved_pct = round((1 - res["output_size_bytes"] / max(res["input_size_bytes"], 1)) * 100, 1)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [{state['completed']}/{len(items)}] {m_id} "
                      f"({res['duration_seconds']}s, {res['input_size_bytes']/1024/1024:.2f}MB -> {res['output_size_bytes']/1024/1024:.2f}MB, -{saved_pct}%)", flush=True)
            else:
                state["failed"] = sum(1 for v in state["items"].values() if v.get("status") == "failed")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [{state['failed']} FAILED] {m_id}: {res['error']}", flush=True)
                
            state["updated_at"] = datetime.now().isoformat()
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                
    print(f"==================================================================")
    print(f"🎉 BATCH COMPLETE: {state['completed']} succeeded, {state['failed']} failed.")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
