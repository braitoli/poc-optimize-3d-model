#!/usr/bin/env python3
"""
cli.py

Command-line interface for 3D model optimization.
"""

import sys
import json
import argparse
from pathlib import Path
from optimizer.pipeline import ModelOptimizer


def parse_resolution_arg(val):
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    try:
        res = int(s)
        return res if res > 0 else "auto"
    except ValueError:
        return "auto"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-Decimation 3D Model (.glb) Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", type=str, help="Path to raw input .glb file")
    parser.add_argument("output", type=str, help="Path to output optimized .glb file")
    parser.add_argument(
        "-r", "--resolution",
        default="auto",
        type=parse_resolution_arg,
        help="Target texture dimension ('auto', 512, 1024, 2048; strictly capped at original texture size, never upscaled)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="ktx2",
        choices=["ktx2", "webp"],
        help="GPU texture compression format"
    )
    parser.add_argument(
        "--no-smooth-normals",
        action="store_true",
        help="Disable angle-weighted smooth vertex normals across UV seams"
    )
    parser.add_argument(
        "--double-sided",
        action="store_true",
        help="Keep doubleSided material (default: FrontSide single-sided)"
    )
    parser.add_argument(
        "--export-steps",
        type=str,
        default=None,
        help="Directory path to export intermediate step models (steps 0 to 6)"
    )
    parser.add_argument(
        "--step-events",
        action="store_true",
        help="Print real-time JSON step events (__STEP_EVENT__:<json>) to stdout"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: suppress informational logs"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    def on_step_event(step_data):
        print(f"__STEP_EVENT__:{json.dumps(step_data)}", flush=True)

    optimizer = ModelOptimizer(
        resolution=args.resolution,
        texture_format=args.format,
        smooth_normals=not args.no_smooth_normals,
        double_sided=args.double_sided,
        verbose=not args.quiet and not args.json,
        export_steps_dir=Path(args.export_steps) if args.export_steps else None,
        step_callback=on_step_event if args.step_events else None
    )

    try:
        summary = optimizer.optimize(input_path, output_path)
        if args.json:
            print(json.dumps(summary, indent=2))
    except Exception as exc:
        print(f"Error during optimization: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
