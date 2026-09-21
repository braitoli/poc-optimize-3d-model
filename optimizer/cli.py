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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-Decimation 3D Model (.glb) Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", type=str, help="Path to raw input .glb file")
    parser.add_argument("output", type=str, help="Path to output optimized .glb file")
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=1024,
        choices=[512, 1024, 2048, 4096],
        help="Target texture dimension"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="ktx2",
        choices=["ktx2", "webp"],
        help="GPU texture compression format"
    )
    parser.add_argument(
        "--rechart",
        action="store_true",
        help="Re-chart and repack UV atlas with xatlas (default: preserve master UV)"
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

    optimizer = ModelOptimizer(
        resolution=args.resolution,
        texture_format=args.format,
        rechart_uv=args.rechart,
        smooth_normals=not args.no_smooth_normals,
        double_sided=args.double_sided,
        verbose=not args.quiet and not args.json
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
