#!/usr/bin/env python3
"""Plot benchmark metrics from guidellm JSON results.

Uses guidellm's built-in GenerativeBenchmarksReport and GenerativeBenchmarkerPlot
to generate a performance visualization from all JSON result files.

Usage:
    python3 plot.py [--results-dir ./results] [--output-dir ./export]
    python3 plot.py --file export/export_full_20260812.csv  # (future: plot from CSV)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path


def plot_from_json_results(results_dir: Path, output_path: Path) -> int:
    """Load all JSON results and generate a combined plot using guidellm's plotter."""
    from guidellm.benchmark import GenerativeBenchmarksReport
    from guidellm.benchmark.outputs.plot import GenerativeBenchmarkerPlot

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"[error] No JSON files found in {results_dir}", file=sys.stderr)
        return 1

    # Load all reports and merge benchmarks
    all_benchmarks = []
    base_report = None

    for jf in json_files:
        print(f"[load] {jf.name}")
        try:
            report = GenerativeBenchmarksReport.load_file(jf)
            if base_report is None:
                base_report = report
            all_benchmarks.extend(report.benchmarks)
        except Exception as exc:
            print(f"[warn] Failed to load {jf.name}: {exc}", file=sys.stderr)
            continue

    if not all_benchmarks or base_report is None:
        print("[error] No valid benchmarks loaded.", file=sys.stderr)
        return 1

    # Create a merged report with all benchmarks
    base_report.benchmarks = all_benchmarks

    # Generate plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = GenerativeBenchmarkerPlot(output_path=output_path, dpi=150)

    print(f"[plot] Generating plot with {len(all_benchmarks)} benchmarks...")
    asyncio.run(plotter.finalize(base_report))
    print(f"[plot] Saved -> {output_path}")
    return 0


def plot_from_file(json_path: Path, output_path: Path) -> int:
    """Load a single merged JSON report and generate a plot."""
    from guidellm.benchmark import GenerativeBenchmarksReport
    from guidellm.benchmark.outputs.plot import GenerativeBenchmarkerPlot

    print(f"[load] {json_path}")
    try:
        report = GenerativeBenchmarksReport.load_file(json_path)
    except Exception as exc:
        print(f"[error] Failed to load {json_path}: {exc}", file=sys.stderr)
        return 1

    if not report.benchmarks:
        print("[error] No benchmarks in report.", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = GenerativeBenchmarkerPlot(output_path=output_path, dpi=150)

    print(f"[plot] Generating plot with {len(report.benchmarks)} benchmarks...")
    asyncio.run(plotter.finalize(report))
    print(f"[plot] Saved -> {output_path}")
    return 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Plot guidellm benchmark results")
    parser.add_argument("--results-dir", type=Path, default=script_dir / "results",
                        help="Directory containing guidellm JSON results (default: auto/results)")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "export",
                        help="Output directory for plot image (default: auto/export)")
    parser.add_argument("--file", type=Path, default=None,
                        help="Plot from a specific JSON file (e.g. export/export_full_xxx.json)")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"plot_{timestamp}.png"

    if args.file:
        file_path = args.file.resolve()
        if not file_path.exists():
            print(f"[error] File not found: {file_path}", file=sys.stderr)
            return 1
        return plot_from_file(file_path, output_path)

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    return plot_from_json_results(results_dir, output_path)


if __name__ == "__main__":
    sys.exit(main())
