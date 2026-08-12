#!/usr/bin/env python3
"""Export benchmark results from guidellm CSV outputs into a single summary CSV.

Usage:
    python3 export.py [--results-dir ./results] [--output-dir ./export]

Extracts key metrics from all CSV files in results dir and merges into one CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

# Columns we want to extract (group + field patterns from guidellm CSV)
# The guidellm CSV uses multi-row headers: [group, field, unit/stat]
# We target metrics similar to vllm benchmark_serving output.
DESIRED_METRICS = [
    # Request counts
    ("Request Counts", "Successful"),
    ("Request Counts", "Total"),
    # Timing
    ("Timings", "Duration"),
    # Throughput
    ("Server Throughput", "Requests/Sec"),
    ("Server Throughput", "Concurrency"),
    ("Token Throughput", "Output Tokens/Sec"),
    ("Token Throughput", "Total Tokens/Sec"),
    ("Token Throughput", "Input Tokens/Sec"),
    # Token counts
    ("Token Metrics", "Input Tokens"),
    ("Token Metrics", "Output Tokens"),
    ("Token Metrics", "Total Tokens"),
    # Latency - TTFT
    ("Time to First Token", "ms"),
    # Latency - TPOT
    ("Time per Output Token", "ms"),
    # Latency - ITL
    ("Inter Token Latency", "ms"),
    # Request latency
    ("Request Latency", "Sec"),
]

# For distribution metrics, we want these stats
STAT_TYPES = ["Mean", "Median", "Std Dev"]


def parse_guidellm_csv(csv_path: Path) -> dict[str, str | float]:
    """Parse a guidellm multi-row-header CSV and extract desired metrics.

    Returns a flat dict of metric_name -> value.
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 4:
        return {}

    # guidellm CSV has 3 header rows then data rows
    header_row1 = rows[0]  # group
    header_row2 = rows[1]  # field/unit
    header_row3 = rows[2]  # stat (Mean, Median, etc.)
    data_rows = rows[3:]

    # Build column index: (group, field, stat) -> col_index
    col_count = len(header_row1)
    columns: list[tuple[str, str, str]] = []
    for i in range(col_count):
        g = header_row1[i] if i < len(header_row1) else ""
        f = header_row2[i] if i < len(header_row2) else ""
        s = header_row3[i] if i < len(header_row3) else ""
        columns.append((g, f, s))

    results: dict[str, str | float] = {}
    # Add source filename
    results["source_file"] = csv_path.stem

    # Extract service/test/concurrency from filename pattern:
    # SERVICE_NAME.tpTP.TEST_NAME.cCONCURRENCY.csv
    stem = csv_path.stem
    m = re.match(r"^(.+?)\.tp(\d+)\.(.+?)\.c(\d+)$", stem)
    if m:
        results["service"] = m.group(1)
        results["tp"] = m.group(2)
        results["test"] = m.group(3)
        results["concurrency"] = m.group(4)
    else:
        results["service"] = stem
        results["tp"] = ""
        results["test"] = ""
        results["concurrency"] = ""

    if not data_rows:
        return results

    # Use first data row (if multiple benchmarks in file, take first)
    data = data_rows[0]

    for group, field in DESIRED_METRICS:
        # Find matching columns
        for i, (g, f, s) in enumerate(columns):
            if g != group:
                continue

            # For simple fields (Duration, Successful, etc.)
            if field in ("Successful", "Total", "Duration"):
                if f == field:
                    val = data[i] if i < len(data) else ""
                    key = f"{group} / {field}"
                    if key not in results:
                        results[key] = val
                continue

            # For distribution metrics, match field and filter stats
            # field is the unit part (e.g. "ms", "Requests/Sec", "Sec")
            if f.startswith("Successful "):
                unit_part = f.replace("Successful ", "")
            else:
                unit_part = f

            if unit_part != field:
                continue

            if s in STAT_TYPES:
                key = f"{group} / {field} / {s}"
                val = data[i] if i < len(data) else ""
                if key not in results:
                    results[key] = val

    return results


def collect_all_csvs(results_dir: Path) -> list[dict[str, str | float]]:
    """Parse all CSV files in results_dir and return list of metric dicts."""
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        print(f"[error] No CSV files found in {results_dir}", file=sys.stderr)
        return []

    all_results = []
    for csv_file in csv_files:
        print(f"[parse] {csv_file.name}")
        metrics = parse_guidellm_csv(csv_file)
        if metrics:
            all_results.append(metrics)

    return all_results


def export_merged_csv(all_results: list[dict[str, str | float]], output_path: Path) -> None:
    """Write all results into a single merged CSV."""
    if not all_results:
        print("[warn] No results to export.", file=sys.stderr)
        return

    # Collect all unique column names preserving order
    all_columns: list[str] = []
    seen: set[str] = set()
    for row in all_results:
        for key in row:
            if key not in seen:
                all_columns.append(key)
                seen.add(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"[export] Written {len(all_results)} rows -> {output_path}")


def copy_full_csv(results_dir: Path, output_path: Path) -> None:
    """Merge all raw CSV files (with multi-row headers) into one full CSV."""
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[list[str]] = []
    header_rows: list[list[str]] = []

    for i, csv_file in enumerate(csv_files):
        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 4:
            continue

        if i == 0:
            # Use headers from first file, prepend "Source" column
            header_rows = [["Source"] + rows[0], [""] + rows[1], [""] + rows[2]]

        # Append data rows with source filename
        for data_row in rows[3:]:
            all_rows.append([csv_file.stem] + data_row)

    if not all_rows:
        return

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for hr in header_rows:
            writer.writerow(hr)
        for row in all_rows:
            writer.writerow(row)

    print(f"[export] Full CSV ({len(all_rows)} rows) -> {output_path}")


def merge_json_results(results_dir: Path, output_path: Path) -> None:
    """Merge all JSON result files into one combined JSON report."""
    import json

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load first file as base, then merge benchmarks from others
    base_report = None
    all_benchmarks = []

    for jf in json_files:
        try:
            with jf.open("r") as f:
                data = json.load(f)
            benchmarks = data.get("benchmarks", [])
            all_benchmarks.extend(benchmarks)
            if base_report is None:
                base_report = data
        except Exception as exc:
            print(f"[warn] Failed to load {jf.name}: {exc}", file=sys.stderr)
            continue

    if base_report is None or not all_benchmarks:
        return

    base_report["benchmarks"] = all_benchmarks

    with output_path.open("w") as f:
        json.dump(base_report, f)

    print(f"[export] Full JSON ({len(all_benchmarks)} benchmarks) -> {output_path}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Export guidellm benchmark CSVs to one summary CSV")
    parser.add_argument("--results-dir", type=Path, default=script_dir / "results",
                        help="Directory containing guidellm CSV results (default: auto/results)")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "export",
                        help="Output directory for merged CSV (default: auto/export)")
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Summary export (extracted key metrics)
    all_results = collect_all_csvs(results_dir)
    if all_results:
        summary_path = output_dir / f"export_{timestamp}.csv"
        export_merged_csv(all_results, summary_path)

    # 2. Full export (all columns from raw CSVs merged)
    full_path = output_dir / f"export_full_{timestamp}.csv"
    copy_full_csv(results_dir, full_path)

    # 3. Full JSON export (merge all JSON results into one report)
    full_json_path = output_dir / f"export_full_{timestamp}.json"
    merge_json_results(results_dir, full_json_path)

    if not all_results:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
