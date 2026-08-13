#!/usr/bin/env python3
"""Export benchmark results from guidellm CSV outputs into a single summary CSV.

Usage:
    python3 export.py [--results-dir ./results] [--output-dir ./export]

Extracts key metrics from all CSV files in results dir and merges into one CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
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
    # Audio metrics
    ("Audio Seconds", "Input"),
    ("Audio Samples", "Input"),
    ("Audio Bytes", "Input"),
]

# For distribution metrics, we want these stats
STAT_TYPES = ["Mean", "Median", "Std Dev"]

# Status prefixes to check (guidellm prefixes distribution columns with status)
STATUS_PREFIXES = ["Successful ", "Errored ", ""]


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
            for prefix in STATUS_PREFIXES:
                if f == f"{prefix}{field}" or (prefix == "" and f == field):
                    if s in STAT_TYPES:
                        key = f"{group} / {field} / {s}"
                        val = data[i] if i < len(data) else ""
                        if key not in results and val:
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


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load key=value pairs from a .env file."""
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    with env_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                # Strip surrounding quotes
                value = value.strip().strip("'\"")
                env_vars[key.strip()] = value
    return env_vars


def sync_to_notion(all_results: list[dict[str, str | float]], page_title: str, script_dir: Path) -> int:
    """Create a Notion page with the summary CSV data as a table.

    Uses the export CSV filename as the page title to avoid duplicates.
    Reads NOTION_API and NOTION_DATABASE_ID from .env file in project root.
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    # Load .env from auto/ directory (same dir as this script)
    env_file = script_dir / ".env"
    env_vars = load_env_file(env_file)

    # Also check os.environ as fallback
    notion_api = env_vars.get("NOTION_API") or os.environ.get("NOTION_API")
    database_id = env_vars.get("NOTION_DATABASE_ID") or os.environ.get("NOTION_DATABASE_ID")
    page_id = env_vars.get("NOTION_PAGE_ID") or os.environ.get("NOTION_PAGE_ID")

    if not notion_api:
        print("[error] NOTION_API not found in .env or environment", file=sys.stderr)
        return 1
    if not database_id and not page_id:
        print("[error] NOTION_DATABASE_ID or NOTION_PAGE_ID not found in .env or environment", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {notion_api}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    # Check if page with this title already exists
    if database_id:
        # Query database for duplicate
        filter_payload = json.dumps({
            "filter": {
                "property": "title",
                "title": {"equals": page_title}
            }
        }).encode("utf-8")
        query_req = Request(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            data=filter_payload, headers=headers, method="POST"
        )
        try:
            with urlopen(query_req, timeout=30) as resp:
                result = json.loads(resp.read())
                if result.get("results"):
                    print(f"[sync] Page '{page_title}' already exists in Notion, skipping.")
                    return 0
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"[warn] Could not check for duplicates: {exc}", file=sys.stderr)
    elif page_id:
        # List children of the parent page to check for duplicate title
        list_url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        try:
            list_req = Request(list_url, headers=headers, method="GET")
            with urlopen(list_req, timeout=30) as resp:
                result = json.loads(resp.read())
                for block in result.get("results", []):
                    if block.get("type") == "child_page":
                        existing_title = block.get("child_page", {}).get("title", "")
                        if existing_title == page_title:
                            print(f"[sync] Page '{page_title}' already exists in Notion, skipping.")
                            return 0
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"[warn] Could not check for duplicates: {exc}", file=sys.stderr)

    # Build Notion page with table
    title = page_title

    # Build table rows as Notion blocks
    # Header row from all_results keys
    if not all_results:
        return 0

    columns = list(all_results[0].keys())
    # Notion tables have max 100 columns; truncate if needed
    max_cols = min(len(columns), 100)
    columns = columns[:max_cols]

    table_children: list[dict] = []

    # Header row
    header_cells = []
    for col in columns:
        header_cells.append([{"type": "text", "text": {"content": str(col)[:100]}}])
    table_children.append({
        "type": "table_row",
        "table_row": {"cells": header_cells}
    })

    # Data rows
    for row in all_results:
        cells = []
        for col in columns:
            val = str(row.get(col, ""))[:100]
            cells.append([{"type": "text", "text": {"content": val}}])
        table_children.append({
            "type": "table_row",
            "table_row": {"cells": cells}
        })

    # Create page payload
    if database_id:
        parent = {"database_id": database_id}
    else:
        parent = {"page_id": page_id}

    payload = {
        "parent": parent,
        "properties": {
            "title": {"title": [{"text": {"content": title}}]}
        },
        "children": [
            {
                "type": "table",
                "table": {
                    "table_width": max_cols,
                    "has_column_header": True,
                    "has_row_header": False,
                    "children": table_children
                }
            }
        ]
    }

    # Send to Notion API
    headers = {
        "Authorization": f"Bearer {notion_api}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request("https://api.notion.com/v1/pages", data=data, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=30) as resp:
            if 200 <= resp.status < 300:
                result = json.loads(resp.read())
                page_url = result.get("url", "")
                print(f"[sync] Notion page created: {page_url}")
                return 0
            else:
                print(f"[error] Notion API returned status {resp.status}", file=sys.stderr)
                return 1
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[error] Notion API error {exc.code}: {body}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print(f"[error] Notion API request failed: {exc}", file=sys.stderr)
        return 1


def generate_report(all_results: list[dict[str, str | float]], report_title: str, script_dir: Path) -> int:
    """Generate a benchmark analysis report using OpenAI API and upload to Notion.

    Reads OPENAI_API_KEY, OPENAI_API_URL, NOTION_API, and NOTION_PAGE_ID from .env.
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    env_file = script_dir / ".env"
    env_vars = load_env_file(env_file)

    openai_api_key = env_vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    openai_api_url = env_vars.get("OPENAI_API_URL") or os.environ.get("OPENAI_API_URL")
    notion_api = env_vars.get("NOTION_API") or os.environ.get("NOTION_API")
    database_id = env_vars.get("NOTION_DATABASE_ID") or os.environ.get("NOTION_DATABASE_ID")
    page_id = env_vars.get("NOTION_PAGE_ID") or os.environ.get("NOTION_PAGE_ID")

    if not openai_api_key:
        print("[error] OPENAI_API_KEY not found in .env or environment", file=sys.stderr)
        return 1
    if not openai_api_url:
        print("[error] OPENAI_API_URL not found in .env or environment", file=sys.stderr)
        return 1
    if not notion_api:
        print("[error] NOTION_API not found in .env or environment", file=sys.stderr)
        return 1
    if not database_id and not page_id:
        print("[error] NOTION_DATABASE_ID or NOTION_PAGE_ID not found in .env or environment", file=sys.stderr)
        return 1

    # Build CSV content for the prompt
    csv_lines = []
    if all_results:
        columns = list(all_results[0].keys())
        csv_lines.append(",".join(columns))
        for row in all_results:
            csv_lines.append(",".join(str(row.get(c, "")) for c in columns))
    csv_content = "\n".join(csv_lines)

    # Build the prompt
    prompt = f"""You are a benchmark performance analyst. Analyze the following LLM inference benchmark data and write a comprehensive report in Markdown format (in Traditional Chinese, 繁體中文).

The report MUST include these chapters in this exact order:
1. 一、執行摘要 - Key findings, peak throughput, recommended concurrency
2. 二、研究問題與實驗設計 - Platform, models, configurations tested, workload description
3. 三、指標定義與計算方式 - Define RPS, TPS, TTFT, TPOT, ITL, E2E latency
4. 四、資料完整性與公平性檢查 - Data validation, request counts, input/output lengths
5. 五、完整結果 - Per-model/configuration results with tables (peak points, comparisons)
6. 六、原因分析 - Why certain configs perform better (memory bandwidth, batch size, quantization effects)
7. 七、部署建議 - Recommended configurations for different use cases (interactive, throughput, balanced)
8. 八、結論 - Summary and next steps

For each section, include relevant data tables from the benchmark results.
Use | table | format | for data tables.

Here is the benchmark data (CSV format):
```csv
{csv_content}
```

Write the full report now in Markdown:"""

    # Call OpenAI API
    print("[report] Checking OpenAI API connectivity...")
    openai_base = openai_api_url.rstrip("/")

    openai_headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    # Health check: list models to verify API key and URL
    health_req = Request(
        f"{openai_base}/models",
        headers=openai_headers,
        method="GET",
    )
    try:
        with urlopen(health_req, timeout=15) as resp:
            if resp.status >= 400:
                print(f"[error] OpenAI API health check failed: status {resp.status}", file=sys.stderr)
                return 1
        print("[report] OpenAI API connection OK")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[error] OpenAI API health check failed ({exc.code}): {body}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print(f"[error] OpenAI API unreachable: {exc}", file=sys.stderr)
        return 1

    print("[report] Generating report via OpenAI API (this may take up to 30 minutes)...")
    openai_url = openai_base + "/chat/completions"
    openai_payload = json.dumps({
        "model": env_vars.get("OPENAI_API_MODEL") or os.environ.get("OPENAI_API_MODEL") or "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a professional benchmark performance analyst writing detailed technical reports."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 16000,
    }).encode("utf-8")

    req = Request(openai_url, data=openai_payload, headers=openai_headers, method="POST")
    try:
        with urlopen(req, timeout=1800) as resp:
            result = json.loads(resp.read())
            report_md = result["choices"][0]["message"]["content"]
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[error] OpenAI API error {exc.code}: {body}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print(f"[error] OpenAI API request failed: {exc}", file=sys.stderr)
        return 1

    print(f"[report] Report generated ({len(report_md)} chars)")

    # Save report locally
    export_dir = script_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / f"{report_title}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[report] Saved locally: {report_path}")

    # Upload to Notion
    notion_headers = {
        "Authorization": f"Bearer {notion_api}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    # Check duplicate
    if page_id:
        list_url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        try:
            list_req = Request(list_url, headers=notion_headers, method="GET")
            with urlopen(list_req, timeout=30) as resp:
                result = json.loads(resp.read())
                for block in result.get("results", []):
                    if block.get("type") == "child_page":
                        existing_title = block.get("child_page", {}).get("title", "")
                        if existing_title == report_title:
                            print(f"[report] Page '{report_title}' already exists in Notion, skipping.")
                            return 0
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"[warn] Could not check for duplicates: {exc}", file=sys.stderr)

    # Convert markdown to Notion blocks (simplified: paragraph blocks)
    children = _markdown_to_notion_blocks(report_md)

    if database_id:
        parent = {"database_id": database_id}
    else:
        parent = {"page_id": page_id}

    # Notion API limits children to 100 blocks per request
    # Create page with first batch, then append remaining
    first_batch = children[:100]
    remaining = children[100:]

    payload = json.dumps({
        "parent": parent,
        "properties": {
            "title": {"title": [{"text": {"content": report_title}}]}
        },
        "children": first_batch,
    }).encode("utf-8")

    req = Request("https://api.notion.com/v1/pages", data=payload, headers=notion_headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            page_url = result.get("url", "")
            new_page_id = result.get("id", "")
            print(f"[report] Notion page created: {page_url}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[error] Notion API error {exc.code}: {body}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print(f"[error] Notion API request failed: {exc}", file=sys.stderr)
        return 1

    # Append remaining blocks in batches of 100
    while remaining and new_page_id:
        batch = remaining[:100]
        remaining = remaining[100:]
        append_payload = json.dumps({"children": batch}).encode("utf-8")
        append_url = f"https://api.notion.com/v1/blocks/{new_page_id}/children"
        append_req = Request(append_url, data=append_payload, headers=notion_headers, method="PATCH")
        try:
            with urlopen(append_req, timeout=60) as resp:
                pass
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"[warn] Failed to append blocks: {exc}", file=sys.stderr)
            break

    return 0


def _markdown_to_notion_blocks(md: str) -> list[dict]:
    """Convert markdown text to Notion block objects (headings, paragraphs, tables)."""
    blocks: list[dict] = []
    lines = md.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Headings
        if line.startswith("# "):
            blocks.append({
                "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()[:2000]}}]}
            })
            i += 1
        elif line.startswith("## "):
            blocks.append({
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:].strip()[:2000]}}]}
            })
            i += 1
        elif line.startswith("### "):
            blocks.append({
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:].strip()[:2000]}}]}
            })
            i += 1
        # Code blocks
        elif line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_content = "\n".join(code_lines)[:2000]
            blocks.append({
                "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": code_content}}],
                    "language": "plain text"
                }
            })
        # Table (pipe-delimited)
        elif "|" in line and line.strip().startswith("|"):
            table_rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                # Skip separator rows (|---|---|)
                if re.match(r"^\|[\s\-:|]+\|$", row_line):
                    i += 1
                    continue
                cells = [c.strip() for c in row_line.split("|")[1:-1]]
                table_rows.append(cells)
                i += 1
            if table_rows:
                max_cols = max(len(r) for r in table_rows)
                # Pad rows to same width
                for r in table_rows:
                    while len(r) < max_cols:
                        r.append("")
                # Notion table max 100 columns
                max_cols = min(max_cols, 100)
                notion_rows = []
                for r in table_rows:
                    cells = [[{"type": "text", "text": {"content": c[:100]}}] for c in r[:max_cols]]
                    notion_rows.append({"type": "table_row", "table_row": {"cells": cells}})
                blocks.append({
                    "type": "table",
                    "table": {
                        "table_width": max_cols,
                        "has_column_header": True,
                        "has_row_header": False,
                        "children": notion_rows
                    }
                })
        # Bulleted list
        elif line.strip().startswith("- "):
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line.strip()[2:][:2000]}}]}
            })
            i += 1
        # Blockquote
        elif line.strip().startswith("> "):
            blocks.append({
                "type": "quote",
                "quote": {"rich_text": [{"type": "text", "text": {"content": line.strip()[2:][:2000]}}]}
            })
            i += 1
        # Empty line
        elif line.strip() == "":
            i += 1
        # Regular paragraph
        else:
            content = line.strip()[:2000]
            if content:
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]}
                })
            i += 1

    return blocks


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
    parser.add_argument("--sync", action="store_true",
                        help="Sync summary CSV to Notion. Requires NOTION_API and NOTION_DATABASE_ID in .env")
    parser.add_argument("--report", action="store_true",
                        help="Generate analysis report via OpenAI and upload to Notion. "
                             "Requires OPENAI_API_KEY and OPENAI_API_URL in .env")
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d")

    # Hash source filenames for deterministic dedup
    import hashlib
    csv_files = sorted(results_dir.glob("*.csv"))
    source_names = "_".join(f.stem for f in csv_files)
    content_hash = hashlib.md5(source_names.encode()).hexdigest()[:8]
    export_base = f"export_{content_hash}_{timestamp}"

    # 1. Summary export (extracted key metrics)
    all_results = collect_all_csvs(results_dir)
    if all_results:
        summary_path = output_dir / f"{export_base}.csv"
        export_merged_csv(all_results, summary_path)

    # 2. Full export (all columns from raw CSVs merged)
    full_path = output_dir / f"{export_base}_full.csv"
    copy_full_csv(results_dir, full_path)

    # 3. Full JSON export (merge all JSON results into one report)
    full_json_path = output_dir / f"{export_base}_full.json"
    merge_json_results(results_dir, full_json_path)

    # 4. Sync to Notion (if --sync)
    if args.sync and all_results:
        page_title = f"{export_base}.csv"
        rc = sync_to_notion(all_results, page_title=page_title, script_dir=script_dir)
        if rc != 0:
            return rc

    # 5. Generate report (if --report)
    if args.report and all_results:
        report_title = f"report_{export_base}"
        rc = generate_report(all_results, report_title=report_title, script_dir=script_dir)
        if rc != 0:
            return rc

    if not all_results:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
