#!/usr/bin/env python3
"""GuideLLM autotest workflow.

Loops over service configs (docker-compose files) and test configs (shell scripts),
running each test at each concurrency level.

Usage:
    python3 workflow.py \
        --service-dir ./example-service-dir \
        --test-dir ./example-test-dir \
        --concurrency 1,16,32,64,128,256,512
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOG = logging.getLogger("workflow")

DEFAULT_HEALTH_TIMEOUT_SEC = 36 * 60
DEFAULT_PORT_FREE_WAIT_SEC = 90


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path, check: bool = True, capture: bool = False):
    return subprocess.run(cmd, cwd=str(cwd), check=check, text=True,
                          capture_output=capture)


def wait_for_port_free(port: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return
            except OSError:
                pass
        time.sleep(1)
    raise TimeoutError(f"Port {port} still busy after {timeout_sec}s")


def wait_for_health(url: str, timeout_sec: int, interval_sec: int = 6) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    LOG.info("[ok] health check passed: %s", url)
                    return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            remaining = max(0, int(deadline - time.time()))
            LOG.info("[wait] %s (%s) remaining=%ss", url, exc, remaining)
        time.sleep(interval_sec)
    raise TimeoutError(f"Health check timed out: {url}")


# ---------------------------------------------------------------------------
# Compose file helpers
# ---------------------------------------------------------------------------

def extract_port_from_compose(compose_file: Path) -> int:
    """Extract host port from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    # Match literal port: - "8976:8976" or - 8976:8976
    match = re.search(r'(?m)^\s*-\s*"?(\d+):\d+(?:/\w+)?"?\s*$', content)
    if match:
        return int(match.group(1))
    # Match env var with default: - "${HOST_PORT:-8976}:8976"
    match = re.search(r'(?m)^\s*-\s*"?\$\{[^:}]+:-(\d+)\}:\d+(?:/\w+)?"?\s*$', content)
    if match:
        return int(match.group(1))
    # Match --port flag in command
    match = re.search(r'--port\s+(\d+)', content)
    if match:
        return int(match.group(1))
    raise RuntimeError(f"Cannot extract port from {compose_file}")


def extract_model_from_compose(compose_file: Path) -> str:
    """Extract model path from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    # Try --model flag
    m = re.search(r"--model(?:-path)?(?:=|\s+)([^\s\"']+)", content)
    if m:
        return _resolve_env_default(m.group(1))
    # Try "<engine> serve <model>" (e.g. "vllm serve ...", "tokenspeed serve ...")
    m = re.search(r"\w+\s+serve\s+([^\s\"'\\]+)", content)
    if m:
        return _resolve_env_default(m.group(1))
    raise RuntimeError(f"Cannot extract model from {compose_file}")


def extract_served_model_name(compose_file: Path) -> str | None:
    content = compose_file.read_text(encoding="utf-8")
    m = re.search(r"--served-model-name(?:=|\s+)([^\s\"']+)", content)
    return _resolve_env_default(m.group(1)) if m else None


def extract_tp_from_compose(compose_file: Path) -> str:
    """Extract tensor-parallel-size from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    m = re.search(r"--tensor-parallel-size(?:=|\s+)([^\s\"']+)", content)
    if m:
        return _resolve_env_default(m.group(1))
    return "1"


def _resolve_env_default(value: str) -> str:
    """Resolve ${VAR:-default} to just 'default'. Pass through literals."""
    m = re.match(r'^\$\{[^:}]+:-(.+)\}$', value)
    if m:
        return m.group(1)
    return value


def compose_up(compose_file: Path) -> None:
    cwd = compose_file.parent
    name = compose_file.name
    cmd = ["docker", "compose", "-f", name, "up", "-d"]
    LOG.info("[compose-up] %s", " ".join(cmd))
    run_cmd(cmd, cwd=cwd)


def compose_down(compose_file: Path) -> None:
    cwd = compose_file.parent
    name = compose_file.name
    cmd = ["docker", "compose", "-f", name, "down"]
    LOG.info("[compose-down] %s", " ".join(cmd))
    run_cmd(cmd, cwd=cwd, check=False)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    test_script: Path,
    port: int,
    model: str,
    concurrency: int | None,
    output_dir: Path,
    service_name: str,
    tp: str,
    served_model: str | None = None,
    output_prefix: str = "",
) -> None:
    """Run a test script with the workflow settings injected as environment variables."""
    test_name = test_script.stem  # e.g. "rag_bench"
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL"] = model
    env["SERVED_MODEL"] = served_model or service_name
    if concurrency is not None:
        env["CONCURRENCY"] = str(concurrency)
        env["COUNT"] = str(concurrency * 10)
        env["RUN_TAG"] = f"c{concurrency}"
    else:
        env["RUN_TAG"] = ""
    env["OUTPUT_DIR"] = str(output_dir)
    env["OUTPUT_PREFIX"] = output_prefix
    env["SERVICE_NAME"] = service_name
    env["TEST_NAME"] = test_name
    env["TP"] = tp

    # Test scripts may live outside the project directory. Make GuideLLM
    # available through the child shell's PATH without exposing its location
    # to individual test scripts.
    guidellm_path = shutil.which("guidellm", path=env.get("PATH"))
    if not guidellm_path:
        candidates = (
            Path(sys.executable).with_name("guidellm"),
            Path(__file__).resolve().parents[1] / ".venv" / "bin" / "guidellm",
        )
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                guidellm_path = str(candidate)
                break
    if guidellm_path:
        guidellm_dir = str(Path(guidellm_path).resolve().parent)
        env["PATH"] = os.pathsep.join(
            [guidellm_dir, env.get("PATH", "")]
        ).rstrip(os.pathsep)

    LOG.info(
        "[test] %s PORT=%s MODEL=%s CONCURRENCY=%s TP=%s OUTPUT_DIR=%s",
        test_script.name,
        port,
        model,
        concurrency or "(test decides)",
        tp,
        output_dir,
    )

    # Run test script with output forwarded to screen (stdout/stderr passthrough)
    result = subprocess.run(
        ["bash", "--norc", "--noprofile", str(test_script)],
        cwd=str(test_script.parent),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Test {test_script.name} failed with exit code {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def iterate_files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(sorted(directory.glob(f"*{ext}")))
    return files


def result_exists(output_dir: Path, service_name: str, tp: str, test_name: str, concurrency: int | None, output_prefix: str = "") -> bool:
    """Check if results already exist for this combination (json+csv+png)."""
    prefix = f"{output_prefix}." if output_prefix else ""
    if concurrency is None:
        base = f"{prefix}{service_name}.tp{tp}.{test_name}"
    else:
        base = f"{prefix}{service_name}.tp{tp}.{test_name}.c{concurrency}"
    return (
        (output_dir / f"{base}.json").exists()
        and (output_dir / f"{base}.csv").exists()
        and (output_dir / f"{base}.png").exists()
    )


def parse_concurrency(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",")]
    if not values:
        raise ValueError("At least one concurrency value required")
    return values


def parse_config_pairs(config_str: str) -> list[tuple[Path, Path]]:
    """Parse config string into (service_dir, test_dir) pairs.

    Format:
        service_dir1,test_dir1
        service_dir2,test_dir2
        ...

    Or single-line: service_dir1,test_dir1;service_dir2,test_dir2
    """
    pairs: list[tuple[Path, Path]] = []
    # Support multiline or semicolon-separated
    lines = config_str.replace(";", "\n").strip().split("\n")
    for i, line in enumerate(lines, 1):
        line = line.strip().strip("()")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            LOG.warning("[config] Skipping invalid pair (line %d): '%s' — must be 'service_dir,test_dir'", i, line)
            continue
        svc_dir = Path(parts[0])
        tst_dir = Path(parts[1])
        if not svc_dir.exists():
            LOG.warning("[config] Skipping pair (line %d): service-dir not found: %s", i, svc_dir)
            continue
        if not tst_dir.exists():
            LOG.warning("[config] Skipping pair (line %d): test-dir not found: %s", i, tst_dir)
            continue
        pairs.append((svc_dir.resolve(), tst_dir.resolve()))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GuideLLM autotest workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config format (--config):
  Each line is a pair: service_dir,test_dir
  Example:
    --config "
      /path/to/services1,/path/to/tests1
      /path/to/services2,/path/to/tests2
    "
""",
    )
    parser.add_argument("--config", default=None,
                        help="Pairs of (service_dir,test_dir), one per line or semicolon-separated. "
                             "Each pair runs service configs against test scripts.")
    parser.add_argument("--service-dir", type=Path, default=None,
                        help="(Legacy) Single service dir. Use --config for multiple pairs.")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="(Legacy) Single test dir. Use --config for multiple pairs.")
    parser.add_argument("--target", default=None,
                        help="Test a running service directly (no container management). "
                             "Value is the endpoint URL, e.g. http://127.0.0.1:8976. "
                             "Requires --model, --service-name, --tp, and --test-dir.")
    parser.add_argument("--model", default=None,
                        help="Model/tokenizer path (required with --target)")
    parser.add_argument("--tp", default=None,
                        help="Tensor parallel size (required with --target)")
    parser.add_argument("--service-name", default=None,
                        help="Service name for output filenames (required with --target)")
    parser.add_argument("--concurrency", default=None,
                        help="Comma-separated concurrency values. If not set, loop once with concurrency decided by test script.")
    parser.add_argument("--nostop", action="store_true",
                        help="Keep containers alive after tests (skip compose down)")
    parser.add_argument("--output", type=Path, default=Path("./results"),
                        help="Output directory for benchmark results (default: ./results)")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_SEC,
                        help="Health check timeout in seconds (default: 2160)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_dir: Path = args.output.resolve()
    concurrency_values = parse_concurrency(args.concurrency) if args.concurrency else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # --target mode: test a running service directly, no containers
    if args.target:
        if not args.test_dir:
            LOG.error("--test-dir is required with --target")
            return 1
        test_dir = args.test_dir.resolve()
        if not test_dir.exists():
            LOG.error("test-dir not found: %s", test_dir)
            return 1
        test_files = iterate_files(test_dir, (".sh",))
        if not test_files:
            LOG.error("No *.sh files in test-dir: %s", test_dir)
            return 1

        missing = []
        if not args.model:
            missing.append("--model")
        if not args.service_name:
            missing.append("--service-name")
        if not args.tp:
            missing.append("--tp")
        if missing:
            LOG.error("--target mode requires: --model, --service-name, --tp")
            LOG.error("Missing: %s", ", ".join(missing))
            return 1

        from urllib.parse import urlsplit
        parsed = urlsplit(args.target.rstrip("/"))
        port = parsed.port or 8976
        model = args.model
        tp = args.tp
        service_name = args.service_name

        LOG.info("[config] mode: --target (no container management)")
        LOG.info("[config] target: %s", args.target)
        LOG.info("[config] model: %s", model)
        LOG.info("[config] tp: %s", tp)
        LOG.info("[config] service-name: %s", service_name)
        LOG.info("[config] test-dir: %s (%d tests)", test_dir, len(test_files))
        LOG.info("[config] output: %s", output_dir)
        LOG.info("[config] concurrency: %s", concurrency_values)

        # Wait for health
        health_url = f"{args.target.rstrip('/')}/health"
        try:
            wait_for_health(health_url, timeout_sec=args.health_timeout)
        except Exception as exc:
            LOG.error("[error] health check failed: %s", exc)
            return 1

        failures: list[str] = []
        output_prefix = test_dir.name
        for test_file in test_files:
            LOG.info("-" * 40)
            LOG.info("[test] %s", test_file.name)

            for conc in (concurrency_values or [None]):
                if result_exists(output_dir, service_name, tp, test_file.stem, conc, output_prefix):
                    LOG.info("[skip] %s c=%s — results already exist",
                             test_file.name, conc)
                    continue
                try:
                    run_test(
                        test_script=test_file,
                        port=port,
                        model=model,
                        concurrency=conc,
                        output_dir=output_dir,
                        service_name=service_name,
                        tp=tp,
                        output_prefix=output_prefix,
                    )
                except Exception as exc:
                    msg = f"{test_file.name} c={conc}: {exc}"
                    failures.append(msg)
                    LOG.exception("[error] %s", msg)

        LOG.info("=" * 60)
        if failures:
            LOG.error("[summary] %d failure(s):", len(failures))
            for f in failures:
                LOG.error("  - %s", f)
            return 1
        LOG.info("[summary] All tests passed.")
        return 0

    # Build config pairs
    config_pairs: list[tuple[Path, Path]] = []
    if args.config:
        config_pairs = parse_config_pairs(args.config)
    elif args.service_dir and args.test_dir:
        svc = args.service_dir.resolve()
        tst = args.test_dir.resolve()
        if not svc.exists():
            LOG.error("service-dir not found: %s", svc)
            return 1
        if not tst.exists():
            LOG.error("test-dir not found: %s", tst)
            return 1
        config_pairs = [(svc, tst)]
    else:
        LOG.error("Provide --config or both --service-dir and --test-dir (or use --target)")
        return 1

    if not config_pairs:
        LOG.error("No valid (service_dir, test_dir) pairs found.")
        return 1

    LOG.info("[config] %d pair(s) to run", len(config_pairs))
    LOG.info("[config] output: %s", output_dir)
    LOG.info("[config] concurrency: %s", concurrency_values or "(test decides)")
    LOG.info("[config] nostop: %s", args.nostop)

    failures: list[str] = []

    for pair_idx, (service_dir, test_dir) in enumerate(config_pairs, 1):
        # Use dir names as prefix to avoid collisions between pairs
        output_prefix = f"{service_dir.name}.{test_dir.name}"

        LOG.info("=" * 60)
        LOG.info("[pair %d/%d] service-dir: %s", pair_idx, len(config_pairs), service_dir)
        LOG.info("[pair %d/%d] test-dir: %s", pair_idx, len(config_pairs), test_dir)
        LOG.info("[pair %d/%d] output-prefix: %s", pair_idx, len(config_pairs), output_prefix)

        service_files = iterate_files(service_dir, (".yml", ".yaml"))
        test_files = iterate_files(test_dir, (".sh",))

        if not service_files:
            LOG.warning("[pair %d] No *.yml/*.yaml in service-dir: %s — skipping", pair_idx, service_dir)
            continue
        if not test_files:
            LOG.warning("[pair %d] No *.sh in test-dir: %s — skipping", pair_idx, test_dir)
            continue

        if args.nostop:
            for svc_file in service_files:
                LOG.info("=" * 60)
                LOG.info("[service] %s", svc_file.name)

                try:
                    port = extract_port_from_compose(svc_file)
                    model = extract_model_from_compose(svc_file)
                    tp = extract_tp_from_compose(svc_file)
                    served_model = extract_served_model_name(svc_file)
                except Exception as exc:
                    failures.append(f"{svc_file.name}: parse error: {exc}")
                    LOG.exception("[error] %s", exc)
                    continue

                for test_file in test_files:
                    LOG.info("-" * 40)
                    LOG.info("[test] %s on %s", test_file.name, svc_file.name)

                    try:
                        wait_for_port_free(port, DEFAULT_PORT_FREE_WAIT_SEC)
                        compose_up(svc_file)
                    except Exception as exc:
                        failures.append(f"{svc_file.name}/{test_file.name}: startup error: {exc}")
                        LOG.exception("[error] %s", exc)
                        continue

                    try:
                        wait_for_health(
                            f"http://127.0.0.1:{port}/health",
                            timeout_sec=args.health_timeout,
                        )
                    except Exception as exc:
                        failures.append(f"{svc_file.name}/{test_file.name}: health timeout: {exc}")
                        LOG.exception("[error] %s", exc)
                        compose_down(svc_file)
                        continue

                    for conc in (concurrency_values or [None]):
                        if result_exists(output_dir, svc_file.stem, tp, test_file.stem, conc, output_prefix):
                            LOG.info("[skip] %s/%s c=%s — results already exist",
                                     svc_file.name, test_file.name, conc)
                            continue
                        try:
                            run_test(
                                test_script=test_file,
                                port=port,
                                model=model,
                                concurrency=conc,
                                output_dir=output_dir,
                                service_name=svc_file.stem,
                                tp=tp,
                                served_model=served_model,
                                output_prefix=output_prefix,
                            )
                        except Exception as exc:
                            msg = f"{svc_file.name}/{test_file.name} c={conc}: {exc}"
                            failures.append(msg)
                            LOG.exception("[error] %s", msg)

                    compose_down(svc_file)
        else:
            for svc_file in service_files:
                LOG.info("=" * 60)
                LOG.info("[service] %s", svc_file.name)

                try:
                    port = extract_port_from_compose(svc_file)
                    model = extract_model_from_compose(svc_file)
                    tp = extract_tp_from_compose(svc_file)
                    served_model = extract_served_model_name(svc_file)
                except Exception as exc:
                    failures.append(f"{svc_file.name}: parse error: {exc}")
                    LOG.exception("[error] %s", exc)
                    continue

                for test_file in test_files:
                    LOG.info("-" * 40)
                    LOG.info("[test] %s on %s", test_file.name, svc_file.name)

                    for conc in (concurrency_values or [None]):
                        if result_exists(output_dir, svc_file.stem, tp, test_file.stem, conc, output_prefix):
                            LOG.info("[skip] %s/%s c=%s — results already exist",
                                     svc_file.name, test_file.name, conc)
                            continue
                        try:
                            wait_for_port_free(port, DEFAULT_PORT_FREE_WAIT_SEC)
                            compose_up(svc_file)
                        except Exception as exc:
                            failures.append(f"{svc_file.name}/{test_file.name} c={conc}: startup error: {exc}")
                            LOG.exception("[error] %s", exc)
                            continue

                        try:
                            wait_for_health(
                                f"http://127.0.0.1:{port}/health",
                                timeout_sec=args.health_timeout,
                            )
                        except Exception as exc:
                            failures.append(f"{svc_file.name}/{test_file.name} c={conc}: health timeout: {exc}")
                            LOG.exception("[error] %s", exc)
                            compose_down(svc_file)
                            continue

                        try:
                            run_test(
                                test_script=test_file,
                                port=port,
                                model=model,
                                concurrency=conc,
                                output_dir=output_dir,
                                service_name=svc_file.stem,
                                tp=tp,
                                served_model=served_model,
                                output_prefix=output_prefix,
                            )
                        except Exception as exc:
                            msg = f"{svc_file.name}/{test_file.name} c={conc}: {exc}"
                            failures.append(msg)
                            LOG.exception("[error] %s", msg)

                        compose_down(svc_file)

    # Summary
    LOG.info("=" * 60)
    if failures:
        LOG.error("[summary] %d failure(s):", len(failures))
        for f in failures:
            LOG.error("  - %s", f)
        return 1

    LOG.info("[summary] All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
