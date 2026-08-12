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
    match = re.search(r'(?m)^\s*-\s*"?(\d+):\d+(?:/\w+)?"?\s*$', content)
    if not match:
        raise RuntimeError(f"Cannot extract port from {compose_file}")
    return int(match.group(1))


def extract_model_from_compose(compose_file: Path) -> str:
    """Extract model path from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    # Try --model flag
    m = re.search(r"--model(?:-path)?(?:=|\s+)([^\s\"']+)", content)
    if m:
        return m.group(1)
    # Try vllm serve <model>
    m = re.search(r"vllm\s+serve\s+([^\s\"'\\]+)", content)
    if m:
        return m.group(1)
    raise RuntimeError(f"Cannot extract model from {compose_file}")


def extract_served_model_name(compose_file: Path) -> str | None:
    content = compose_file.read_text(encoding="utf-8")
    m = re.search(r"--served-model-name(?:=|\s+)([^\s\"']+)", content)
    return m.group(1) if m else None


def extract_tp_from_compose(compose_file: Path) -> str:
    """Extract tensor-parallel-size from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    m = re.search(r"--tensor-parallel-size(?:=|\s+)(\d+)", content)
    return m.group(1) if m else "1"


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
    concurrency: int,
    output_dir: Path,
    service_name: str,
    tp: str,
) -> None:
    """Run a test script with the workflow settings injected as environment variables."""
    count = concurrency * 10
    test_name = test_script.stem  # e.g. "rag_bench"
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["MODEL"] = model
    env["CONCURRENCY"] = str(concurrency)
    env["COUNT"] = str(count)
    env["OUTPUT_DIR"] = str(output_dir)
    env["SERVICE_NAME"] = service_name
    env["TEST_NAME"] = test_name
    env["TP"] = tp
    env["MODEL"] = model
    env["CONCURRENCY"] = str(concurrency)
    env["COUNT"] = str(count)
    env["OUTPUT_DIR"] = str(output_dir)
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
        "[test] %s PORT=%s MODEL=%s CONCURRENCY=%s COUNT=%s TP=%s OUTPUT_DIR=%s",
        test_script.name,
        port,
        model,
        concurrency,
        count,
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


def result_exists(output_dir: Path, service_name: str, tp: str, test_name: str, concurrency: int) -> bool:
    """Check if results already exist for this combination (json+csv+png)."""
    base = f"{service_name}.tp{tp}.{test_name}.c{concurrency}"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="GuideLLM autotest workflow")
    parser.add_argument("--service-dir", type=Path, required=True,
                        help="Directory with docker-compose *.yml service configs")
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Directory with *.sh test configs")
    parser.add_argument("--concurrency", default="1,16,32,64,128,256,512",
                        help="Comma-separated concurrency values")
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

    service_dir: Path = args.service_dir.resolve()
    test_dir: Path = args.test_dir.resolve()
    output_dir: Path = args.output.resolve()
    concurrency_values = parse_concurrency(args.concurrency)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not service_dir.exists():
        LOG.error("service-dir not found: %s", service_dir)
        return 1
    if not test_dir.exists():
        LOG.error("test-dir not found: %s", test_dir)
        return 1

    service_files = iterate_files(service_dir, (".yml", ".yaml"))
    test_files = iterate_files(test_dir, (".sh",))

    if not service_files:
        LOG.error("No *.yml/*.yaml files in service-dir: %s", service_dir)
        return 1
    if not test_files:
        LOG.error("No *.sh files in test-dir: %s", test_dir)
        return 1

    LOG.info("[config] service-dir: %s (%d services)", service_dir, len(service_files))
    LOG.info("[config] test-dir: %s (%d tests)", test_dir, len(test_files))
    LOG.info("[config] output: %s", output_dir)
    LOG.info("[config] concurrency: %s", concurrency_values)
    LOG.info("[config] nostop: %s", args.nostop)

    failures: list[str] = []

    if args.nostop:
        # --nostop mode:
        #   1. loop services
        #   2. loop tests
        #      docker compose up
        #      3. loop concurrency -> run test
        #      docker compose down (after all concurrency done)
        for svc_file in service_files:
            LOG.info("=" * 60)
            LOG.info("[service] %s", svc_file.name)

            try:
                port = extract_port_from_compose(svc_file)
                model = extract_model_from_compose(svc_file)
                tp = extract_tp_from_compose(svc_file)
            except Exception as exc:
                failures.append(f"{svc_file.name}: parse error: {exc}")
                LOG.exception("[error] %s", exc)
                continue

            for test_file in test_files:
                LOG.info("-" * 40)
                LOG.info("[test] %s on %s", test_file.name, svc_file.name)

                # Start container
                try:
                    wait_for_port_free(port, DEFAULT_PORT_FREE_WAIT_SEC)
                    compose_up(svc_file)
                except Exception as exc:
                    failures.append(f"{svc_file.name}/{test_file.name}: startup error: {exc}")
                    LOG.exception("[error] %s", exc)
                    continue

                # Wait for health
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

                # Loop concurrency
                for conc in concurrency_values:
                    if result_exists(output_dir, svc_file.stem, tp, test_file.stem, conc):
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
                        )
                    except Exception as exc:
                        msg = f"{svc_file.name}/{test_file.name} c={conc}: {exc}"
                        failures.append(msg)
                        LOG.exception("[error] %s", msg)

                # Stop container after all concurrency levels
                compose_down(svc_file)
    else:
        # Default mode:
        #   1. loop services
        #   2. loop tests
        #   3. loop concurrency
        #      docker compose up -> run test -> docker compose down
        for svc_file in service_files:
            LOG.info("=" * 60)
            LOG.info("[service] %s", svc_file.name)

            try:
                port = extract_port_from_compose(svc_file)
                model = extract_model_from_compose(svc_file)
                tp = extract_tp_from_compose(svc_file)
            except Exception as exc:
                failures.append(f"{svc_file.name}: parse error: {exc}")
                LOG.exception("[error] %s", exc)
                continue

            for test_file in test_files:
                LOG.info("-" * 40)
                LOG.info("[test] %s on %s", test_file.name, svc_file.name)

                for conc in concurrency_values:
                    if result_exists(output_dir, svc_file.stem, tp, test_file.stem, conc):
                        LOG.info("[skip] %s/%s c=%s — results already exist",
                                 svc_file.name, test_file.name, conc)
                        continue
                    # Start container
                    try:
                        wait_for_port_free(port, DEFAULT_PORT_FREE_WAIT_SEC)
                        compose_up(svc_file)
                    except Exception as exc:
                        failures.append(f"{svc_file.name}/{test_file.name} c={conc}: startup error: {exc}")
                        LOG.exception("[error] %s", exc)
                        continue

                    # Wait for health
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

                    # Run test
                    try:
                        run_test(
                            test_script=test_file,
                            port=port,
                            model=model,
                            concurrency=conc,
                            output_dir=output_dir,
                            service_name=svc_file.stem,
                            tp=tp,
                        )
                    except Exception as exc:
                        msg = f"{svc_file.name}/{test_file.name} c={conc}: {exc}"
                        failures.append(msg)
                        LOG.exception("[error] %s", msg)

                    # Stop container
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
