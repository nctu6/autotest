#!/usr/bin/env python3
"""Shared docker-compose / service YAML helpers (extract + env defaults).

Used by workflow.py and optuna_runner.py. Paths are caller-owned; this
module only reads the given compose file (CWD-relative paths stay as-is
when the caller passes Path relative to CWD).

Extraction prefers a YAML-aware path (parse compose → normalize ``command``
to tokens → read ``--model`` / ``--served-model-name`` / ``--tensor-parallel-size``
/ ``--port``). That correctly handles list-form commands dumped by Optuna
trials (``yaml.safe_dump``), where a naive regex would capture the next
line's list dash (``-``) as the option value. Regex remains as fallback
for string-form commands / non-YAML edge cases.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any


def command_to_tokens(command: Any) -> list[str]:
    """Normalize compose ``command`` (string or list) to a list of CLI tokens."""
    if command is None:
        return []
    if isinstance(command, list):
        tokens: list[str] = []
        for item in command:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                tokens.append(s)
        return tokens
    if isinstance(command, str):
        # Folded (>) / literal (|) scalars → split like a shell would.
        return shlex.split(command)
    raise TypeError(f"unsupported compose command type: {type(command).__name__}")


def _import_yaml():
    try:
        import yaml
    except ImportError:
        return None
    return yaml


def _service_command_tokens(compose_text: str) -> list[str] | None:
    """Parse compose YAML and return command tokens from the first service that has one."""
    yaml = _import_yaml()
    if yaml is None:
        return None
    try:
        data = yaml.safe_load(compose_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    services = data.get("services")
    if not isinstance(services, dict) or not services:
        return None
    for svc in services.values():
        if isinstance(svc, dict) and "command" in svc:
            try:
                return command_to_tokens(svc.get("command"))
            except TypeError:
                return None
    return None


def _token_option_value(tokens: list[str], *options: str) -> str | None:
    """Return the value for ``--opt`` / ``--opt=val`` / ``--opt val`` from tokens.

    Skips a following token that looks like another flag (``-…``) so a bare
    ``--model`` without a value does not swallow the next list dash / switch.
    """
    for i, tok in enumerate(tokens):
        for option in options:
            if tok == option:
                if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("-"):
                    return str(tokens[i + 1])
                return None
            if tok.startswith(option + "="):
                return tok[len(option) + 1 :]
    return None


def _token_serve_model(tokens: list[str]) -> str | None:
    """Extract model from ``… serve <model>`` token sequence (vLLM-style)."""
    for i, tok in enumerate(tokens):
        if tok == "serve" and i + 1 < len(tokens):
            nxt = str(tokens[i + 1])
            if nxt and not nxt.startswith("-"):
                return nxt
    return None


def extract_port_from_compose(compose_file: Path) -> int:
    """Extract host port from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")

    tokens = _service_command_tokens(content)
    if tokens is not None:
        port_s = _token_option_value(tokens, "--port")
        if port_s is not None:
            try:
                return int(_resolve_env_default(port_s))
            except ValueError:
                pass

    # Match literal port: - "8976:8976" or - 8976:8976
    match = re.search(r'(?m)^\s*-\s*"?(\d+):\d+(?:/\w+)?"?\s*$', content)
    if match:
        return int(match.group(1))
    # Match env var with default: - "${HOST_PORT:-8976}:8976"
    match = re.search(r'(?m)^\s*-\s*"?\$\{[^:}]+:-(\d+)\}:\d+(?:/\w+)?"?\s*$', content)
    if match:
        return int(match.group(1))
    # Match --port option in command (string-form regex fallback)
    match = re.search(r'--port\s+(\d+)', content)
    if match:
        return int(match.group(1))
    # Host-network / env-driven ports (no ports: mapping), e.g. VLLM_PORT: "8976"
    match = re.search(
        r'(?m)^\s*(?:VLLM_PORT|HOST_PORT|PORT):\s*["\']?(\d+)["\']?\s*$',
        content,
    )
    if match:
        return int(match.group(1))
    raise RuntimeError(f"Cannot extract port from {compose_file}")


def extract_model_from_compose(compose_file: Path) -> str:
    """Extract model path from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")
    model: str | None = None

    tokens = _service_command_tokens(content)
    if tokens is not None:
        # Prefer long options only: bare "-m" collides with `python -m module`.
        raw = _token_option_value(tokens, "--model", "--model-path")
        if raw is None:
            raw = _token_serve_model(tokens)
        if raw is not None:
            model = _resolve_env_default(raw)

    if not model:
        # Try --model / --model-path / GGUF-style -m (string-form regex fallback)
        m = re.search(r"(?:--model(?:-path)?|-m)(?:=|\s+)([^\s\"']+)", content)
        if m:
            cand = m.group(1)
            # List-form dumps leave a lone "-" on the next line; reject it.
            if cand != "-":
                model = _resolve_env_default(cand)
    if not model:
        # Try "<engine> serve <model>" (e.g. "vllm serve ...", "tokenspeed serve ...")
        m = re.search(r"\w+\s+serve\s+([^\s\"'\\]+)", content)
        if m:
            cand = m.group(1)
            if cand != "-":
                model = _resolve_env_default(cand)
    if not model:
        # Host-network / env-driven model path, e.g. MODEL_PATH: /models
        m = re.search(
            r'(?m)^\s*(?:MODEL_PATH|MODEL_DIR|MODEL|TOKENIZER):\s*["\']?([^"\'\s]+)["\']?\s*$',
            content,
        )
        if m:
            model = _resolve_env_default(m.group(1))
    if not model:
        raise RuntimeError(f"Cannot extract model from {compose_file}")
    return _remap_model_via_volumes(model, content)


def extract_served_model_name(compose_file: Path) -> str | None:
    content = compose_file.read_text(encoding="utf-8")

    tokens = _service_command_tokens(content)
    if tokens is not None:
        raw = _token_option_value(tokens, "--served-model-name", "--alias")
        if raw is not None:
            return _resolve_env_default(raw)

    m = re.search(r"(?:--served-model-name|--alias)(?:=|\s+)([^\s\"']+)", content)
    if m and m.group(1) != "-":
        return _resolve_env_default(m.group(1))
    m = re.search(
        r'(?m)^\s*(?:SERVED_MODEL_NAME|MODEL_NAME):\s*["\']?([^"\'\s]+)["\']?\s*$',
        content,
    )
    return _resolve_env_default(m.group(1)) if m else None


def extract_tp_from_compose(compose_file: Path) -> str:
    """Extract tensor-parallel-size from a docker-compose yml file."""
    content = compose_file.read_text(encoding="utf-8")

    tokens = _service_command_tokens(content)
    if tokens is not None:
        raw = _token_option_value(tokens, "--tensor-parallel-size")
        if raw is not None:
            return _resolve_env_default(raw)

    m = re.search(r"--tensor-parallel-size(?:=|\s+)([^\s\"']+)", content)
    if m and m.group(1) != "-":
        return _resolve_env_default(m.group(1))
    m = re.search(
        r'(?m)^\s*(?:TP_SIZE|TENSOR_PARALLEL|TENSOR_PARALLEL_SIZE):\s*["\']?([^"\'\s]+)["\']?\s*$',
        content,
    )
    if m:
        return _resolve_env_default(m.group(1))
    return "1"


def _resolve_env_default(value: str) -> str:
    """Resolve ${VAR:-default} to just 'default'. Pass through literals."""
    m = re.match(r'^\$\{[^:}]+:-(.+)\}$', value)
    if m:
        return m.group(1)
    return value


def _remap_model_via_volumes(model: str, compose_text: str) -> str:
    """Map a container model path to the host bind-mount source when possible."""
    if not model:
        return model
    # Paths must not span whitespace/newlines or a bind without :mode eats the next line.
    vol_re = r'(?m)^\s*-\s*["\']?([^\s:"\']+):([^\s:"\']+)(?::[^\s"\']*)?["\']?\s*$'
    for match in re.finditer(vol_re, compose_text):
        source, target = match.group(1).strip(), match.group(2).strip()
        if not source.startswith(("./", "../", "/")):
            continue
        if model == target or model.startswith(target.rstrip("/") + "/"):
            suffix = model[len(target):] if model.startswith(target) else ""
            return f"{source}{suffix}"
    return model
