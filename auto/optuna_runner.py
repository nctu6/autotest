#!/usr/bin/env python3
"""Simplified Optuna runner: tune server CLI args from opt.yml.

Schema (flat — see example-opt-dir/opt.yml):
  direction, compose, test, concurrency, health_timeout,
  output, metric, args (name -> categorical list).
  n_trials is unused: runner auto-computes full grid size as the product of
  len(choices) for each args key and runs Optuna GridSampler over all combos.

Each trial:
  1. Optuna suggests values for args
  2. If ``output/trial_N/`` already has complete results (params.json +
     scored CSV), skip the run: reuse the CSV score for Optuna and count
     as **skip** (log ``[skip] trial N — results already exist``)
  3. Else load base compose as YAML → patch service ``command`` tokens
     from args → dump temp compose under output/trial_N/ → compose up →
     health → one test at fixed concurrency → score CSV → compose down
  4. Report score to Optuna (**success**). Compose/health/test/parse
     errors prune the trial (study continues) and count as **failed**.

Final log: success / failed / skip counts.

No env search space, no Hydra, no UnieConfig, no compose ${VAR} requirement
for tunable params (literals are written into the trial compose command).
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from compose_util import (
    command_to_tokens,
    extract_model_from_compose,
    extract_port_from_compose,
    extract_served_model_name,
    extract_tp_from_compose,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths / YAML
# ---------------------------------------------------------------------------

def resolve_cwd_path(raw: str | Path) -> Path:
    """Resolve relative paths against process CWD; absolute paths unchanged."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def load_opt_yaml(opt_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --optuna. Install with: pip install pyyaml"
        ) from exc
    with opt_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"opt.yml must be a mapping: {opt_path}")
    return data


def _cli_option(name: str) -> str:
    """Normalize a bare name or ``--name`` to ``--name``."""
    name = str(name).strip()
    if not name:
        raise ValueError("empty arg name")
    return name if name.startswith("--") else f"--{name}"


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --optuna. Install with: pip install pyyaml"
        ) from exc
    return yaml


# ---------------------------------------------------------------------------
# Compose command patching via YAML tree (unit-testable; no docker)
# ---------------------------------------------------------------------------

def _find_option_span(tokens: list[str], option: str) -> tuple[int, int] | None:
    """Return ``(start, end_exclusive)`` covering ``--opt`` / ``--opt=val`` / ``--opt val``."""
    for i, tok in enumerate(tokens):
        if tok == option:
            # Following value only if it does not look like another flag/switch.
            if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("-"):
                return (i, i + 2)
            return (i, i + 1)
        if tok.startswith(option + "="):
            return (i, i + 1)
    return None


def _set_bool_token(tokens: list[str], option: str, enabled: bool) -> list[str]:
    """Ensure boolean ``--name`` is present (True) or removed (False)."""
    span = _find_option_span(tokens, option)
    if not enabled:
        if span is None:
            return tokens
        start, end = span
        # Remove --name and any following value if it was a valued flag.
        return tokens[:start] + tokens[end:]
    # True → bare switch (no value). Replace valued form if present.
    if span is not None:
        start, end = span
        return tokens[:start] + [option] + tokens[end:]
    return tokens + [option]


def _set_valued_token(tokens: list[str], option: str, value: object) -> list[str]:
    """Set ``--name <value>`` (replace existing or append)."""
    value_s = str(value)
    span = _find_option_span(tokens, option)
    if span is not None:
        start, end = span
        return tokens[:start] + [option, value_s] + tokens[end:]
    return tokens + [option, value_s]


def apply_args_to_tokens(
    tokens: list[str],
    args: dict[str, Any] | None = None,
) -> list[str]:
    """Apply opt.yml ``args`` values onto a CLI token list."""
    out = list(tokens)
    for name, value in (args or {}).items():
        option = _cli_option(name)
        if isinstance(value, bool):
            out = _set_bool_token(out, option, value)
        else:
            out = _set_valued_token(out, option, value)
    return out


def _service_with_command(data: dict[str, Any]) -> dict[str, Any]:
    """Return the first service mapping that has a ``command`` (else first service)."""
    services = data.get("services")
    if not isinstance(services, dict) or not services:
        raise RuntimeError("compose YAML has no services")
    for svc in services.values():
        if isinstance(svc, dict) and "command" in svc:
            return svc
    first = next(iter(services.values()))
    if not isinstance(first, dict):
        raise RuntimeError("compose service is not a mapping")
    return first


def patch_compose_data(
    data: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate ``data`` in place: rewrite service command tokens from ``args``."""
    svc = _service_with_command(data)
    tokens = command_to_tokens(svc.get("command"))
    svc["command"] = apply_args_to_tokens(tokens, args)
    return data


def materialize_trial_compose(
    base_compose: Path,
    trial_dir: Path,
    args: dict[str, Any],
) -> Path:
    """Load base compose YAML, patch command from args, dump under trial_dir."""
    yaml = _import_yaml()
    trial_dir.mkdir(parents=True, exist_ok=True)
    dest = trial_dir / base_compose.name

    with base_compose.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"compose must be a mapping: {base_compose}")

    data = copy.deepcopy(data)
    patch_compose_data(data, args=args)

    with dest.open("w", encoding="utf-8") as f:
        # Comments from the base file are lost in the temp compose (OK).
        # Prefer list-form command for clarity.
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return dest


# ---------------------------------------------------------------------------
# Suggest / score helpers
# ---------------------------------------------------------------------------

def suggest_from_opt(
    trial: Any,
    args_space: dict[str, list],
) -> dict[str, Any]:
    """Suggest categorical args from the single opt.yml args search space."""
    args: dict[str, Any] = {}
    for name, choices in (args_space or {}).items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"args[{name!r}] must be a non-empty list")
        args[name] = trial.suggest_categorical(f"arg:{name}", list(choices))
    return args


def grid_search_space(args_space: dict[str, list]) -> dict[str, list]:
    """Build Optuna GridSampler search_space keyed like suggest names (arg:name)."""
    space: dict[str, list] = {}
    for name, choices in (args_space or {}).items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"args[{name!r}] must be a non-empty list")
        space[f"arg:{name}"] = list(choices)
    return space


def grid_size_from_args(args_space: dict[str, list]) -> int:
    """Full combinatorial grid size = product of len(choices) for each arg."""
    from math import prod

    lengths: list[int] = []
    for name, choices in (args_space or {}).items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"args[{name!r}] must be a non-empty list")
        lengths.append(len(choices))
    return int(prod(lengths)) if lengths else 0


def metric_from_parsed(parsed: dict, metric_key: str) -> float | None:
    """Pull a numeric metric; try exact key then ' / Mean' suffix."""
    candidates = [metric_key, f"{metric_key} / Mean"]
    for key in candidates:
        if key not in parsed:
            continue
        raw = parsed[key]
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def trial_results_exist(trial_dir: Path) -> bool:
    """True when trial dir looks complete: params.json + at least one CSV.

    Mirrors workflow ``result_exists`` intent (skip re-run) with a lighter
    check suited to optuna trial dirs (params + scored CSV).
    """
    if not trial_dir.is_dir():
        return False
    if not (trial_dir / "params.json").is_file():
        return False
    return any(trial_dir.glob("*.csv"))


def score_from_trial_csvs(
    trial_dir: Path,
    metric_key: str,
    parse_guidellm_csv,
) -> float | None:
    """Mean metric across ``*.csv`` in trial_dir, or None if unreadable."""
    csv_files = sorted(trial_dir.glob("*.csv"))
    if not csv_files:
        return None
    scores: list[float] = []
    for csv_path in csv_files:
        try:
            parsed = parse_guidellm_csv(csv_path)
        except Exception as exc:
            LOG.warning(
                "[optuna] failed to parse %s: %s", csv_path.name, exc
            )
            continue
        score = metric_from_parsed(parsed, metric_key)
        if score is None:
            LOG.warning(
                "[optuna] metric %r missing in %s", metric_key, csv_path.name
            )
            continue
        scores.append(score)
        LOG.info(
            "[optuna] %s %s -> %s=%s",
            trial_dir.name,
            csv_path.name,
            metric_key,
            score,
        )
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def params_flat(args: dict[str, Any]) -> dict[str, Any]:
    """Return args as a flat dict for JSON, dotenv, and Optuna logging."""
    return dict(args)


def write_dotenv(path: Path, params: dict[str, Any]) -> None:
    """Write best params as KEY=value lines (CLI names kept as-is)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Best Optuna CLI args (from opt.yml search space)",
        "# Keys are argument names without leading -- when stored bare in opt.yml.",
    ]
    for k in sorted(params):
        v = params[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main study loop
# ---------------------------------------------------------------------------

def run_optuna(opt_path: Path) -> int:
    """Run an Optuna study from the simplified opt.yml. Returns exit code."""
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        LOG.error(
            "optuna is required for --optuna. Install with: pip install optuna"
        )
        LOG.error("%s", exc)
        return 1

    # Import workflow helpers lazily to avoid circular import at module load.
    import workflow as wf

    try:
        from export import parse_guidellm_csv
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export import parse_guidellm_csv

    opt_path = resolve_cwd_path(opt_path)
    if not opt_path.is_file():
        LOG.error("opt.yml not found: %s", opt_path)
        return 1

    try:
        cfg = load_opt_yaml(opt_path)
    except Exception as exc:
        LOG.error("[optuna] failed to load YAML: %s", exc)
        return 1

    # --- flat schema (MUST match example-opt-dir/opt.yml) ---
    direction = str(cfg.get("direction", "maximize")).lower()
    if direction not in ("maximize", "minimize"):
        LOG.error("[optuna] direction must be maximize|minimize, got %r", direction)
        return 1

    compose_raw = cfg.get("compose")
    test_raw = cfg.get("test")
    if not compose_raw or not test_raw:
        LOG.error("[optuna] opt.yml requires 'compose' and 'test' paths")
        return 1

    compose_file = resolve_cwd_path(compose_raw)
    test_file = resolve_cwd_path(test_raw)
    if not compose_file.is_file():
        LOG.error("[optuna] compose not found: %s", compose_file)
        return 1
    if not test_file.is_file():
        LOG.error("[optuna] test script not found: %s", test_file)
        return 1

    concurrency = cfg.get("concurrency")
    concurrency_i = int(concurrency) if concurrency is not None else None
    health_timeout = int(
        cfg.get("health_timeout", wf.DEFAULT_HEALTH_TIMEOUT_SEC)
    )
    output_root = resolve_cwd_path(cfg.get("output", "./auto/results/optuna"))
    metric_key = str(
        cfg.get("metric") or "Token Throughput / Output Tokens/Sec"
    )
    args_space = cfg.get("args") or {}
    if not isinstance(args_space, dict):
        LOG.error("[optuna] 'args' must be a mapping")
        return 1
    if not args_space:
        LOG.error("[optuna] opt.yml has empty args — nothing to tune")
        return 1

    try:
        search_space = grid_search_space(args_space)
        n_trials = grid_size_from_args(args_space)
    except ValueError as exc:
        LOG.error("[optuna] %s", exc)
        return 1

    # Prefer IGNORE: opt.yml n_trials is unused; always run the full grid.
    if "n_trials" in cfg:
        LOG.info(
            "[optuna] ignoring opt.yml n_trials=%s; using full grid of %s trials",
            cfg.get("n_trials"),
            n_trials,
        )
    else:
        LOG.info("[optuna] using full grid of %s trials", n_trials)

    # Reject leftover UnieConfig / env-DSL keys loudly (user rejected that style)
    legacy_keys = [k for k in ("study", "workflow", "objective", "parameters", "env") if k in cfg]
    if legacy_keys:
        LOG.warning(
            "[optuna] ignoring legacy UnieConfig keys in opt.yml: %s "
            "(params live only under args)",
            ", ".join(legacy_keys),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    wf.set_nostop(False)

    LOG.info("[optuna] config: %s", opt_path)
    LOG.info(
        "[optuna] direction=%s n_trials=%s metric=%s",
        direction,
        n_trials,
        metric_key,
    )
    LOG.info(
        "[optuna] full grid size = %s from args keys %s",
        n_trials,
        list(args_space.keys()),
    )
    LOG.info("[optuna] compose=%s", compose_file)
    LOG.info("[optuna] test=%s concurrency=%s", test_file, concurrency_i)
    LOG.info("[optuna] output=%s", output_root)
    LOG.info("[optuna] args=%s", list(args_space.keys()))

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction=direction, sampler=sampler)
    # Local UX counters (independent of Optuna COMPLETE/PRUNED):
    #   success — fresh run produced a score
    #   skip    — trial_N results already existed (score reused or bypassed)
    #   failed  — compose/health/test/materialize/parse/CSV errors (pruned)
    stats = {"success": 0, "failed": 0, "skip": 0}
    # Per-trial wall-clock seconds (includes skip reuse / score-from-CSV path).
    trial_durations: dict[int, float] = {}
    trial_outcomes: dict[int, str] = {}  # success | failed | skip

    def objective(trial: "optuna.Trial") -> float:
        t0 = time.perf_counter()
        outcome = "failed"
        try:
            if wf._INTERRUPTED:
                raise optuna.TrialPruned("interrupted")

            args = suggest_from_opt(trial, args_space)
            flat = params_flat(args)
            trial_dir = output_root / f"trial_{trial.number}"

            # --- skip if complete results already exist under trial_N/ ---
            if trial_results_exist(trial_dir):
                existing = score_from_trial_csvs(
                    trial_dir, metric_key, parse_guidellm_csv
                )
                if existing is not None:
                    LOG.info(
                        "[skip] trial %s — results already exist (score=%s)",
                        trial.number,
                        existing,
                    )
                    stats["skip"] += 1
                    outcome = "skip"
                    return float(existing)
                LOG.warning(
                    "[skip] trial %s — results already exist but metric %r "
                    "unreadable; pruning",
                    trial.number,
                    metric_key,
                )
                stats["skip"] += 1
                outcome = "skip"
                raise optuna.TrialPruned("existing results unreadable")

            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "params.json").write_text(
                json.dumps(flat, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            LOG.info(
                "[optuna] trial %s params: %s",
                trial.number,
                " ".join(f"{k}={v}" for k, v in flat.items()),
            )

            trial_compose: Path | None = None
            try:
                trial_compose = materialize_trial_compose(
                    compose_file, trial_dir, args
                )
            except Exception as exc:
                LOG.exception("[optuna] materialize compose failed: %s", exc)
                stats["failed"] += 1
                outcome = "failed"
                raise optuna.TrialPruned(f"materialize failed: {exc}") from exc

            try:
                port = extract_port_from_compose(trial_compose)
                model = extract_model_from_compose(trial_compose)
                tp = extract_tp_from_compose(trial_compose)
                served_model = extract_served_model_name(trial_compose)
            except Exception as exc:
                LOG.exception("[optuna] parse compose failed: %s", exc)
                stats["failed"] += 1
                outcome = "failed"
                raise optuna.TrialPruned(f"parse failed: {exc}") from exc

            # Stricter workflow.result_exists (json+csv+png) for naming match;
            # covers leftover artifacts without params.json (already handled above).
            try:
                named_exists = wf.result_exists(
                    trial_dir,
                    compose_file.stem,
                    str(tp),
                    test_file.stem,
                    concurrency_i,
                    output_prefix=f"optuna.t{trial.number}",
                )
            except Exception:
                named_exists = False
            if named_exists:
                existing = score_from_trial_csvs(
                    trial_dir, metric_key, parse_guidellm_csv
                )
                stats["skip"] += 1
                outcome = "skip"
                if existing is not None:
                    LOG.info(
                        "[skip] trial %s — results already exist (score=%s)",
                        trial.number,
                        existing,
                    )
                    return float(existing)
                LOG.warning(
                    "[skip] trial %s — results already exist but metric "
                    "unreadable; pruning",
                    trial.number,
                )
                raise optuna.TrialPruned("existing results unreadable")

            compose_started = False
            try:
                if wf._INTERRUPTED:
                    raise optuna.TrialPruned("interrupted")
                wf.wait_for_port_free(port, wf.DEFAULT_PORT_FREE_WAIT_SEC)
                # No extra_env for search params — values are already literal in command.
                wf.compose_up(trial_compose)
                compose_started = True
                wf.wait_for_health(
                    f"http://127.0.0.1:{port}/health",
                    timeout_sec=health_timeout,
                )
                wf.run_test(
                    test_script=test_file,
                    port=port,
                    model=model,
                    concurrency=concurrency_i,
                    output_dir=trial_dir,
                    service_name=compose_file.stem,
                    tp=str(tp),
                    served_model=served_model,
                    output_prefix=f"optuna.t{trial.number}",
                )
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                # Expected skip-continue path (OOM, health fail, etc.) — no traceback
                LOG.error("[optuna] trial %s step failed: %s", trial.number, exc)
                stats["failed"] += 1
                outcome = "failed"
                raise optuna.TrialPruned(f"trial step failed: {exc}") from exc
            finally:
                if trial_compose is not None and (
                    compose_started or trial_compose in wf._ACTIVE_COMPOSE_FILES
                ):
                    wf.compose_down(trial_compose)

            final = score_from_trial_csvs(trial_dir, metric_key, parse_guidellm_csv)
            if final is None:
                LOG.warning(
                    "[optuna] trial %s: no usable CSV metric %r",
                    trial.number,
                    metric_key,
                )
                stats["failed"] += 1
                outcome = "failed"
                raise optuna.TrialPruned(
                    f"metric {metric_key!r} not found in CSVs"
                )

            LOG.info("[optuna] trial %s score=%s", trial.number, final)
            stats["success"] += 1
            outcome = "success"
            return float(final)
        finally:
            duration_sec = round(time.perf_counter() - t0, 3)
            trial_durations[trial.number] = duration_sec
            trial_outcomes[trial.number] = outcome
            try:
                trial.set_user_attr("duration_sec", duration_sec)
                trial.set_user_attr("outcome", outcome)
            except Exception:
                pass
            LOG.info(
                "[optuna] trial %s duration_sec=%.3f outcome=%s",
                trial.number,
                duration_sec,
                outcome,
            )


    interrupted = False
    try:
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    except KeyboardInterrupt:
        interrupted = True
        LOG.warning("[optuna] study interrupted")

    LOG.info(
        "[optuna] summary: success=%d failed=%d skip=%d",
        stats["success"],
        stats["failed"],
        stats["skip"],
    )

    # Timing aggregates (wall clock per trial body, including skip reuse).
    durations_all = [trial_durations[n] for n in sorted(trial_durations)]
    total_sec = round(sum(durations_all), 3) if durations_all else 0.0
    avg_sec = (
        round(sum(durations_all) / len(durations_all), 3) if durations_all else 0.0
    )
    success_durs = [
        trial_durations[n]
        for n in sorted(trial_durations)
        if trial_outcomes.get(n) == "success"
    ]
    avg_sec_success = (
        round(sum(success_durs) / len(success_durs), 3) if success_durs else None
    )
    trials_timing = [
        {
            "number": n,
            "duration_sec": trial_durations[n],
            "outcome": trial_outcomes.get(n, "?"),
        }
        for n in sorted(trial_durations)
    ]
    LOG.info(
        "[optuna] timing: total_sec=%.3f avg_sec=%.3f avg_sec_success=%s "
        "(%d trials with duration)",
        total_sec,
        avg_sec,
        avg_sec_success if avg_sec_success is not None else "n/a",
        len(durations_all),
    )

    # Always persist counters + timing for export.py --optuna summary.
    stats_payload = {
        "success": stats["success"],
        "failed": stats["failed"],
        "skip": stats["skip"],
        "metric": metric_key,
        "direction": direction,
        "total_sec": total_sec,
        "avg_sec": avg_sec,
        "avg_sec_success": avg_sec_success,
        "trials": trials_timing,
    }
    stats_path = output_root / "study_stats.json"
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG.info("[optuna] wrote %s", stats_path)

    if interrupted:
        return 130

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        LOG.error("[optuna] no completed trials — nothing to write")
        return 1

    best = study.best_trial
    # Rebuild human-friendly params (strip arg: prefixes from Optuna names)
    best_params: dict[str, Any] = {}
    for k, v in best.params.items():
        if k.startswith("arg:"):
            best_params[k[4:]] = v
        else:
            best_params[k] = v

    best_params_path = output_root / "best_params.json"
    dotenv_path = output_root / ".env"
    trials_csv_path = output_root / "trials.csv"

    payload = {
        "best_trial": best.number,
        "best_value": best.value,
        "direction": direction,
        "metric": metric_key,
        "compose": str(compose_file),
        "test": str(test_file),
        "concurrency": concurrency_i,
        "params": best_params,
        "args": best_params,
        "stats": stats_payload,
        "total_sec": total_sec,
        "avg_sec": avg_sec,
        "avg_sec_success": avg_sec_success,
        "best_trial_duration_sec": trial_durations.get(best.number),
    }
    best_params_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG.info("[optuna] wrote %s", best_params_path)

    write_dotenv(dotenv_path, best_params)
    LOG.info("[optuna] wrote %s", dotenv_path)

    try:
        df = study.trials_dataframe()
        # Ensure plain duration_sec / outcome columns (not only user_attrs_*).
        if "number" in df.columns:
            df["duration_sec"] = df["number"].map(
                lambda n: trial_durations.get(int(n))
            )
            df["outcome"] = df["number"].map(
                lambda n: trial_outcomes.get(int(n))
            )
        df.to_csv(trials_csv_path, index=False)
        LOG.info("[optuna] wrote %s", trials_csv_path)
    except Exception as exc:
        LOG.warning("[optuna] failed to write trials.csv: %s", exc)

    # Keep the end-of-run summary readable and consistent with export.py.
    LOG.info("[optuna] best trial: #%s", best.number)
    LOG.info("[optuna] best %s: %s", metric_key, best.value)
    LOG.info("[optuna] direction: %s", direction)
    LOG.info("[optuna] best params:")
    for key in sorted(best_params):
        value = best_params[key]
        if isinstance(value, bool):
            LOG.info("  --%s  (%s)", key, "on" if value else "off")
        else:
            LOG.info("  --%s %s", key, value)
    LOG.info(
        "[optuna] summary: success=%s failed=%s skip=%s",
        stats["success"],
        stats["failed"],
        stats["skip"],
    )
    LOG.info("[optuna] total time: %ss", total_sec)

    return 0


if __name__ == "__main__":
    # Minimal CLI for isolated smoke / debugging (prefer workflow.py --optuna).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <opt.yml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_optuna(Path(sys.argv[1])))
