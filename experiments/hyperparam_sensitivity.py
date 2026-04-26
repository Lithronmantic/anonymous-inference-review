#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner for MLPR hyperparameter sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "experiments" / "train_with_hparams.py"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "paper_exact_eswa.yaml"
DEFAULT_RESULTS_DIR = ROOT / "results" / "hyperparam_sensitivity"
DEFAULT_RAW_OUTPUT_ROOT = ROOT / "results" / "hyperparam_sensitivity_runs"
DEFAULT_TMP_CONFIG_ROOT = ROOT / "results" / "hyperparam_sensitivity_configs"
PREFERRED_REUSE_RESULT = ROOT / "outputs" / "retry5_grid" / "r25" / "ours_mlpr" / "ratio25" / "seed42" / "results.json"


def format_float(value: float) -> str:
    return format(float(value), ".10g")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict YAML at {path}, got {type(payload)}")
    return payload


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(payload)}")
    return payload


def read_epoch_time_minutes(result_json: Path) -> Optional[float]:
    if not result_json.exists():
        return None
    payload = load_json(result_json)
    epoch_csv = payload.get("epoch_metrics_csv")
    if not epoch_csv:
        return None
    epoch_path = Path(epoch_csv)
    if not epoch_path.is_absolute():
        epoch_path = ROOT / epoch_path
    if not epoch_path.exists():
        return None
    total_seconds = 0.0
    with epoch_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("epoch_time_s")
            if raw in (None, ""):
                continue
            total_seconds += float(raw)
    return total_seconds / 60.0


def result_json_path(results_dir: Path, inner_lr: float, meta_lr: float, seed: int) -> Path:
    filename = f"alpha_{format_float(inner_lr)}_eta_{format_float(meta_lr)}_seed_{seed}.json"
    return results_dir / filename


def has_valid_output(output_json: Path) -> bool:
    if not output_json.exists():
        return False
    try:
        payload = load_json(output_json)
    except Exception:
        return False
    return all(payload.get(key) is not None for key in ("accuracy", "macro_f1", "macro_auc"))


def discover_defaults(base_config: Path) -> Tuple[float, float]:
    cfg = load_yaml(base_config)
    mlpr = cfg.get("mlpr", {})
    inner_lr = float(mlpr.get("inner_lr", mlpr.get("inner_lr_alpha", 0.01)))
    meta_lr = float(mlpr.get("meta_lr", 1e-4))
    return inner_lr, meta_lr


def expected_reuse_signature(base_config: Path, default_inner_lr: float) -> Dict[str, Any]:
    cfg = load_yaml(base_config)
    training = cfg.get("training", {})
    mlpr = cfg.get("mlpr", {})
    ssl = training.get("ssl", {})
    return {
        "profile": cfg.get("profile"),
        "epochs": int(training.get("num_epochs", 0)),
        "batch_size": int(training.get("batch_size", 0)),
        "inner_lr": float(default_inner_lr),
        "mlpr_feature_mode": mlpr.get("feature_mode"),
        "learning_rate": float(training.get("learning_rate", 0.0)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "ema_decay_base": float(ssl.get("ema_decay_base", ssl.get("ema_momentum", 0.0))),
    }


def alpha_scan(default_inner_lr: float) -> List[float]:
    values = [0.001, 0.005, default_inner_lr, 0.05, 0.1]
    ordered: List[float] = []
    for value in values:
        if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-12) for existing in ordered):
            ordered.append(float(value))
    return ordered


def eta_scan(default_meta_lr: float) -> List[float]:
    factors = [0.1, 0.5, 1.0, 5.0, 10.0]
    return [float(default_meta_lr * factor) for factor in factors]


def candidate_reuse_results(seed: int, labeled_ratio: float) -> List[Path]:
    ratio_dir = f"ratio{int(round(labeled_ratio * 100)):02d}"
    seed_dir = f"seed{seed}"
    outputs_root = ROOT / "outputs"
    if not outputs_root.exists():
        return []
    return list(outputs_root.glob(f"**/ours_mlpr/{ratio_dir}/{seed_dir}/results.json"))


def read_audit_current(result_json: Path) -> Optional[Dict[str, Any]]:
    audit_path = result_json.parent / "stats" / "config_audit.json"
    if not audit_path.exists():
        return None
    try:
        payload = load_json(audit_path)
    except Exception:
        return None
    current = payload.get("current")
    return current if isinstance(current, dict) else None


def is_reuse_compatible(result_json: Path, expected: Dict[str, Any]) -> bool:
    current = read_audit_current(result_json)
    if current is None:
        return False
    for key, expected_value in expected.items():
        current_value = current.get(key)
        if isinstance(expected_value, float):
            if current_value is None or not math.isclose(float(current_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
                return False
        else:
            if current_value != expected_value:
                return False
    return True


def choose_reuse_result(
    seed: int,
    labeled_ratio: float,
    explicit_path: Optional[Path],
    expected: Dict[str, Any],
) -> Optional[Path]:
    if explicit_path is not None:
        candidate = explicit_path if explicit_path.is_absolute() else ROOT / explicit_path
        return candidate
    if (
        seed == 42
        and math.isclose(labeled_ratio, 0.25, rel_tol=0.0, abs_tol=1e-12)
        and PREFERRED_REUSE_RESULT.exists()
        and is_reuse_compatible(PREFERRED_REUSE_RESULT, expected)
    ):
        return PREFERRED_REUSE_RESULT

    excluded_tokens = ("reviewer_", "same_budget", "raw_no_qc", "clean_vs_raw")
    candidates = []
    for path in candidate_reuse_results(seed, labeled_ratio):
        path_text = path.as_posix().lower()
        penalty = 0
        if any(token in path_text for token in excluded_tokens):
            penalty += 1000
        if "retry5_grid" not in path_text:
            penalty += 100
        penalty += len(path.parts)
        if is_reuse_compatible(path, expected):
            candidates.append((penalty, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].as_posix()))
    return candidates[0][1]


def estimated_total_minutes(reuse_result: Path, num_new_runs: int) -> Optional[float]:
    minutes = read_epoch_time_minutes(reuse_result)
    if minutes is None:
        return None
    return minutes * float(num_new_runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLPR hyperparameter sensitivity experiments.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--tmp-config-root", type=Path, default=DEFAULT_TMP_CONFIG_ROOT)
    parser.add_argument("--reuse-result", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Optional list of seeds. Overrides --seed when provided.")
    parser.add_argument("--labeled-ratio", type=float, default=0.25)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-reuse-default", action="store_true")
    parser.add_argument("--job-indices", type=str, default=None,
                        help="Comma-separated 1-based job indices to run, e.g. 1,2,6")
    parser.add_argument("--list-jobs", action="store_true",
                        help="Print planned job indices and exit.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_job_indices(spec: Optional[str]) -> Optional[set[int]]:
    if spec is None:
        return None
    out: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def invoke_wrapper(
    *,
    inner_lr: float,
    meta_lr: float,
    default_inner_lr: float,
    default_meta_lr: float,
    seed: int,
    labeled_ratio: float,
    base_config: Path,
    output_json: Path,
    raw_output_root: Path,
    tmp_config_root: Path,
    python_exe: str,
    nproc_per_node: int,
    master_port: int,
    num_epochs: Optional[int],
    reuse_result: Optional[Path],
    smoke_epochs: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    cmd = [
        python_exe,
        str(WRAPPER),
        "--inner-lr",
        str(inner_lr),
        "--meta-lr",
        str(meta_lr),
        "--default-inner-lr",
        str(default_inner_lr),
        "--default-meta-lr",
        str(default_meta_lr),
        "--seed",
        str(seed),
        "--labeled-ratio",
        str(labeled_ratio),
        "--base-config",
        str(base_config),
        "--output-json",
        str(output_json),
        "--output-dir",
        str(output_json.parent),
        "--raw-output-root",
        str(raw_output_root),
        "--tmp-config-root",
        str(tmp_config_root),
        "--python",
        python_exe,
        "--nproc-per-node",
        str(nproc_per_node),
        "--master-port",
        str(master_port),
    ]
    if num_epochs is not None:
        cmd.extend(["--num-epochs", str(num_epochs)])
    if reuse_result is not None:
        cmd.extend(["--reuse-result", str(reuse_result)])
    if smoke_epochs > 0:
        cmd.extend(["--smoke-epochs", str(smoke_epochs)])
    if skip_existing:
        cmd.append("--skip-existing")
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def print_plan(
    *,
    reuse_result: Optional[Path],
    default_inner_lr: float,
    default_meta_lr: float,
    alpha_values: Sequence[float],
    eta_values: Sequence[float],
    num_new_runs: int,
    estimated_minutes: Optional[float],
) -> None:
    print("=" * 72, flush=True)
    print("MLPR Hyperparameter Sensitivity Plan", flush=True)
    print(f"Base default inner_lr (alpha): {format_float(default_inner_lr)}", flush=True)
    print(f"Base default meta_lr  (eta)  : {format_float(default_meta_lr)}", flush=True)
    print(
        "Reuse default result         : "
        + (repo_relative(reuse_result) if reuse_result is not None else "none (default point will be rerun)"),
        flush=True,
    )
    print(f"Alpha scan                   : {[format_float(v) for v in alpha_values]}", flush=True)
    print(f"Eta scan                     : {[format_float(v) for v in eta_values]}", flush=True)
    print(f"New training runs            : {num_new_runs}", flush=True)
    if estimated_minutes is not None:
        print(f"Estimated total runtime      : {estimated_minutes:.1f} minutes", flush=True)
    print("=" * 72, flush=True)


def main() -> int:
    args = parse_args()
    base_config = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    raw_output_root = args.raw_output_root if args.raw_output_root.is_absolute() else ROOT / args.raw_output_root
    tmp_config_root = args.tmp_config_root if args.tmp_config_root.is_absolute() else ROOT / args.tmp_config_root
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_output_root.mkdir(parents=True, exist_ok=True)
    tmp_config_root.mkdir(parents=True, exist_ok=True)

    default_inner_lr, default_meta_lr = discover_defaults(base_config)
    alpha_values = alpha_scan(default_inner_lr)
    eta_values = eta_scan(default_meta_lr)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    expected_reuse = expected_reuse_signature(base_config, default_inner_lr)
    skip_existing = not args.force

    planned_jobs: List[Tuple[str, int, float, float, Path, Optional[Path]]] = []
    reuse_results: List[Path] = []
    for seed in seeds:
        reuse_result = None if args.no_reuse_default else choose_reuse_result(
            seed,
            args.labeled_ratio,
            args.reuse_result,
            expected_reuse,
        )
        if reuse_result is not None:
            reuse_results.append(reuse_result)
        default_output = result_json_path(results_dir, default_inner_lr, default_meta_lr, seed)
        if reuse_result is not None:
            planned_jobs.append(("default_reuse", seed, default_inner_lr, default_meta_lr, default_output, reuse_result))
        else:
            planned_jobs.append(("default_run", seed, default_inner_lr, default_meta_lr, default_output, None))
        for alpha_value in alpha_values:
            if math.isclose(alpha_value, default_inner_lr, rel_tol=0.0, abs_tol=1e-12):
                continue
            output_json = result_json_path(results_dir, alpha_value, default_meta_lr, seed)
            planned_jobs.append(("alpha_scan", seed, alpha_value, default_meta_lr, output_json, None))
        for eta_value in eta_values:
            if math.isclose(eta_value, default_meta_lr, rel_tol=0.0, abs_tol=1e-12):
                continue
            output_json = result_json_path(results_dir, default_inner_lr, eta_value, seed)
            planned_jobs.append(("eta_scan", seed, default_inner_lr, eta_value, output_json, None))

    new_run_count = sum(1 for job in planned_jobs if job[0] != "default_reuse")
    eta_minutes = estimated_total_minutes(reuse_results[0], new_run_count) if reuse_results else None
    print_plan(
        reuse_result=reuse_results[0] if reuse_results else None,
        default_inner_lr=default_inner_lr,
        default_meta_lr=default_meta_lr,
        alpha_values=alpha_values,
        eta_values=eta_values,
        num_new_runs=new_run_count,
        estimated_minutes=eta_minutes,
    )

    selected_indices = parse_job_indices(args.job_indices)
    if args.list_jobs:
        for idx, (job_type, seed, inner_lr, meta_lr, output_json, reuse_path) in enumerate(planned_jobs, start=1):
            source = "reuse" if reuse_path is not None else "run"
            print(
                f"{idx}: seed={seed} {job_type} alpha={format_float(inner_lr)} "
                f"eta={format_float(meta_lr)} output={repo_relative(output_json)} mode={source}"
            )
        return 0

    failures = 0
    port = int(args.master_port)
    for idx, (job_type, seed, inner_lr, meta_lr, output_json, reuse_path) in enumerate(planned_jobs, start=1):
        if selected_indices is not None and idx not in selected_indices:
            continue
        label = f"[{idx}/{len(planned_jobs)}] {job_type}"
        print(f"{label}: seed={seed} alpha={format_float(inner_lr)} eta={format_float(meta_lr)}", flush=True)
        rc = invoke_wrapper(
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            default_inner_lr=default_inner_lr,
            default_meta_lr=default_meta_lr,
            seed=seed,
            labeled_ratio=args.labeled_ratio,
            base_config=base_config,
            output_json=output_json,
            raw_output_root=raw_output_root,
            tmp_config_root=tmp_config_root,
            python_exe=args.python,
            nproc_per_node=args.nproc_per_node,
            master_port=port,
            num_epochs=args.num_epochs,
            reuse_result=reuse_path,
            smoke_epochs=args.smoke_epochs,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures += 1
        port += 1

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"Completed planned jobs: {len(planned_jobs)}", flush=True)
    print(f"Failures             : {failures}", flush=True)
    print(f"Results dir          : {repo_relative(results_dir)}", flush=True)
    print("Next step            : python experiments/summarize_sensitivity.py", flush=True)
    print("=" * 72, flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
