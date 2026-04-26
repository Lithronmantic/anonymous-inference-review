#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper for MLPR hyperparameter sensitivity runs.

This script does not modify any existing training code. It creates a temporary
config with overridden MLPR hyperparameters, then delegates the actual training
to the existing benchmark entrypoint: ``scripts/run_ssl_benchmark.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "configs" / "paper_exact_eswa.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hyperparam_sensitivity"
DEFAULT_RAW_OUTPUT_ROOT = ROOT / "results" / "hyperparam_sensitivity_runs"
DEFAULT_TMP_CONFIG_ROOT = ROOT / "results" / "hyperparam_sensitivity_configs"


def format_float(value: float) -> str:
    return format(float(value), ".10g")


def ratio_tag(labeled_ratio: float) -> str:
    return f"ratio{int(round(float(labeled_ratio) * 100.0)):02d}"


def experiment_tag(inner_lr: float, meta_lr: float, seed: int) -> str:
    return f"alpha_{format_float(inner_lr)}_eta_{format_float(meta_lr)}_seed_{seed}"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict JSON at {path}, got {type(payload)}")
    return payload


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict YAML at {path}, got {type(payload)}")
    return payload


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def has_valid_metrics(payload: Dict[str, Any]) -> bool:
    return all(payload.get(key) is not None for key in ("accuracy", "macro_f1", "macro_auc"))


def read_training_time_minutes(epoch_metrics_path: Optional[Path]) -> Optional[float]:
    if epoch_metrics_path is None or not epoch_metrics_path.exists():
        return None
    total_seconds = 0.0
    with epoch_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("epoch_time_s")
            if raw in (None, ""):
                continue
            total_seconds += float(raw)
    return total_seconds / 60.0


def build_output_path(inner_lr: float, meta_lr: float, seed: int, output_dir: Path) -> Path:
    filename = f"{experiment_tag(inner_lr, meta_lr, seed)}.json"
    return output_dir / filename


def build_final_payload(
    *,
    raw_result: Dict[str, Any],
    inner_lr: float,
    meta_lr: float,
    default_inner_lr: float,
    default_meta_lr: float,
    seed: int,
    labeled_ratio: float,
    num_epochs: Optional[int],
    source: str,
    base_config: Path,
    raw_results_json: Optional[Path],
    existing_result_json: Optional[Path],
    temp_config: Optional[Path],
    raw_output_root: Optional[Path],
    return_code: int,
) -> Dict[str, Any]:
    epoch_metrics_csv = raw_result.get("epoch_metrics_csv")
    epoch_metrics_path = resolve_repo_path(epoch_metrics_csv) if epoch_metrics_csv else None
    training_time_minutes = read_training_time_minutes(epoch_metrics_path)
    payload: Dict[str, Any] = {
        "inner_lr": float(inner_lr),
        "meta_lr": float(meta_lr),
        "default_inner_lr": float(default_inner_lr),
        "default_meta_lr": float(default_meta_lr),
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "num_epochs_requested": int(num_epochs) if num_epochs is not None else None,
        "accuracy": raw_result.get("acc"),
        "macro_f1": raw_result.get("f1_macro"),
        "macro_auc": raw_result.get("auc_macro"),
        "training_time_minutes": training_time_minutes,
        "best_f1_during_training": raw_result.get("best_f1_during_training"),
        "source": source,
        "return_code": int(return_code),
        "base_config": repo_relative(base_config),
        "raw_results_json": repo_relative(raw_results_json) if raw_results_json else None,
        "existing_result_json": repo_relative(existing_result_json) if existing_result_json else None,
        "temp_config": repo_relative(temp_config) if temp_config else None,
        "raw_output_root": repo_relative(raw_output_root) if raw_output_root else None,
        "raw_output_dir": raw_result.get("output_dir"),
        "raw_best_ckpt": raw_result.get("best_ckpt"),
        "raw_epoch_metrics_csv": raw_result.get("epoch_metrics_csv"),
        "error": raw_result.get("error"),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MLPR hyperparameter sensitivity experiment.")
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--meta-lr", type=float, required=True)
    parser.add_argument("--default-inner-lr", type=float, required=True)
    parser.add_argument("--default-meta-lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-ratio", type=float, default=0.25)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--tmp-config-root", type=Path, default=DEFAULT_TMP_CONFIG_ROOT)
    parser.add_argument("--reuse-result", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def maybe_skip_existing(output_json: Path, skip_existing: bool) -> bool:
    if not skip_existing or not output_json.exists():
        return False
    try:
        existing = load_json(output_json)
    except Exception:
        return False
    if has_valid_metrics(existing):
        print(f"[SKIP] Existing completed result: {repo_relative(output_json)}")
        return True
    print(f"[RETRY] Existing result is incomplete, rebuilding: {repo_relative(output_json)}")
    return False


def build_temp_config(
    base_config: Path,
    tmp_config_path: Path,
    inner_lr: float,
    meta_lr: float,
    num_epochs: Optional[int],
) -> None:
    cfg = load_yaml(base_config)
    cfg.setdefault("mlpr", {})
    cfg.setdefault("training", {})
    cfg["mlpr"]["enabled"] = True
    cfg["mlpr"]["inner_lr"] = float(inner_lr)
    cfg["mlpr"]["inner_lr_alpha"] = float(inner_lr)
    cfg["mlpr"]["meta_lr"] = float(meta_lr)
    cfg["training"]["ssl_method"] = "ours_mlpr"
    cfg["training"]["use_ssl"] = True
    if num_epochs is not None:
        cfg["training"]["num_epochs"] = int(num_epochs)
        # Ensure "统一跑 50 epoch" really means the trainer will not stop earlier.
        cfg["training"]["early_stop_min_epochs"] = int(num_epochs)
        cfg["training"]["early_stop_patience"] = max(
            int(cfg["training"].get("early_stop_patience", 0)),
            int(num_epochs) + 1,
        )
    dump_yaml(tmp_config_path, cfg)


def write_reused_result(args: argparse.Namespace, output_json: Path) -> int:
    reuse_result = resolve_repo_path(args.reuse_result)
    if not reuse_result.exists():
        raise FileNotFoundError(f"Reuse result not found: {reuse_result}")
    raw_result = load_json(reuse_result)
    payload = build_final_payload(
        raw_result=raw_result,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        default_inner_lr=args.default_inner_lr,
        default_meta_lr=args.default_meta_lr,
        seed=args.seed,
        labeled_ratio=args.labeled_ratio,
        num_epochs=args.num_epochs,
        source="reused_existing",
        base_config=resolve_repo_path(args.base_config),
        raw_results_json=reuse_result,
        existing_result_json=reuse_result,
        temp_config=None,
        raw_output_root=None,
        return_code=0,
    )
    write_payload(output_json, payload)
    print(f"[REUSE] Wrote {repo_relative(output_json)} from {repo_relative(reuse_result)}")
    return 0 if has_valid_metrics(payload) else 1


def run_new_experiment(args: argparse.Namespace, output_json: Path) -> int:
    base_config = resolve_repo_path(args.base_config)
    raw_output_root = resolve_repo_path(args.raw_output_root) / experiment_tag(args.inner_lr, args.meta_lr, args.seed)
    tmp_config_path = resolve_repo_path(args.tmp_config_root) / f"{experiment_tag(args.inner_lr, args.meta_lr, args.seed)}.yaml"
    build_temp_config(base_config, tmp_config_path, args.inner_lr, args.meta_lr, args.num_epochs)

    cmd = [
        args.python,
        str(ROOT / "scripts" / "run_ssl_benchmark.py"),
        "--config_template",
        str(tmp_config_path),
        "--output_root",
        str(raw_output_root),
        "--seeds",
        str(args.seed),
        "--labeled_ratios",
        str(args.labeled_ratio),
        "--ssl_methods",
        "ours_mlpr",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port),
        "--python",
        args.python,
    ]
    if args.num_epochs is not None:
        # Epoch override is written into the temp config, so no extra CLI flag is needed.
        pass
    if args.smoke_epochs > 0:
        cmd.extend(["--smoke_epochs", str(args.smoke_epochs)])
    if args.skip_existing:
        cmd.append("--skip_existing")

    if args.dry_run:
        print("[DRY-RUN] Benchmark command:")
        print(" ".join(cmd))
        return 0

    print(f"[RUN] inner_lr={args.inner_lr} meta_lr={args.meta_lr} seed={args.seed}")
    print(f"[RUN] temp_config={repo_relative(tmp_config_path)}")
    proc = subprocess.run(cmd, cwd=ROOT, check=False)

    raw_results_json = raw_output_root / "ours_mlpr" / ratio_tag(args.labeled_ratio) / f"seed{args.seed}" / "results.json"
    if not raw_results_json.exists():
        raw_result: Dict[str, Any] = {
            "acc": None,
            "f1_macro": None,
            "auc_macro": None,
            "error": f"Missing raw results.json after benchmark run (return_code={proc.returncode})",
        }
    else:
        raw_result = load_json(raw_results_json)

    payload = build_final_payload(
        raw_result=raw_result,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        default_inner_lr=args.default_inner_lr,
        default_meta_lr=args.default_meta_lr,
        seed=args.seed,
        labeled_ratio=args.labeled_ratio,
        num_epochs=args.num_epochs,
        source="new_run",
        base_config=base_config,
        raw_results_json=raw_results_json if raw_results_json.exists() else None,
        existing_result_json=None,
        temp_config=tmp_config_path,
        raw_output_root=raw_output_root,
        return_code=proc.returncode,
    )
    write_payload(output_json, payload)

    if has_valid_metrics(payload):
        print(
            "[DONE] "
            f"accuracy={payload['accuracy']:.4f} "
            f"macro_f1={payload['macro_f1']:.4f} "
            f"macro_auc={payload['macro_auc']:.4f} "
            f"minutes={payload['training_time_minutes']:.2f}"
        )
        return 0

    print(f"[FAIL] Wrote incomplete result to {repo_relative(output_json)}")
    return 1


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_json = resolve_repo_path(args.output_json) if args.output_json else build_output_path(
        args.inner_lr,
        args.meta_lr,
        args.seed,
        output_dir,
    )

    if maybe_skip_existing(output_json, args.skip_existing):
        return 0

    if args.reuse_result is not None:
        if args.dry_run:
            print(
                "[DRY-RUN] Reuse existing result:",
                repo_relative(resolve_repo_path(args.reuse_result)),
                "->",
                repo_relative(output_json),
            )
            return 0
        return write_reused_result(args, output_json)

    return run_new_experiment(args, output_json)


if __name__ == "__main__":
    raise SystemExit(main())
