#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize MLPR hyperparameter sensitivity results.

When multiple seeds are present for the same hyperparameter value, this script
reports mean +/- sample standard deviation across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results" / "hyperparam_sensitivity"
DEFAULT_RESULTS_DIR = ROOT / "results"


def format_float(value: float) -> str:
    return format(float(value), ".10g")


def format_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} +/- {float(std or 0.0):.4f}"


def load_records(input_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("alpha_*_eta_*_seed_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload["_path"] = path
            records.append(payload)
    return records


def is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def mean_std(values: Sequence[Any]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def select_rows(records: Sequence[Dict[str, Any]], key: str, fixed_key: str, fixed_value: float) -> List[Dict[str, Any]]:
    rows = [record for record in records if is_close(record[fixed_key], fixed_value)]
    grouped: Dict[float, List[Dict[str, Any]]] = {}
    for record in rows:
        grouped.setdefault(float(record[key]), []).append(record)

    out: List[Dict[str, Any]] = []
    for value, group in sorted(grouped.items()):
        acc_mean, acc_std = mean_std([record.get("accuracy") for record in group])
        f1_mean, f1_std = mean_std([record.get("macro_f1") for record in group])
        auc_mean, auc_std = mean_std([record.get("macro_auc") for record in group])
        ordered = sorted(group, key=lambda item: int(item.get("seed", 0)))
        out.append({
            key: value,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "macro_auc_mean": auc_mean,
            "macro_auc_std": auc_std,
            "n": len(group),
            "seeds": ",".join(str(record.get("seed")) for record in ordered),
        })
    return out


def write_csv(path: Path, value_key: str, rows: Sequence[Dict[str, Any]]) -> None:
    header_name = "alpha" if value_key == "inner_lr" else "eta"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            header_name,
            "Accuracy_mean",
            "Accuracy_std",
            "Macro-F1_mean",
            "Macro-F1_std",
            "Macro-AUC_mean",
            "Macro-AUC_std",
            "n",
            "seeds",
        ])
        for row in rows:
            writer.writerow([
                format_float(row[value_key]),
                row.get("accuracy_mean"),
                row.get("accuracy_std"),
                row.get("macro_f1_mean"),
                row.get("macro_f1_std"),
                row.get("macro_auc_mean"),
                row.get("macro_auc_std"),
                row.get("n"),
                row.get("seeds"),
            ])


def write_pdf(path: Path, value_key: str, rows: Sequence[Dict[str, Any]], default_value: float) -> None:
    label = "Inner Learning Rate alpha" if value_key == "inner_lr" else "Outer Learning Rate eta"
    x_values = [float(row[value_key]) for row in rows if row.get("macro_f1_mean") is not None]
    y_values = [float(row["macro_f1_mean"]) for row in rows if row.get("macro_f1_mean") is not None]
    if not x_values:
        raise ValueError(f"No valid Macro-F1 values found for {label}")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.0))
    plt.semilogx(x_values, y_values, marker="o", linewidth=1.8, markersize=6)
    plt.axvline(float(default_value), color="red", linestyle="--", linewidth=1.5, label=f"default={format_float(default_value)}")
    plt.xlabel(label)
    plt.ylabel("Macro-F1")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def markdown_table(title: str, value_header: str, value_key: str, rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        title,
        f"| {value_header} | Accuracy | Macro-F1 | Macro-AUC | n | seeds |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {format_float(row[value_key])} | "
            f"{format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))} | "
            f"{format_mean_std(row.get('macro_f1_mean'), row.get('macro_f1_std'))} | "
            f"{format_mean_std(row.get('macro_auc_mean'), row.get('macro_auc_std'))} | "
            f"{row.get('n')} | {row.get('seeds')} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MLPR hyperparameter sensitivity results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    records = load_records(input_dir)
    if not records:
        raise FileNotFoundError(f"No sensitivity JSON files found under {input_dir}")

    default_alpha = float(records[0].get("default_inner_lr", 0.01))
    default_eta = float(records[0].get("default_meta_lr", 1e-4))

    alpha_rows = select_rows(records, "inner_lr", "meta_lr", default_eta)
    eta_rows = select_rows(records, "meta_lr", "inner_lr", default_alpha)

    alpha_csv = results_dir / "alpha_sensitivity.csv"
    eta_csv = results_dir / "eta_sensitivity.csv"
    alpha_pdf = results_dir / "alpha_sensitivity.pdf"
    eta_pdf = results_dir / "eta_sensitivity.pdf"

    write_csv(alpha_csv, "inner_lr", alpha_rows)
    write_csv(eta_csv, "meta_lr", eta_rows)
    write_pdf(alpha_pdf, "inner_lr", alpha_rows, default_alpha)
    write_pdf(eta_pdf, "meta_lr", eta_rows, default_eta)

    print(markdown_table("### Alpha Sensitivity", "alpha", "inner_lr", alpha_rows))
    print()
    print(markdown_table("### Eta Sensitivity", "eta", "meta_lr", eta_rows))
    print()
    print(f"Saved CSV: {alpha_csv}")
    print(f"Saved CSV: {eta_csv}")
    print(f"Saved PDF: {alpha_pdf}")
    print(f"Saved PDF: {eta_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
