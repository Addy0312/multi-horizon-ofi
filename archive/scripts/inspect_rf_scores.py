#!/usr/bin/env python3
"""Inspect and interpret RF classification metrics from rf_wallbridge_results.json.

Usage:
  python scripts/inspect_rf_scores.py
  python scripts/inspect_rf_scores.py --results results/rf_wallbridge_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _safe_float(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None, ndigits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{ndigits}f}"


def _band(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"

    if metric in {"accuracy", "f1_weighted"}:
        if value >= 0.75:
            return "strong"
        if value >= 0.60:
            return "moderate"
        if value >= 0.45:
            return "weak"
        return "poor"

    if metric in {"f1_macro", "precision_macro", "recall_macro"}:
        if value >= 0.60:
            return "strong"
        if value >= 0.45:
            return "moderate"
        if value >= 0.33:
            return "weak"
        return "poor"

    return "n/a"


def _interpret_horizon(metrics: dict[str, Any], h: int) -> list[str]:
    acc = _safe_float(metrics, f"h{h}_accuracy")
    f1m = _safe_float(metrics, f"h{h}_f1_macro")
    f1w = _safe_float(metrics, f"h{h}_f1_weighted")
    pm = _safe_float(metrics, f"h{h}_precision_macro")
    rm = _safe_float(metrics, f"h{h}_recall_macro")

    notes: list[str] = []

    if acc is not None and f1m is not None:
        gap = acc - f1m
        if gap > 0.20:
            notes.append(
                "Large gap between accuracy and macro-F1: likely class imbalance or majority-class behavior."
            )

    if f1m is not None and f1w is not None and (f1w - f1m) > 0.15:
        notes.append(
            "Weighted-F1 much higher than macro-F1: minority classes are likely performing poorly."
        )

    if acc is not None and f1m is not None and f1w is not None:
        if acc > 0.99 and f1m > 0.99 and f1w > 0.99:
            notes.append(
                "Near-perfect metrics: verify no leakage and inspect class distribution."
            )

    if pm is not None and rm is not None:
        pr_gap = abs(pm - rm)
        if pr_gap > 0.10:
            notes.append(
                "Macro-precision and macro-recall differ significantly; decision threshold behavior may be skewed."
            )

    if not notes:
        notes.append("No obvious metric pathology detected at this horizon.")

    return notes


def analyze_results(results_path: str) -> int:
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return 1

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("test_metrics", {})
    horizons = payload.get("horizons", [10, 20, 50, 100])

    print("=" * 92)
    print("Random Forest Results: Metric Summary")
    print("=" * 92)
    print(f"results_path: {results_path}")
    print(f"timestamp   : {payload.get('timestamp', '-')}")
    print(f"tickers     : {payload.get('tickers', '-')}")
    print(f"n_train     : {payload.get('n_train', '-')}")
    print(f"n_val       : {payload.get('n_val', '-')}")
    print(f"n_test      : {payload.get('n_test', '-')}")
    print("-" * 92)
    print(
        f"{'h':>6} | {'accuracy':>9} | {'f1_macro':>9} | {'f1_weighted':>11} | "
        f"{'precision_m':>11} | {'recall_m':>8}"
    )
    print("-" * 92)

    for h in horizons:
        acc = _safe_float(metrics, f"h{h}_accuracy")
        f1m = _safe_float(metrics, f"h{h}_f1_macro")
        f1w = _safe_float(metrics, f"h{h}_f1_weighted")
        pm = _safe_float(metrics, f"h{h}_precision_macro")
        rm = _safe_float(metrics, f"h{h}_recall_macro")

        print(
            f"{h:>6} | {_fmt(acc):>9} | {_fmt(f1m):>9} | {_fmt(f1w):>11} | "
            f"{_fmt(pm):>11} | {_fmt(rm):>8}"
        )

    print("=" * 92)
    print("How To Read These Metrics")
    print("=" * 92)
    print("accuracy      : overall fraction of correct predictions")
    print("f1_macro      : class-balanced F1 (each class weighted equally)")
    print("f1_weighted   : F1 weighted by class frequency (major classes dominate)")
    print("precision_m   : macro precision across classes")
    print("recall_m      : macro recall across classes")

    print("\nPer-horizon interpretation:")
    for h in horizons:
        acc = _safe_float(metrics, f"h{h}_accuracy")
        f1m = _safe_float(metrics, f"h{h}_f1_macro")
        pm = _safe_float(metrics, f"h{h}_precision_macro")
        rm = _safe_float(metrics, f"h{h}_recall_macro")
        print("-" * 92)
        print(f"h={h}")
        print(f"  accuracy band    : {_band('accuracy', acc)}")
        print(f"  macro-F1 band    : {_band('f1_macro', f1m)}")
        print(f"  macro-prec band  : {_band('precision_macro', pm)}")
        print(f"  macro-recall band: {_band('recall_macro', rm)}")
        for note in _interpret_horizon(metrics, h):
            print(f"  note: {note}")

    print("=" * 92)
    print("Recommended sanity checks if metrics look too good")
    print("=" * 92)
    print("1) Check class counts in train/val/test per horizon (down/neutral/up).")
    print("2) Compare against a trivial baseline that always predicts the majority class.")
    print("3) Verify temporal split has no leakage from future events.")
    print("4) Evaluate confusion matrix per horizon to inspect minority classes.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and interpret RF result metrics")
    parser.add_argument(
        "--results",
        type=str,
        default="results/rf_wallbridge_results.json",
        help="Path to results JSON",
    )
    args = parser.parse_args()
    return analyze_results(args.results)


if __name__ == "__main__":
    raise SystemExit(main())
