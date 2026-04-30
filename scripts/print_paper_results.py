"""Print machine-readable paper results as Markdown tables.

Usage:
    python -m scripts.print_paper_results --table confidence_controls
    python -m scripts.print_paper_results --table version_shift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "paper_results.json"


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _print_table(rows: Iterable[dict], columns: list[str]) -> None:
    rows = list(rows)
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--table",
        choices=[
            "main_results",
            "confidence_controls",
            "selection_diagnostics",
            "version_shift",
            "e2e_rag",
        ],
        default="confidence_controls",
    )
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    if args.table == "confidence_controls":
        rows = data["confidence_controls_ord_qa"]
        cols = [
            "variant",
            "confidence_source",
            "stop_gradient",
            "weighted_training",
            "top_k_star",
            "ndcg@5",
            "delta_ndcg@5",
            "ece",
            "stale@5",
        ]
    elif args.table == "main_results":
        rows = [
            {
                "backbone": r["backbone"],
                "variant": r["variant"],
                "ord_ndcg@5": r["ord_qa"]["ndcg@5"],
                "hotpotqa_ndcg@5": r["hotpotqa"]["ndcg@5"],
            }
            for r in data["main_results"]
        ]
        cols = ["backbone", "variant", "ord_ndcg@5", "hotpotqa_ndcg@5"]
    else:
        rows = data[args.table]
        cols = list(rows[0].keys()) if rows else []
    _print_table(rows, cols)


if __name__ == "__main__":
    main()
