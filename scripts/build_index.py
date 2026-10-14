#!/usr/bin/env python3
"""Build a reproducible Lexis disk index and save construction metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexis.scalable import build_disk_index, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("spimi", "bsbi"), default="spimi")
    parser.add_argument("--memory-limit-terms", type=int, default=10_000)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    _, metrics = build_disk_index(
        args.corpus,
        args.output,
        method=args.method,
        memory_limit_terms=args.memory_limit_terms,
        shards=args.shards,
    )
    save_metrics(metrics, args.output / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
