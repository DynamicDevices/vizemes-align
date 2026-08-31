#!/usr/bin/env python3
"""Summarise reproducible three-vs-five-block TCN outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()
    rows = []
    for label in ("three-block", "five-block"):
        meta = json.loads((args.out_root / label / "model.json").read_text())
        rows.append(
            {
                "run": label,
                "layers": meta["layers"],
                "params": meta["params"],
                "history_hops": meta["receptive_history_hops"],
                "history_ms": meta["receptive_history_ms"],
                "steps": meta["step"],
                "seed": meta["seed"],
                "fit_acc": meta["fit_quality"]["all_acc"],
                "held_out_acc": meta["quality"]["all_acc"],
                "best_val_acc": meta["best_val_acc"],
                "train_split_sha256": meta["train_split_sha256"],
                "val_split_sha256": meta["val_split_sha256"],
            }
        )
    if rows[0]["train_split_sha256"] != rows[1]["train_split_sha256"] or rows[0][
        "val_split_sha256"
    ] != rows[1]["val_split_sha256"]:
        raise RuntimeError("comparison runs used different data splits")
    output = {"schema": 1, "runs": rows}
    path = args.out_root / "comparison.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"TCN_DEPTH_COMPARISON_OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
