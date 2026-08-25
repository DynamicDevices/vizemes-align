#!/usr/bin/env python3
"""Dump demo_inputs.npz row 0 → float32 LE file for gdextension smoke_context."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "export" / "ci-smoke" / "demo_inputs.npz"
DEFAULT_OUT = ROOT / "export" / "ci-smoke" / "demo_row0.f32"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()
    if not args.npz.is_file():
        print(f"missing {args.npz}", file=sys.stderr)
        return 1
    z = np.load(args.npz)
    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.int64)
    if args.row < 0 or args.row >= len(X):
        print(f"row {args.row} out of range N={len(X)}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    X[args.row].tofile(args.out)
    print(f"wrote {args.out} floats={X.shape[1]} expect_class={int(y[args.row])}")
    print(int(y[args.row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
