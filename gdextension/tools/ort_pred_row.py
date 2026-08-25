#!/usr/bin/env python3
"""Print Python-ORT argmax for demo row (C smoke must match — not ground-truth labels)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "export" / "ci-smoke" / "model.onnx"
DEFAULT_NPZ = ROOT / "export" / "ci-smoke" / "demo_inputs.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        print("Need onnxruntime (nix develop .#train or pip)", file=sys.stderr)
        return 1
    z = np.load(args.npz)
    X = np.asarray(z["X"][args.row : args.row + 1], dtype=np.float32)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    logits = sess.run([out], {inp: X})[0]
    print(int(logits.argmax(axis=-1)[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
