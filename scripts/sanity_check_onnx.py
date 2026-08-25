#!/usr/bin/env python3
#!/usr/bin/env python3
"""ONNX sanity — short & readable, not clever.

Loads export/ci-smoke/demo_inputs.csv (one probe per viseme when available).
Runs inference one row at a time; prints expect vs predict.

  nix develop .#train --command python3 scripts/sanity_check_onnx.py
  python3 scripts/build_demo_inputs.py   # rebuild CSV/npz from tensors
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "export" / "ci-smoke" / "model.onnx"
CSV_PATH = ROOT / "export" / "ci-smoke" / "demo_inputs.csv"


def main() -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Need onnxruntime (nix develop .#train)", file=sys.stderr)
        return 1

    meta = json.loads(ONNX.with_suffix(".json").read_text())
    names = {int(v): k for k, v in meta["visemes"].items()}
    n_feat = int(meta["input_features"])
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    xin = sess.get_inputs()[0].name

    print(f"{'probe':>5}  {'expect':8}  {'predict':8}  {'P(exp)':>7}  hit")
    hits = n = 0
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            expect_id = int(row["expect_id"])
            x = [[float(row[f"f{i}"]) for i in range(n_feat)]]
            logits = sess.run(None, {xin: x})[0][0]

            m = max(logits)
            probs = [math.exp(float(v - m)) for v in logits]
            z = sum(probs)
            probs = [p / z for p in probs]
            pred_id = max(range(len(logits)), key=lambda i: logits[i])
            hit = pred_id == expect_id
            hits += int(hit)
            print(
                f"{int(row['probe_id']):5d}  {row['expect_name']:8}  "
                f"{names[pred_id]:8}  {probs[expect_id]:7.3f}  "
                f"{'Y' if hit else '.'}"
            )
    print(f"hit_rate={hits}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
