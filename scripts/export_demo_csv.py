#!/usr/bin/env python3
"""Export export/ci-smoke/demo_inputs.npz → demo_inputs.csv (one probe per row)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "export" / "ci-smoke" / "demo_inputs.npz"
CSV_OUT = ROOT / "export" / "ci-smoke" / "demo_inputs.csv"
MODEL = ROOT / "export" / "ci-smoke" / "model.onnx"
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import id_to_label, load_model_contract  # noqa: E402


def main() -> int:
    meta = load_model_contract(MODEL)
    id_to_name = id_to_label(meta)
    z = np.load(NPZ)
    X, y = z["X"], z["y"]
    n_feat = X.shape[1]
    header = ["probe_id", "expect_id", "expect_name"] + [f"f{i}" for i in range(n_feat)]
    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (row, yi) in enumerate(zip(X, y)):
            yi = int(yi)
            w.writerow([i, yi, id_to_name[yi]] + [f"{v:.6g}" for v in row.tolist()])
    print(f"wrote {CSV_OUT} rows={len(y)} feats={n_feat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
