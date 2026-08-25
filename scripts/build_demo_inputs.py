#!/usr/bin/env python3
"""Build demo_inputs.npz + .csv: one causal mel window per viseme (when present).

Picks real windows from data/tensors/<subset> so the sanity table is human-
readable and non-zero. Prefers coverage of every viseme id in model.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def windows(X: np.ndarray, y: np.ndarray, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] < ctx:
        pad = np.repeat(X[:1], ctx - X.shape[0], axis=0)
        X = np.concatenate([pad, X], axis=0)
        y = np.concatenate([np.repeat(y[:1], ctx - y.shape[0]), y], axis=0)
    xs, ys = [], []
    for i in range(ctx - 1, X.shape[0]):
        xs.append(X[i - ctx + 1 : i + 1].reshape(-1))
        ys.append(y[i])
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="ci-fixture")
    ap.add_argument("--context", type=int, default=0, help="0 = use model.json")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "export" / "ci-smoke",
    )
    args = ap.parse_args()

    meta_path = args.out_dir / "model.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ctx = args.context or int(meta["context_frames"])
    n_feat = int(meta["input_features"])
    n_visemes = int(meta["n_visemes"])
    id_to_name = {int(v): str(k) for k, v in meta["visemes"].items()}

    tdir = ROOT / "data" / "tensors" / args.subset
    index = json.loads((tdir / "index.json").read_text(encoding="utf-8"))
    Xs, ys = [], []
    for u in index["utterances"]:
        z = np.load(tdir / u["path"])
        xw, yw = windows(z["X"], z["y"], ctx)
        Xs.append(xw)
        ys.append(yw)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if X.shape[1] != n_feat:
        print(f"BAD feat dim {X.shape[1]} != {n_feat}", file=sys.stderr)
        return 1

    # One strongest non-zero window per viseme id (prefer absmax).
    picks: list[tuple[int, int]] = []  # (row_index, class)
    missing: list[str] = []
    for vid in range(n_visemes):
        idxs = np.where(y == vid)[0]
        if len(idxs) == 0:
            missing.append(id_to_name[vid])
            continue
        # Prefer non-zero rows
        scored = [(float(np.abs(X[i]).max()), int(i)) for i in idxs]
        scored.sort(reverse=True)
        best = scored[0][1]
        if scored[0][0] <= 0.0:
            print(f"WARN viseme {id_to_name[vid]} windows are all-zero", file=sys.stderr)
        picks.append((best, vid))

    if not picks:
        print("No labeled windows found", file=sys.stderr)
        return 1

    X_out = np.stack([X[i] for i, _ in picks]).astype(np.float32)
    y_out = np.asarray([vid for _, vid in picks], dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "demo_inputs.npz"
    csv_path = args.out_dir / "demo_inputs.csv"
    np.savez_compressed(npz_path, X=X_out, y=y_out)

    header = ["probe_id", "expect_id", "expect_name"] + [f"f{i}" for i in range(n_feat)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (row, yi) in enumerate(zip(X_out, y_out)):
            w.writerow([i, int(yi), id_to_name[int(yi)]] + [f"{v:.6g}" for v in row.tolist()])

    print(
        f"wrote {npz_path.name} + {csv_path.name} "
        f"rows={len(y_out)} labels={[id_to_name[int(v)] for v in y_out]}"
    )
    if missing:
        print(f"labels_missing={missing}")
    print(f"absmax={float(np.abs(X_out).max()):.4f} nonzero_rows={int((np.abs(X_out).max(axis=1) > 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
