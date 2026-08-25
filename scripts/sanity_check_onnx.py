#!/usr/bin/env python3
"""Sanity-check a viseme smoke ONNX (+ sidecar JSON) against tensor windows.

Loads model.onnx next to model.json (or paths you pass). Builds the same
flattened causal mel context as train_viseme_smoke.py, runs ONNX Runtime,
and prints shape / argmax / optional frame accuracy vs labels.

  nix develop .#train   # or any env with numpy + onnxruntime
  pip install onnxruntime   # if missing
  python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx
  python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx --subset ci-fixture --limit 1
"""
from __future__ import annotations

import argparse
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
    ap.add_argument(
        "onnx",
        type=Path,
        nargs="?",
        default=ROOT / "export" / "ci-smoke" / "model.onnx",
        help="Path to model.onnx (sidecar model.json expected beside it)",
    )
    ap.add_argument("--subset", default="ci-fixture", help="data/tensors/<subset>")
    ap.add_argument("--limit", type=int, default=1, help="Utterances to load (0=all)")
    ap.add_argument("--max-frames", type=int, default=64, help="Cap windows for the check")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("Need onnxruntime: pip install onnxruntime", file=sys.stderr)
        return 1

    onnx_path = args.onnx.resolve()
    meta_path = onnx_path.with_suffix(".json")
    if not onnx_path.is_file():
        print(f"Missing ONNX: {onnx_path}", file=sys.stderr)
        return 1
    if not meta_path.is_file():
        print(f"Missing sidecar JSON: {meta_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ctx = int(meta["context_frames"])
    n_mels = int(meta["n_mels"])
    in_features = int(meta["input_features"])
    n_visemes = int(meta["n_visemes"])
    if in_features != ctx * n_mels:
        print(
            f"WARN meta input_features={in_features} != context*n_mels={ctx * n_mels}",
            file=sys.stderr,
        )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    tdir = ROOT / "data" / "tensors" / args.subset
    index_path = tdir / "index.json"
    used_real = False
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        utts = index["utterances"]
        if args.limit:
            utts = utts[: args.limit]
        Xs, ys = [], []
        for u in utts:
            z = np.load(tdir / u["path"])
            xw, yw = windows(z["X"], z["y"], ctx)
            Xs.append(xw)
            ys.append(yw)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if args.max_frames and X.shape[0] > args.max_frames:
            X = X[: args.max_frames]
            y = y[: args.max_frames]
        used_real = True
    else:
        # Synthetic batch: correct shape only (no label check).
        X = np.random.randn(min(8, args.max_frames or 8), in_features).astype(np.float32)
        y = None
        print(f"No tensors at {tdir}; using random inputs (shape check only)")

    if X.shape[1] != in_features:
        print(f"BAD input width {X.shape[1]} != meta {in_features}", file=sys.stderr)
        return 1

    logits = sess.run([out_name], {in_name: X})[0]
    if logits.ndim != 2 or logits.shape[0] != X.shape[0] or logits.shape[1] != n_visemes:
        print(f"BAD logits shape {logits.shape}; expected ({X.shape[0]}, {n_visemes})", file=sys.stderr)
        return 1

    pred = logits.argmax(axis=-1)
    print(f"ONNX_SANITY_OK file={onnx_path.name} in={in_name}{tuple(X.shape)} out={out_name}{tuple(logits.shape)}")
    print(f"  meta: context={ctx} n_mels={n_mels} n_visemes={n_visemes} subset={meta.get('subset')}")
    print(f"  sample pred[:8]={pred[:8].tolist()}")
    if used_real and y is not None:
        acc = float((pred == y).mean())
        print(f"  frame_acc_vs_labels={acc:.3f} frames={len(y)} real_tensors={args.subset}")
        # Spot-check: at least some class diversity or non-crash
        print(f"  unique_pred={sorted(set(pred.tolist()))[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
