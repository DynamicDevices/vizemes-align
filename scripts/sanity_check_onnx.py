#!/usr/bin/env python3
"""Sanity-check a viseme smoke ONNX (+ sidecar JSON).

Default mode (--probes): run ~20 hard-coded inputs aimed at each viseme and
print a human-readable expected→predicted table (good vs weak ONNX signal).

Optional --subset mode: score causal windows from data/tensors/<subset>.

  nix develop .#train --command \\
    python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx
  python3 scripts/build_viseme_probes.py   # regenerate probe bank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBES = ROOT / "data" / "tensors" / "ci-fixture" / "viseme_probes.npz"


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


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "onnx",
        type=Path,
        nargs="?",
        default=ROOT / "export" / "ci-smoke" / "model.onnx",
        help="Path to model.onnx (sidecar model.json expected beside it)",
    )
    ap.add_argument(
        "--mode",
        choices=("probes", "subset"),
        default="probes",
        help="probes=hard-coded viseme bank (default); subset=tensor windows",
    )
    ap.add_argument(
        "--probes",
        type=Path,
        default=DEFAULT_PROBES,
        help="viseme_probes.npz from scripts/build_viseme_probes.py",
    )
    ap.add_argument("--subset", default="ci-fixture", help="data/tensors/<subset>")
    ap.add_argument("--limit", type=int, default=1, help="Utterances to load (0=all)")
    ap.add_argument("--max-frames", type=int, default=64, help="Cap windows for subset mode")
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
    id_to_name = {int(v): k for k, v in meta["visemes"].items()}
    if in_features != ctx * n_mels:
        print(
            f"WARN meta input_features={in_features} != context*n_mels={ctx * n_mels}",
            file=sys.stderr,
        )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    tags = None
    sources = None
    if args.mode == "probes":
        if not args.probes.is_file():
            print(
                f"Missing probes {args.probes}; run: python3 scripts/build_viseme_probes.py",
                file=sys.stderr,
            )
            return 1
        z = np.load(args.probes, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        tags = [str(t) for t in z["tags"].tolist()]
        sources = [str(s) for s in z["source"].tolist()]
    else:
        tdir = ROOT / "data" / "tensors" / args.subset
        index_path = tdir / "index.json"
        if not index_path.is_file():
            print(f"No tensors at {tdir}", file=sys.stderr)
            return 1
        index = json.loads(index_path.read_text(encoding="utf-8"))
        utts = index["utterances"]
        if args.limit:
            utts = utts[: args.limit]
        Xs, ys = [], []
        for u in utts:
            zz = np.load(tdir / u["path"])
            xw, yw = windows(zz["X"], zz["y"], ctx)
            Xs.append(xw)
            ys.append(yw)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if args.max_frames and X.shape[0] > args.max_frames:
            X = X[: args.max_frames]
            y = y[: args.max_frames]

    if X.shape[1] != in_features:
        print(f"BAD input width {X.shape[1]} != meta {in_features}", file=sys.stderr)
        return 1

    logits = sess.run([out_name], {in_name: X})[0]
    if logits.ndim != 2 or logits.shape[0] != X.shape[0] or logits.shape[1] != n_visemes:
        print(
            f"BAD logits shape {logits.shape}; expected ({X.shape[0]}, {n_visemes})",
            file=sys.stderr,
        )
        return 1

    probs = softmax(logits)
    pred = logits.argmax(axis=-1)
    acc = float((pred == y).mean())

    print(
        f"ONNX_SANITY_OK file={onnx_path.name} mode={args.mode} "
        f"in={in_name}{tuple(X.shape)} out={out_name}{tuple(logits.shape)}"
    )
    print(
        f"  meta: context={ctx} n_mels={n_mels} n_visemes={n_visemes} "
        f"hit_rate={acc:.3f} ({int((pred == y).sum())}/{len(y)})"
    )

    if args.mode == "probes":
        print("  probe                  src    expected  predicted  P(exp)  P(pred)  hit")
        for i in range(len(y)):
            exp_n = id_to_name.get(int(y[i]), str(y[i]))
            pred_n = id_to_name.get(int(pred[i]), str(pred[i]))
            tag = tags[i] if tags else f"p{i}"
            src = sources[i] if sources else "?"
            hit = "Y" if pred[i] == y[i] else "."
            print(
                f"  {tag:22s} {src:6s} {exp_n:8s}  {pred_n:8s}  "
                f"{probs[i, y[i]]:6.3f}  {probs[i, pred[i]]:6.3f}  {hit}"
            )
        # per-viseme summary
        print("  per-viseme hits:")
        for vid in range(n_visemes):
            mask = y == vid
            if not mask.any():
                continue
            hits = int((pred[mask] == vid).sum())
            n = int(mask.sum())
            name = id_to_name.get(vid, str(vid))
            print(f"    {name:8s} {hits}/{n}")
    else:
        print(f"  sample pred[:8]={pred[:8].tolist()}")
        print(f"  frame_acc_vs_labels={acc:.3f} frames={len(y)} real_tensors={args.subset}")
        print(f"  unique_pred={sorted(set(pred.tolist()))[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
