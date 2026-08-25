#!/usr/bin/env python3
"""Sanity-check a viseme smoke ONNX (+ sidecar JSON).

Default: run ~20 **hardcoded** labeled mel-context windows from
`export/ci-smoke/demo_inputs.npz` and print a human table
(expect viseme → predict viseme). That makes a good vs bad ONNX obvious.

Also supports live tensors via `--subset`.

  nix develop .#train
  python3 scripts/sanity_check_onnx.py
  python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx
  python3 scripts/sanity_check_onnx.py --subset ci-fixture --limit 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "export" / "ci-smoke" / "model.onnx"
DEFAULT_DEMO = ROOT / "export" / "ci-smoke" / "demo_inputs.npz"


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
    ap.add_argument("onnx", type=Path, nargs="?", default=DEFAULT_ONNX)
    ap.add_argument(
        "--demo",
        type=Path,
        default=DEFAULT_DEMO,
        help="Hardcoded labeled inputs (.npz with X,y). Empty path disables.",
    )
    ap.add_argument("--subset", default="", help="If set, load data/tensors/<subset> instead of --demo")
    ap.add_argument("--limit", type=int, default=1, help="Utterances when using --subset (0=all)")
    ap.add_argument("--max-frames", type=int, default=64, help="Cap windows when using --subset")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "Need onnxruntime in this Python (use: nix develop .#train)",
            file=sys.stderr,
        )
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
    id_to_name = {int(v): str(k) for k, v in meta["visemes"].items()}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    source = "demo"
    if args.subset:
        tdir = ROOT / "data" / "tensors" / args.subset
        index_path = tdir / "index.json"
        if not index_path.is_file():
            print(f"Missing {index_path}", file=sys.stderr)
            return 1
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
            X, y = X[: args.max_frames], y[: args.max_frames]
        source = f"tensors:{args.subset}"
    else:
        demo_path = Path(args.demo) if args.demo else DEFAULT_DEMO
        if not demo_path.is_file():
            print(f"Missing demo inputs: {demo_path}", file=sys.stderr)
            return 1
        z = np.load(demo_path)
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        source = f"hardcoded:{demo_path.name}"

    if X.ndim != 2 or X.shape[1] != in_features:
        print(f"BAD input shape {X.shape}; expected (_, {in_features})", file=sys.stderr)
        return 1

    logits = sess.run([out_name], {in_name: X})[0]
    if logits.shape != (X.shape[0], n_visemes):
        print(f"BAD logits shape {logits.shape}", file=sys.stderr)
        return 1

    pred = logits.argmax(axis=-1)
    ok = pred == y
    acc = float(ok.mean()) if len(y) else 0.0

    print(
        f"ONNX_SANITY_OK file={onnx_path.name} source={source} "
        f"in={in_name}{tuple(X.shape)} out={out_name}{tuple(logits.shape)}"
    )
    print(
        f"  meta: context={ctx} n_mels={n_mels} n_visemes={n_visemes} "
        f"subset={meta.get('subset')} best_val_acc={meta.get('best_val_acc')}"
    )
    print()
    print(f"{'#':>3}  {'expect':<10} {'predict':<10}  ok  top1_logit")
    print("-" * 48)
    for i in range(len(y)):
        exp = id_to_name.get(int(y[i]), str(int(y[i])))
        pr = id_to_name.get(int(pred[i]), str(int(pred[i])))
        mark = "Y" if ok[i] else "n"
        print(f"{i:3d}  {exp:<10} {pr:<10}  {mark}  {float(logits[i, pred[i]]):7.3f}")
    print("-" * 48)
    print(f"frame_acc_vs_labels={acc:.3f}  N={len(y)}")

    # Coverage vs full viseme set (human quality cue)
    present = sorted(set(int(v) for v in y.tolist()))
    missing = [id_to_name[i] for i in range(n_visemes) if i not in present]
    print(f"labels_in_demo={ [id_to_name[i] for i in present] }")
    if missing:
        print(
            f"labels_missing_from_demo={missing}  "
            "(smoke fixture is tiny — train on a fuller subset for a real quality read)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
