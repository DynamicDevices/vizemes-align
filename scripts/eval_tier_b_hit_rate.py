#!/usr/bin/env python3
"""MFA vs ONNX hit-rate for tier-B (and optional baseline) on held-out stems.

Uses the same soft-label lookahead shift as train_viseme_tier_b.py so scores
match training. Prints per-stem and overall tables; writes JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_viseme_tier_b import apply_lookahead, soft_boundary_targets  # noqa: E402
from model_contract import load_model_contract  # noqa: E402


def windows_xy(
    X: np.ndarray,
    y: np.ndarray,
    ctx: int,
    n_visemes: int,
    lag_frames: int,
    blend_frames: int,
    soft: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y = apply_lookahead(y.astype(np.int64), lag_frames)
    if X.shape[0] < ctx:
        pad = np.repeat(X[:1], ctx - X.shape[0], axis=0)
        X = np.concatenate([pad, X], axis=0)
        y = np.concatenate([np.repeat(y[:1], ctx - y.shape[0]), y], axis=0)
    targets = soft_boundary_targets(y, n_visemes, blend_frames) if soft else np.eye(
        n_visemes, dtype=np.float32
    )[y]
    xs, hard = [], []
    for i in range(ctx - 1, X.shape[0]):
        xs.append(X[i - ctx + 1 : i + 1].reshape(-1))
        hard.append(int(np.argmax(targets[i])))
    return np.stack(xs).astype(np.float32), np.asarray(hard, dtype=np.int64)


def run_model(onnx_path: Path, Xw: np.ndarray, batch: int = 512) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xin = sess.get_inputs()[0].name
    preds = []
    for i in range(0, Xw.shape[0], batch):
        chunk = Xw[i : i + batch]
        logits = sess.run(None, {xin: chunk})[0]
        preds.append(np.argmax(logits, axis=-1))
    return np.concatenate(preds, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "export" / "tier-b",
        help="Directory with self-describing model_*.onnx files",
    )
    ap.add_argument(
        "--baseline-onnx",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "model.onnx",
        help="Optional smoke/baseline ONNX for comparison",
    )
    ap.add_argument("--stems", nargs="*", default=[], help="Utterance ids (default: 5 val stems)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tdir = ROOT / "data" / "tensors" / args.subset
    index = json.loads((tdir / "index.json").read_text())
    meta = load_model_contract(args.model_dir / "model_final.onnx", require_schema=2)
    n_visemes = int(meta["n_visemes"])
    ctx = int(meta["context_frames"])
    lag = int(meta.get("lookahead_frames", 5))
    blend = int(meta.get("blend_frames", 6))
    soft = bool(meta.get("soft_boundaries", True))

    ids = sorted(u["id"] for u in index["utterances"])
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = ids[-n_val:]
    stems = args.stems or val_ids[len(val_ids) // 2 : len(val_ids) // 2 + 5]
    by_id = {u["id"]: u for u in index["utterances"]}

    models = {
        "tier_b_10m": args.model_dir / "model_10m.onnx",
        "tier_b_20m": args.model_dir / "model_20m.onnx",
        "tier_b_final": args.model_dir / "model_final.onnx",
    }
    if args.baseline_onnx.is_file():
        models["ci_smoke"] = args.baseline_onnx

    id_to_name = {int(v): str(k) for k, v in meta["visemes"].items()}
    rows = []
    print(
        f"{'stem':22} {'model':14} {'hit':>7} {'n':>6} {'acc':>7}  mid_acc  (non-silence)"
    )
    for stem in stems:
        u = by_id.get(stem)
        if not u:
            print(f"skip missing {stem}", file=sys.stderr)
            continue
        z = np.load(tdir / u["path"])
        Xw, y_hard = windows_xy(z["X"], z["y"], ctx, n_visemes, lag, blend, soft)
        nonsil = y_hard != 0
        for name, path in models.items():
            if not path.is_file():
                continue
            pred = run_model(path, Xw)
            hits = int((pred == y_hard).sum())
            n = int(y_hard.size)
            acc = hits / max(1, n)
            if nonsil.any():
                mid_hits = int((pred[nonsil] == y_hard[nonsil]).sum())
                mid_n = int(nonsil.sum())
                mid_acc = mid_hits / max(1, mid_n)
            else:
                mid_hits = mid_n = 0
                mid_acc = 0.0
            print(
                f"{stem:22} {name:14} {hits:5d}/{n:<5d} {acc:7.3f}  {mid_hits}/{mid_n}={mid_acc:.3f}"
            )
            rows.append(
                {
                    "stem": stem,
                    "model": name,
                    "hits": hits,
                    "n": n,
                    "acc": round(acc, 4),
                    "mid_hits": mid_hits,
                    "mid_n": mid_n,
                    "mid_acc": round(mid_acc, 4),
                }
            )

    # Aggregate per model
    print("\n=== overall ===")
    summary = {"stems": stems, "lookahead_frames": lag, "blend_frames": blend, "models": {}}
    for name in models:
        rs = [r for r in rows if r["model"] == name]
        if not rs:
            continue
        hits = sum(r["hits"] for r in rs)
        n = sum(r["n"] for r in rs)
        mh = sum(r["mid_hits"] for r in rs)
        mn = sum(r["mid_n"] for r in rs)
        summary["models"][name] = {
            "hits": hits,
            "n": n,
            "acc": round(hits / max(1, n), 4),
            "mid_hits": mh,
            "mid_n": mn,
            "mid_acc": round(mh / max(1, mn), 4),
        }
        print(
            f"{name:14}  all={hits}/{n}={hits/max(1,n):.3f}  "
            f"non-sil={mh}/{mn}={mh/max(1,mn):.3f}"
        )

    out = args.out or (args.model_dir / "hit_rate_summary.json")
    out.write_text(json.dumps({"per_stem": rows, "overall": summary}, indent=2) + "\n")
    print(f"\nwrote {out}")
    _ = id_to_name  # reserved for future confusion matrix
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
