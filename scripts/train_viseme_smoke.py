#!/usr/bin/env python3
"""Minimal train smoke: frame classifier on context windows; print val accuracy.

Stage: training (quality measure before Godot/ONNX). Uses data/tensors/<subset>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from embed_model_metadata import embed_onnx, estimate_latency  # noqa: E402
from model_contract import load_model_contract  # noqa: E402


def windows(X: np.ndarray, y: np.ndarray, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal past context: frame i uses X[i-ctx+1:i+1], label y[i]."""
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
    ap.add_argument("--context", type=int, default=20, help="Past mel frames (ablate vs 100)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only the first N utterances from index (0 = all). Good for 16GB laptops.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap total training windows after concat (0 = unlimited).",
    )
    ap.add_argument("--hidden", type=int, default=64, help="MLP hidden size (default 64 for smoke)")
    ap.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        help="Write a self-describing ONNX after training, e.g. data/export/smoke/model.onnx",
    )
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    # Cap BLAS/OMP threads so smoke does not peg every core on a laptop.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    tdir = ROOT / "data" / "tensors" / args.subset
    index_path = tdir / "index.json"
    if not index_path.exists():
        print(f"Missing {index_path}; run build_train_tensors.py first", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text())
    n_visemes = len(index["visemes"])
    utterances = index["utterances"]
    if args.limit:
        utterances = utterances[: args.limit]

    Xs, ys = [], []
    for u in utterances:
        z = np.load(tdir / u["path"])
        xw, yw = windows(z["X"], z["y"], args.context)
        Xs.append(xw)
        ys.append(yw)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if args.max_frames and X.shape[0] > args.max_frames:
        X = X[: args.max_frames]
        y = y[: args.max_frames]

    # tiny held-out: last 20% of frames (smoke only)
    n = X.shape[0]
    n_val = max(1, n // 5)
    Xtr, ytr = X[:-n_val], y[:-n_val]
    Xva, yva = X[-n_val:], y[-n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = int(X.shape[1])
    model = nn.Sequential(
        nn.Linear(in_features, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, n_visemes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def batch_acc(xb, yb):
        with torch.no_grad():
            pred = model(xb).argmax(dim=-1)
            return float((pred == yb).float().mean().item())

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        model.eval()
        va = batch_acc(Xva_t, yva_t)
        best = max(best, va)
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            tr = batch_acc(Xtr_t, ytr_t)
            print(
                f"epoch {ep:3d}  loss={loss.item():.4f}  "
                f"train_acc={tr:.3f}  val_acc={va:.3f}"
            )

    print(
        f"SMOKE_OK subset={args.subset} context={args.context} "
        f"frames={n} utterances={len(utterances)} hidden={args.hidden} "
        f"n_visemes={n_visemes} best_val_acc={best:.3f} device={device}"
    )

    if args.export_onnx is not None:
        out = args.export_onnx
        out.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = model.to("cpu").eval()
        dummy = torch.zeros(1, in_features, dtype=torch.float32)
        torch.onnx.export(
            model_cpu,
            dummy,
            str(out),
            input_names=["mel_context"],
            output_names=["viseme_logits"],
            dynamic_axes={"mel_context": {0: "batch"}, "viseme_logits": {0: "batch"}},
            opset_version=17,
        )
        meta = {
            "model": "viseme_smoke_mlp",
            "doc": "Smoke MLP over flattened causal log-Mel context.",
            "subset": args.subset,
            "context_frames": args.context,
            "n_mels": int(index["audio"]["n_mels"]),
            "input_features": in_features,
            "hidden": args.hidden,
            "n_visemes": n_visemes,
            "visemes": index["visemes"],
            "audio": index["audio"],
            "best_val_acc": best,
            "frames_trained": n,
            "utterances": len(utterances),
            "onnx": out.name,
            "normalization": "per_utterance_per_mel_mean_std",
            "lookahead_ms": 0.0,
            "checkpoint": "best-validation-observed/final-weights",
            "quality": {"best_validation_frame_accuracy": best},
            "note": "Input is flattened causal mel context: (batch, context*n_mels). "
            "Runtime mel SoT: OpenLipSync/audio/mel_spectrogram.c (match torchaudio).",
        }
        meta["latency"] = estimate_latency(index["audio"], args.context, 0.0)
        embed_onnx(out, meta)
        contract = load_model_contract(out, require_schema=2)
        assert contract["input_features"] == in_features
        print(f"ONNX_OK {out} schema={contract['_schema']} metadata=embedded")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
