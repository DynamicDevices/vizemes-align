#!/usr/bin/env python3
"""Tier-B viseme train: Libri aligned subset, soft boundaries, lookahead lag.

Wall-clock limited run with interim ONNX checkpoints and a convergence plot.
Defaults (Julian/Alex 2026-08-29): soft-boundary labels, 50 ms lookahead.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from embed_model_metadata import embed_onnx, estimate_latency  # noqa: E402
from model_contract import load_model_contract  # noqa: E402


def soft_boundary_targets(y: np.ndarray, n_visemes: int, blend_frames: int) -> np.ndarray:
    """One-hot with monotonic linear blend across ±blend_frames at each change."""
    soft = np.eye(n_visemes, dtype=np.float32)[y]
    if blend_frames <= 0 or y.size < 2:
        return soft
    changes = np.where(y[1:] != y[:-1])[0] + 1
    for c in changes:
        left, right = int(y[c - 1]), int(y[c])
        for d in range(-blend_frames, blend_frames + 1):
            j = c + d
            if j < 0 or j >= y.size:
                continue
            alpha = (d + blend_frames) / float(2 * blend_frames)
            soft[j] = 0.0
            soft[j, left] = 1.0 - alpha
            soft[j, right] = alpha
    return soft


def apply_lookahead(y: np.ndarray, lag_frames: int) -> np.ndarray:
    """Label at t comes from t+lag (lookahead); pad end with last label."""
    if lag_frames <= 0:
        return y
    if y.size <= lag_frames:
        return np.repeat(y[-1:], y.size)
    return np.concatenate([y[lag_frames:], np.repeat(y[-1], lag_frames)])


def iter_utterance_windows(
    tdir: Path,
    utterances: list[dict],
    ctx: int,
    n_visemes: int,
    lag_frames: int,
    blend_frames: int,
    soft: bool,
):
    for u in utterances:
        z = np.load(tdir / u["path"])
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        valid = z["valid"].astype(bool) if "valid" in z else np.ones(y.shape, dtype=bool)
        y = apply_lookahead(y, lag_frames)
        valid = apply_lookahead(valid, lag_frames).astype(bool)
        if blend_frames > 0 and not valid.all():
            invalid = np.convolve(
                (~valid).astype(np.int8), np.ones(2 * blend_frames + 1, dtype=np.int8), mode="same"
            )
            valid = invalid == 0
        if X.shape[0] < ctx:
            pad = np.repeat(X[:1], ctx - X.shape[0], axis=0)
            X = np.concatenate([pad, X], axis=0)
            y = np.concatenate([np.repeat(y[:1], ctx - y.shape[0]), y], axis=0)
            valid = np.concatenate([np.repeat(valid[:1], ctx - valid.shape[0]), valid], axis=0)
        targets = soft_boundary_targets(y, n_visemes, blend_frames) if soft else np.eye(
            n_visemes, dtype=np.float32
        )[y]
        for i in range(ctx - 1, X.shape[0]):
            if valid[i]:
                yield X[i - ctx + 1 : i + 1].reshape(-1), targets[i]


def export_onnx(model, path: Path, meta: dict, in_features: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros(1, in_features, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(path),
        input_names=["mel_context"],
        output_names=["viseme_logits"],
        dynamic_axes={"mel_context": {0: "batch"}, "viseme_logits": {0: "batch"}},
        opset_version=17,
    )
    meta = dict(meta)
    meta["onnx"] = path.name
    meta.setdefault("doc", meta.get("note", "Causal Mel-to-viseme MLP."))
    meta.setdefault("normalization", "per_utterance_per_mel_mean_std")
    meta["latency"] = estimate_latency(
        meta["audio"], int(meta["context_frames"]), float(meta.get("lookahead_ms", 0.0))
    )
    embed_onnx(path, meta)
    contract = load_model_contract(path, require_schema=2)
    print(f"ONNX_OK {path} schema={contract['_schema']} metadata=embedded", flush=True)


def plot_convergence(csv_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    t = [float(r["elapsed_s"]) / 60.0 for r in rows]
    loss = [float(r["loss"]) for r in rows]
    vacc = [float(r["val_acc"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(t, loss, color="#1f4e79", label="train loss")
    ax1.set_xlabel("wall minutes")
    ax1.set_ylabel("loss", color="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(t, vacc, color="#c45c26", label="val acc")
    ax2.set_ylabel("val accuracy", color="#c45c26")
    ax1.set_title("Tier-B Libri viseme train (soft boundaries + lookahead)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"PLOT_OK {out_png}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--context", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--wall-seconds", type=float, default=3600.0)
    ap.add_argument("--ckpt-minutes", type=str, default="10,20", help="comma minutes")
    ap.add_argument("--lookahead-ms", type=float, default=50.0)
    ap.add_argument("--blend-ms", type=float, default=60.0, help="soft boundary half-window*2 total")
    ap.add_argument("--soft-boundaries", action="store_true", default=True)
    ap.add_argument("--hard-boundaries", action="store_true", help="disable soft labels")
    ap.add_argument("--limit-utterances", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "export" / "tier-b",
    )
    args = ap.parse_args()
    soft = not args.hard_boundaries

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    tdir = ROOT / "data" / "tensors" / args.subset
    index_path = tdir / "index.json"
    if not index_path.exists():
        print(f"Missing {index_path}; run build_train_tensors.py --subset {args.subset}", flush=True)
        return 1
    index = json.loads(index_path.read_text())
    n_visemes = len(index["visemes"])
    hop_ms = 1000.0 * index["audio"]["hop_length_samples"] / index["audio"]["sample_rate"]
    lag_frames = max(0, int(round(args.lookahead_ms / hop_ms)))
    blend_frames = max(0, int(round(args.blend_ms / hop_ms)))
    utterances = index["utterances"]
    if args.limit_utterances:
        utterances = utterances[: args.limit_utterances]

    # deterministic split by utterance id
    ids = sorted(u["id"] for u in utterances)
    n_val_u = max(1, int(len(ids) * args.val_frac))
    val_ids = set(ids[-n_val_u:])
    train_u = [u for u in utterances if u["id"] not in val_ids]
    val_u = [u for u in utterances if u["id"] in val_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = args.context * int(index["audio"]["n_mels"])
    model = nn.Sequential(
        nn.Linear(in_features, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, n_visemes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "convergence.csv"
    csv_f = csv_path.open("w", newline="")
    writer = csv.DictWriter(
        csv_f, fieldnames=["elapsed_s", "step", "loss", "val_acc", "train_acc_approx"]
    )
    writer.writeheader()

    meta_base = {
        "model": "viseme_tier_b_mlp",
        "subset": args.subset,
        "context_frames": args.context,
        "n_mels": int(index["audio"]["n_mels"]),
        "input_features": in_features,
        "hidden": args.hidden,
        "n_visemes": n_visemes,
        "visemes": index["visemes"],
        "audio": index["audio"],
        "lookahead_ms": args.lookahead_ms,
        "lookahead_frames": lag_frames,
        "soft_boundaries": soft,
        "blend_ms": args.blend_ms if soft else 0,
        "blend_frames": blend_frames if soft else 0,
        "note": "Soft targets + lookahead lag for causal mel; add MelFrontend "
        "get_dsp_latency_seconds() in playback compensation.",
    }

    ckpt_secs = sorted({float(x) * 60.0 for x in args.ckpt_minutes.split(",") if x.strip()})
    ckpt_done: set[float] = set()

    def eval_acc(utts: list[dict], max_windows: int = 20000) -> float:
        model.eval()
        correct = 0
        total = 0
        xs, ys = [], []
        for x, tsoft in iter_utterance_windows(
            tdir, utts, args.context, n_visemes, lag_frames, blend_frames, soft
        ):
            xs.append(x)
            ys.append(int(np.argmax(tsoft)))
            if len(xs) >= 512:
                xb = torch.from_numpy(np.stack(xs)).to(device)
                with torch.no_grad():
                    pred = model(xb).argmax(dim=-1).cpu().numpy()
                yb = np.asarray(ys)
                correct += int((pred == yb).sum())
                total += len(yb)
                xs, ys = [], []
                if total >= max_windows:
                    break
        if xs:
            xb = torch.from_numpy(np.stack(xs)).to(device)
            with torch.no_grad():
                pred = model(xb).argmax(dim=-1).cpu().numpy()
            yb = np.asarray(ys)
            correct += int((pred == yb).sum())
            total += len(yb)
        return float(correct) / float(max(1, total))

    t0 = time.time()
    step = 0
    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    print(
        f"TIER_B_START subset={args.subset} train_utt={len(train_u)} val_utt={len(val_u)} "
        f"hidden={args.hidden} lag_frames={lag_frames} blend_frames={blend_frames} "
        f"soft={soft} device={device} wall_s={args.wall_seconds}",
        flush=True,
    )

    while True:
        elapsed = time.time() - t0
        if elapsed >= args.wall_seconds:
            break
        for x, tsoft in iter_utterance_windows(
            tdir, train_u, args.context, n_visemes, lag_frames, blend_frames, soft
        ):
            buf_x.append(x)
            buf_y.append(tsoft)
            if len(buf_x) < args.batch_size:
                continue
            xb = torch.from_numpy(np.stack(buf_x)).to(device)
            yb = torch.from_numpy(np.stack(buf_y)).to(device)
            buf_x, buf_y = [], []
            model.train()
            opt.zero_grad()
            logits = model(xb)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(yb * logp).sum(dim=-1).mean()
            loss.backward()
            opt.step()
            step += 1
            elapsed = time.time() - t0

            if step == 1 or step % 20 == 0:
                vacc = eval_acc(val_u)
                # cheap train acc on this batch hard labels
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    hard = yb.argmax(dim=-1)
                    tr = float((pred == hard).float().mean().item())
                writer.writerow(
                    {
                        "elapsed_s": f"{elapsed:.1f}",
                        "step": step,
                        "loss": f"{loss.item():.6f}",
                        "val_acc": f"{vacc:.4f}",
                        "train_acc_approx": f"{tr:.4f}",
                    }
                )
                csv_f.flush()
                print(
                    f"step {step:5d}  t={elapsed:6.1f}s  loss={loss.item():.4f}  "
                    f"val_acc={vacc:.3f}  batch_acc={tr:.3f}",
                    flush=True,
                )

            for cs in ckpt_secs:
                if cs not in ckpt_done and elapsed >= cs:
                    tag = f"{int(cs / 60)}m"
                    export_onnx(
                        model,
                        out_dir / f"model_{tag}.onnx",
                        {**meta_base, "checkpoint": tag, "elapsed_s": elapsed, "step": step},
                        in_features,
                    )
                    model.to(device)
                    ckpt_done.add(cs)

            if elapsed >= args.wall_seconds:
                break
        else:
            # finished one epoch over utterances; continue until wall
            continue
        break

    elapsed = time.time() - t0
    vacc = eval_acc(val_u, max_windows=50000)
    export_onnx(
        model,
        out_dir / "model_final.onnx",
        {
            **meta_base,
            "checkpoint": "final",
            "elapsed_s": elapsed,
            "step": step,
            "best_val_acc": vacc,
            "train_utterances": len(train_u),
            "val_utterances": len(val_u),
        },
        in_features,
    )
    csv_f.close()
    plot_convergence(csv_path, out_dir / "convergence.png")
    print(
        f"TIER_B_OK elapsed_s={elapsed:.1f} steps={step} final_val_acc={vacc:.3f} out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
