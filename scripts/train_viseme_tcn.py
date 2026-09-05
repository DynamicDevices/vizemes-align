#!/usr/bin/env python3
"""Train OpenLipSync-style TCN on current MFA→viseme soft labels (10‑min burn).

Same label scheme as train_viseme_tier_b.py (soft boundaries + lookahead).
Only the network changes — so we can A/B convergence vs the MLP without
confounding a phone→viseme pipeline change (Julian 894).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Reuse label helpers from tier-B
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from train_viseme_tier_b import (  # noqa: E402
    apply_lookahead,
    soft_boundary_targets,
)
from embed_model_metadata import embed_onnx, estimate_latency  # noqa: E402


def load_split(tdir: Path, val_frac: float, limit: int = 0, split_by_speaker: bool = False):
    index = json.loads((tdir / "index.json").read_text())
    utterances = index["utterances"]
    if limit:
        utterances = utterances[:limit]
    if split_by_speaker:
        speakers = sorted({u["id"].split("-")[0] for u in utterances})
        n_val = max(1, int(round(len(speakers) * val_frac)))
        val_speakers = set(speakers[-n_val:])
        train_u = [u for u in utterances if u["id"].split("-")[0] not in val_speakers]
        val_u = [u for u in utterances if u["id"].split("-")[0] in val_speakers]
    else:
        ids = sorted(u["id"] for u in utterances)
        n_val = max(1, int(len(ids) * val_frac))
        val_ids = set(ids[-n_val:])
        train_u = [u for u in utterances if u["id"] not in val_ids]
        val_u = [u for u in utterances if u["id"] in val_ids]
    return index, train_u, val_u


def split_digest(utterances: list[dict]) -> str:
    payload = "\n".join(sorted(str(u["id"]) for u in utterances)).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_dilations(value: str, layers: int) -> list[int]:
    if not value.strip():
        return [2**index for index in range(layers)]
    result = [int(part.strip()) for part in value.split(",")]
    if len(result) != layers or any(dilation < 1 for dilation in result):
        raise ValueError(f"--dilations must contain {layers} positive integers")
    return result


def receptive_history_hops(dilations: list[int], kernel: int) -> int:
    # Two causal convolutions per residual block.
    return 1 + 2 * (kernel - 1) * sum(dilations)


def utterance_xy(
    tdir: Path,
    u: dict,
    n_visemes: int,
    lag: int,
    blend: int,
    soft: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(tdir / u["path"])
    X = z["X"].astype(np.float32)  # (T, input_features)
    y = apply_lookahead(z["y"].astype(np.int64), lag)
    valid = z["valid"].astype(bool) if "valid" in z else np.ones(y.shape, dtype=bool)
    valid = apply_lookahead(valid, lag).astype(bool)
    if blend > 0 and not valid.all():
        invalid = np.convolve((~valid).astype(np.int8), np.ones(2 * blend + 1, dtype=np.int8), mode="same")
        valid = invalid == 0
    if soft:
        targets = soft_boundary_targets(y, n_visemes, blend)
    else:
        targets = np.eye(n_visemes, dtype=np.float32)[y]
    return X, targets, valid


def make_tcn(input_features: int, n_visemes: int, dilations: list[int], channels: int, kernel: int, dropout: float):
    import torch.nn as nn
    from torch.nn.utils import weight_norm

    class CausalBlock(nn.Module):
        def __init__(self, ch: int, k: int, dil: int, drop: float):
            super().__init__()
            self.k = k
            self.dil = dil
            self.c1 = weight_norm(nn.Conv1d(ch, ch, k, dilation=dil, bias=False))
            self.c2 = weight_norm(nn.Conv1d(ch, ch, k, dilation=dil, bias=False))
            self.drop = nn.Dropout(drop)
            self.act = nn.ReLU()

        def _causal(self, conv, x):
            pad = (self.k - 1) * self.dil
            x = nn.functional.pad(x, (pad, 0))
            return conv(x)

        def forward(self, x):
            y = self.drop(self.act(self._causal(self.c1, x)))
            y = self.drop(self.act(self._causal(self.c2, y)))
            return y + x

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = weight_norm(nn.Conv1d(input_features, channels, 1, bias=False))
            self.layers = nn.ModuleList(
                [CausalBlock(channels, kernel, dilation, dropout) for dilation in dilations]
            )
            self.out = nn.Linear(channels, n_visemes)
            nn.init.normal_(self.out.weight, std=0.01)
            nn.init.zeros_(self.out.bias)

        def forward(self, features_btf):
            # (B, T, F) → (B, T, C)
            x = features_btf.transpose(1, 2)
            h = self.proj(x)
            for layer in self.layers:
                h = layer(h)
            h = h.transpose(1, 2)
            return self.out(h)

    return TCN()


def export_onnx(model, path: Path, meta: dict, input_features: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros(1, 32, input_features, dtype=torch.float32)
    input_name = "mel_btf" if meta["audio"].get("frontend", "mel") == "mel" else "features_btf"
    torch.onnx.export(
        model_cpu,
        dummy,
        str(path),
        input_names=[input_name],
        output_names=["viseme_logits"],
        dynamic_axes={input_name: {0: "batch", 1: "time"}, "viseme_logits": {0: "batch", 1: "time"}},
        opset_version=17,
        # Keep the pinned Nix training shell self-contained. PyTorch 2.9's
        # default dynamo exporter adds an onnxscript dependency unavailable in
        # nixos-25.11, while the legacy exporter supports this graph fully.
        dynamo=False,
    )
    meta = dict(meta)
    meta["onnx"] = path.name
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ONNX_OK {path}", flush=True)


def plot_convergence(csv_path: Path, out_png: Path, title: str) -> None:
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
    ax1.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"PLOT_OK {out_png}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--tensor-dir", type=Path, help="Override data/tensors/<subset> for an isolated run.")
    ap.add_argument(
        "--overfit-stem",
        default="",
        help="Train and score on exactly this utterance; validates model capacity, not generalisation.",
    )
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument(
        "--dilations", default="",
        help="Comma-separated dilation per layer; default remains exponential.",
    )
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-utterances", type=int, default=8)
    ap.add_argument("--wall-seconds", type=float, default=600.0)
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Exact optimizer-step budget; when non-zero, overrides --wall-seconds.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--lookahead-ms", type=float, default=50.0)
    ap.add_argument("--blend-ms", type=float, default=60.0)
    ap.add_argument("--hard-boundaries", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-by-speaker", action="store_true", help="Hold out complete speakers.")
    ap.add_argument("--limit-utterances", type=int, default=0)
    ap.add_argument("--log-every-s", type=float, default=30.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "export" / "tier-b-tcn")
    args = ap.parse_args()
    soft = not args.hard_boundaries

    import torch
    import torch.nn.functional as F

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    tdir = args.tensor_dir or ROOT / "data" / "tensors" / args.subset
    index, train_u, val_u = load_split(
        tdir, args.val_frac, args.limit_utterances, args.split_by_speaker
    )
    if args.overfit_stem:
        selected = next(
            (utterance for utterance in index["utterances"] if utterance["id"] == args.overfit_stem),
            None,
        )
        if selected is None:
            raise ValueError(f"overfit stem not found in {tdir / 'index.json'}: {args.overfit_stem}")
        train_u = [selected]
        val_u = [selected]
    n_visemes = len(index["visemes"])
    input_features = int(index["audio"].get("input_features") or index["audio"]["n_mels"])
    hop_ms = 1000.0 * index["audio"]["hop_length_samples"] / index["audio"]["sample_rate"]
    lag = max(0, int(round(args.lookahead_ms / hop_ms)))
    blend = max(0, int(round(args.blend_ms / hop_ms)))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)
    dilations = parse_dilations(args.dilations, args.layers)
    model = make_tcn(input_features, n_visemes, dilations, args.channels, args.kernel, args.dropout).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history_hops = receptive_history_hops(dilations, args.kernel)
    print(
        f"TCN layers={args.layers} channels={args.channels} params={n_params} device={device} "
        f"history={history_hops}hops/{history_hops * hop_ms:.0f}ms train_u={len(train_u)} "
        f"val_u={len(val_u)} seed={args.seed}",
        flush=True,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "convergence.csv"
    csv_f = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=["elapsed_s", "step", "loss", "val_acc"])
    writer.writeheader()

    meta_base = {
        "model": "viseme_tcn",
        "subset": args.subset,
        "n_mels": index["audio"].get("n_mels"),
        "n_visemes": n_visemes,
        "visemes": index["visemes"],
        "audio": index["audio"],
        "layers": args.layers,
        "dilations": dilations,
        "channels": args.channels,
        "kernel_size": args.kernel,
        "dropout": args.dropout,
        "lookahead_ms": args.lookahead_ms,
        "lookahead_frames": lag,
        "soft_boundaries": soft,
        "blend_ms": args.blend_ms,
        "blend_frames": blend,
        "note": "OpenLipSync-style TCN; same viseme soft labels as tier-B MLP (no phone head).",
        "doc": (
            "Causal direct-viseme TCN. When overfit_stem is set, this is a one-clip "
            "capacity check and its fit score must not be interpreted as generalisation."
        ),
        "normalization": "per_utterance_per_feature_mean_std",
        "context_frames": 1,
        "input_features": input_features,
        "train_utterances": len(train_u),
        "val_utterances": len(val_u),
        "params": n_params,
        "seed": args.seed,
        "device": str(device),
        "step_budget": args.steps,
        "receptive_history_hops": history_hops,
        "receptive_history_ms": history_hops * hop_ms,
        "train_split_sha256": split_digest(train_u),
        "val_split_sha256": split_digest(val_u),
        "split_unit": "speaker" if args.split_by_speaker else "utterance",
    }
    if args.overfit_stem:
        meta_base["training_mode"] = "single_utterance_overfit"
        meta_base["overfit_stem"] = args.overfit_stem

    def eval_split(utterances: list[dict]) -> tuple[float, int]:
        model.eval()
        hits = tot = 0
        with torch.no_grad():
            for u in utterances:
                X, targets, valid = utterance_xy(tdir, u, n_visemes, lag, blend, soft)
                xb = torch.from_numpy(X[None, ...]).to(device)
                logits = model(xb)[0].cpu().numpy()
                pred = logits.argmax(-1)
                hard = targets.argmax(-1)
                hits += int(((pred == hard) & valid).sum())
                tot += int(valid.sum())
        model.train()
        return hits / max(1, tot), tot

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    step = 0
    last_log = t0
    best_val = -1.0
    model.train()
    while (step < args.steps) if args.steps > 0 else (time.time() - t0 < args.wall_seconds):
        batch_idx = rng.choice(len(train_u), size=min(args.batch_utterances, len(train_u)), replace=False)
        loss_acc = 0.0
        n_b = 0
        opt.zero_grad(set_to_none=True)
        for bi in batch_idx:
            X, targets, valid = utterance_xy(tdir, train_u[int(bi)], n_visemes, lag, blend, soft)
            xb = torch.from_numpy(X[None, ...]).to(device)
            yb = torch.from_numpy(targets[None, ...]).to(device)
            logits = model(xb)
            frame_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none").mean(dim=-1)
            mask = torch.from_numpy(valid[None, ...]).to(device=device, dtype=frame_loss.dtype)
            loss = (frame_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            loss_acc += float(loss.item())
            n_b += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        now = time.time()
        if now - last_log >= args.log_every_s or step == 1:
            vacc, _ = eval_split(val_u)
            best_val = max(best_val, vacc)
            row = {
                "elapsed_s": round(now - t0, 3),
                "step": step,
                "loss": round(loss_acc / max(1, n_b), 6),
                "val_acc": round(vacc, 6),
            }
            writer.writerow(row)
            csv_f.flush()
            print(
                f"step={step} t={row['elapsed_s']:.0f}s loss={row['loss']:.4f} val_acc={vacc:.4f}",
                flush=True,
            )
            last_log = now

    csv_f.close()
    fit_acc, fit_frames = eval_split(train_u)
    held_out_acc, held_out_frames = eval_split(val_u)
    meta = dict(meta_base)
    meta.update(
        {
            "checkpoint": "final",
            "elapsed_s": time.time() - t0,
            "step": step,
            "best_val_acc": best_val,
            "quality": {
                "metric": "argmax_frame_accuracy",
                "all_acc": held_out_acc,
                "n_frames": held_out_frames,
                "scope": "single-clip fit" if args.overfit_stem else "held-out validation",
            },
            "fit_quality": {
                "metric": "argmax_frame_accuracy",
                "all_acc": fit_acc,
                "n_frames": fit_frames,
                "scope": "training split",
            },
            "latency": estimate_latency(index["audio"], 1, args.lookahead_ms),
        }
    )
    model_path = out_dir / "model_final.onnx"
    export_onnx(model, model_path, meta, input_features)
    embed_onnx(model_path, meta)
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot_convergence(
        csv_path,
        out_dir / "convergence.png",
        "Tier-B labels + TCN (OpenLipSync-style) — 10 min",
    )
    print(
        f"DONE best_val_acc={best_val:.4f} fit_acc={fit_acc:.4f} "
        f"held_out_acc={held_out_acc:.4f} out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
