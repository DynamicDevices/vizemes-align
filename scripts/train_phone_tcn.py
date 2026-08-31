#!/usr/bin/env python3
"""Train/export a causal MFA-supervised phone TCN for the two-stage baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from embed_model_metadata import embed_onnx  # noqa: E402
from model_contract import load_model_contract  # noqa: E402
from phone_metrics import boundary_metrics, collapse_phone_frames, edit_distance  # noqa: E402


def make_model(n_mels: int, n_phones: int, channels: int, layers: int, kernel: int):
    import torch.nn as nn

    class CausalBlock(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
            self.norm = nn.GroupNorm(1, channels)
            self.act = nn.ReLU()

        def forward(self, x):
            import torch.nn.functional as functional

            return x + self.act(self.norm(self.conv(functional.pad(x, (self.pad, 0)))))

    class PhoneTcn(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Conv1d(n_mels, channels, 1)
            self.blocks = nn.ModuleList(CausalBlock(2**index) for index in range(layers))
            self.output = nn.Linear(channels, n_phones)

        def forward(self, mel_btf):
            hidden = self.input(mel_btf.transpose(1, 2))
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(hidden.transpose(1, 2))

    return PhoneTcn()


def load_utterance(index_dir: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / row["mel_path"]) as mel:
        features = mel["X"].astype(np.float32)
    with np.load(index_dir / row["phone_path"]) as phones:
        targets = phones["y_phone"].astype(np.int64)
    length = min(features.shape[0], targets.shape[0])
    return features[:length], targets[:length]


def evaluate(model, rows: list[dict], index_dir: Path, labels: list[str], device: str) -> dict:
    import torch

    model.eval()
    correct = frames = edits = reference_phones = 0
    boundary_errors: list[float] = []
    with torch.no_grad():
        for row in rows:
            features, reference = load_utterance(index_dir, row)
            logits = model(torch.from_numpy(features[None]).to(device))
            hypothesis = logits.argmax(dim=-1).cpu().numpy()[0]
            correct += int((hypothesis == reference).sum())
            frames += int(reference.size)
            ref_seq = collapse_phone_frames(reference, labels)
            hyp_seq = collapse_phone_frames(hypothesis, labels)
            edits += edit_distance(ref_seq, hyp_seq)
            reference_phones += len(ref_seq)
            boundaries = boundary_metrics(reference, hypothesis)
            if np.isfinite(boundaries.mean_absolute_ms):
                boundary_errors.append(boundaries.mean_absolute_ms)
    return {
        "frame_accuracy": correct / max(1, frames),
        "phone_error_rate": edits / max(1, reference_phones),
        "boundary_mean_absolute_ms": float(np.mean(boundary_errors)) if boundary_errors else None,
        "frames": frames,
        "utterances": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--output", type=Path, default=ROOT / "export" / "tier-c" / "phone.onnx")
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--limit-utterances", type=int, default=0)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    index_dir = ROOT / "data" / "phone-tensors" / args.subset
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    rows = sorted(index["utterances"], key=lambda row: row["id"])
    if args.limit_utterances:
        rows = rows[: args.limit_utterances]
    if len(rows) < 2:
        raise RuntimeError("need at least two phone-target utterances")
    split = max(1, int(round(len(rows) * 0.1)))
    train_rows, val_rows = rows[:-split], rows[-split:]
    labels = [name for name, _ in sorted(index["phones"].items(), key=lambda item: item[1])]
    n_mels = int(index["audio"]["n_mels"])
    model = make_model(n_mels, len(labels), args.channels, args.layers, args.kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started = time.monotonic()
    steps = 0
    model.train()
    while time.monotonic() - started < args.wall_seconds or steps == 0:
        features, targets = load_utterance(index_dir, random.choice(train_rows))
        x = torch.from_numpy(features[None]).to(device)
        y = torch.from_numpy(targets).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)[0]
        loss = functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 25 == 0:
            print(f"step={steps} elapsed={time.monotonic()-started:.1f}s loss={float(loss):.4f}")

    metrics = evaluate(model, val_rows, index_dir, labels, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros(1, 32, n_mels, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(args.output),
        input_names=["mel_btf"],
        output_names=["phone_logits"],
        dynamic_axes={"mel_btf": {0: "batch", 1: "time"}, "phone_logits": {0: "batch", 1: "time"}},
        opset_version=17,
    )
    phone_vocab = {label: index for index, label in enumerate(labels)}
    meta = {
        "model": "phone_tcn",
        "doc": "Causal MFA-supervised stressed-ARPA phone posterior model for two-stage visemes.",
        "subset": args.subset,
        "checkpoint": "wall-clock-final",
        "step": steps,
        "elapsed_s": round(time.monotonic() - started, 3),
        "train_utterances": len(train_rows),
        "val_utterances": len(val_rows),
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "phones": phone_vocab,
        "n_phones": len(labels),
        "n_mels": n_mels,
        "context_frames": 1,
        "input_features": n_mels,
        "normalization": "per_utterance_per_mel_mean_std",
        "lookahead_ms": 0.0,
        "audio": index["audio"],
        "quality": metrics,
        "architecture": {
            "type": "causal_tcn",
            "channels": args.channels,
            "layers": args.layers,
            "kernel": args.kernel,
        },
    }
    embed_onnx(args.output, meta)
    contract = load_model_contract(args.output, require_schema=2)
    assert contract["model"] == "phone_tcn"
    print(f"PHONE_TCN_ONNX_OK {args.output}")
    print("PHONE_TCN_METRICS " + json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
