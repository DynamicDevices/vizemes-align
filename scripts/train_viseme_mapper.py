#!/usr/bin/env python3
"""Train/export a small causal phone-posterior to viseme ONNX mapper."""
from __future__ import annotations

import argparse
import hashlib
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
from phone_viseme_mapper import PhoneVisemeMap  # noqa: E402


def make_model(n_phones: int, n_visemes: int, channels: int, layers: int, kernel: int):
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

    class VisemeMapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Conv1d(n_phones, channels, 1)
            self.blocks = nn.ModuleList(CausalBlock(2**index) for index in range(layers))
            self.output = nn.Linear(channels, n_visemes)

        def forward(self, phone_posterior_btp):
            hidden = self.input(phone_posterior_btp.transpose(1, 2))
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(hidden.transpose(1, 2))

    return VisemeMapper()


def load_phone_targets(index_dir: Path, row: dict) -> np.ndarray:
    with np.load(index_dir / row["phone_path"]) as phones:
        return phones["y_phone"].astype(np.int64)


def load_mel(row: dict) -> np.ndarray:
    with np.load(ROOT / row["mel_path"]) as mel:
        return mel["X"].astype(np.float32)


def infer_posteriors(session, row: dict, reference: np.ndarray) -> np.ndarray:
    features = load_mel(row)
    length = min(features.shape[0], reference.shape[0])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: features[None, :length]})[0][0]
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits).astype(np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def aggregate_visual(rows: list[tuple[np.ndarray, np.ndarray]], hop_ms: float) -> dict:
    frames = correct = ref_changes = hyp_changes = 0
    boundary_errors: list[float] = []
    for reference, hypothesis in rows:
        frames += int(reference.size)
        correct += int((reference == hypothesis).sum())
        ref_boundaries = np.flatnonzero(reference[1:] != reference[:-1]) + 1
        hyp_boundaries = np.flatnonzero(hypothesis[1:] != hypothesis[:-1]) + 1
        ref_changes += int(ref_boundaries.size)
        hyp_changes += int(hyp_boundaries.size)
        matched = min(ref_boundaries.size, hyp_boundaries.size)
        if matched:
            boundary_errors.extend(
                (np.abs(ref_boundaries[:matched] - hyp_boundaries[:matched]) * hop_ms).tolist()
            )
    duration_s = frames * hop_ms / 1000.0
    return {
        "frame_accuracy": correct / max(1, frames),
        "boundary_mean_absolute_ms": float(np.mean(boundary_errors)) if boundary_errors else None,
        "boundary_p95_absolute_ms": float(np.percentile(boundary_errors, 95)) if boundary_errors else None,
        "reference_transitions": ref_changes,
        "hypothesis_transitions": hyp_changes,
        "transition_ratio": hyp_changes / max(1, ref_changes),
        "excess_transition_jitter_hz": max(
            0.0, (hyp_changes - ref_changes) / max(0.01, duration_s)
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-onnx", type=Path, required=True)
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--output", type=Path, default=ROOT / "export" / "tier-c" / "viseme_map.onnx")
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--limit-utterances", type=int, default=0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import onnxruntime as ort
    import torch
    import torch.nn.functional as functional

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    phone_contract = load_model_contract(args.phone_onnx, require_schema=2)
    index_dir = ROOT / "data" / "phone-tensors" / args.subset
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    rows = sorted(index["utterances"], key=lambda row: row["id"])
    if args.limit_utterances:
        rows = rows[: args.limit_utterances]
    if len(rows) < 4:
        raise RuntimeError("need at least four phone-target utterances")
    split = max(1, int(round(len(rows) * 0.1)))
    train_rows, val_rows = rows[:-split], rows[-split:]
    phone_labels = [name for name, _ in sorted(index["phones"].items(), key=lambda item: item[1])]
    mapper = PhoneVisemeMap.from_config(ROOT / "configs" / "viseme_map_en_us_arpa.json", phone_labels)
    if len(phone_labels) != int(phone_contract.get("n_phones", len(phone_labels))):
        raise RuntimeError("phone model and target inventory disagree")

    session = ort.InferenceSession(str(args.phone_onnx), providers=["CPUExecutionProvider"])
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        reference_phone = load_phone_targets(index_dir, row)
        posterior = infer_posteriors(session, row, reference_phone)
        length = min(posterior.shape[0], reference_phone.shape[0])
        reference_viseme = mapper.matrix[reference_phone[:length]].argmax(axis=1).astype(np.int64)
        cache[row["id"]] = (posterior[:length], reference_viseme)

    model = make_model(
        len(phone_labels), len(mapper.viseme_labels), args.channels, args.layers, args.kernel
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    started = time.monotonic()
    steps = 0
    model.train()
    while time.monotonic() - started < args.wall_seconds or steps == 0:
        posterior, target = cache[random.choice(train_rows)["id"]]
        x = torch.from_numpy(posterior[None]).to(device)
        y = torch.from_numpy(target).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)[0]
        loss = functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 25 == 0:
            print(f"step={steps} elapsed={time.monotonic()-started:.1f}s loss={float(loss):.4f}")

    hop_ms = 1000.0 * index["audio"]["hop_length_samples"] / index["audio"]["sample_rate"]
    deterministic_rows: list[tuple[np.ndarray, np.ndarray]] = []
    learned_rows: list[tuple[np.ndarray, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for row in val_rows:
            posterior, reference = cache[row["id"]]
            deterministic = mapper.map_posteriors(
                posterior, hop_seconds=hop_ms / 1000.0, smoothing_seconds=0.06
            ).argmax(axis=1)
            learned = model(torch.from_numpy(posterior[None]).to(device))[0].argmax(dim=-1).cpu().numpy()
            deterministic_rows.append((reference, deterministic))
            learned_rows.append((reference, learned))
    metrics = {
        "deterministic": aggregate_visual(deterministic_rows, hop_ms),
        "learned": aggregate_visual(learned_rows, hop_ms),
        "validation_utterances": len(val_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros(1, 32, len(phone_labels), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(args.output),
        input_names=["phone_posterior_btp"],
        output_names=["viseme_logits"],
        dynamic_axes={
            "phone_posterior_btp": {0: "batch", 1: "time"},
            "viseme_logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    receptive_frames = 1 + (args.kernel - 1) * sum(2**index for index in range(args.layers))
    viseme_vocab = {label: index for index, label in enumerate(mapper.viseme_labels)}
    metadata = {
        "model": "causal_phone_viseme_mapper",
        "doc": "Small causal learned mapper compared with deterministic posterior summing and smoothing.",
        "subset": args.subset,
        "checkpoint": "wall-clock-final",
        "step": steps,
        "elapsed_s": round(time.monotonic() - started, 3),
        "train_utterances": len(train_rows),
        "val_utterances": len(val_rows),
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "input_domain": "phone_probability_simplex",
        "normalization": "sum_to_one_per_frame",
        "phones": index["phones"],
        "n_phones": len(phone_labels),
        "visemes": viseme_vocab,
        "n_visemes": len(viseme_vocab),
        "input_features": len(phone_labels),
        "context_frames": receptive_frames,
        "history_hops": receptive_frames,
        "lookahead_ms": 0.0,
        "audio": index["audio"],
        "upstream": {
            "model": phone_contract["model"],
            "onnx_sha256": sha256(args.phone_onnx),
            "output": "phone_logits converted to probability simplex",
        },
        "architecture": {
            "type": "causal_tcn",
            "channels": args.channels,
            "layers": args.layers,
            "kernel": args.kernel,
            "receptive_frames": receptive_frames,
            "receptive_ms": receptive_frames * hop_ms,
        },
        "quality": metrics,
    }
    embed_onnx(args.output, metadata)
    contract = load_model_contract(args.output, require_schema=2)
    assert contract["model"] == "causal_phone_viseme_mapper"
    assert contract["upstream"]["onnx_sha256"] == sha256(args.phone_onnx)
    print(f"VISEME_MAPPER_ONNX_OK {args.output}")
    print("VISEME_MAPPER_METRICS " + json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
