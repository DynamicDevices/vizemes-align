#!/usr/bin/env python3
"""Evaluate direct-viseme and phone→deterministic paths on identical frames."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import load_model_contract  # noqa: E402
from train_viseme_mapper import aggregate_visual  # noqa: E402


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(values).astype(np.float32)
    return values / values.sum(axis=1, keepdims=True)


def map_phones(posteriors: np.ndarray, contract: dict) -> np.ndarray:
    mapping = contract["phone_to_viseme"]
    ids = np.asarray(mapping["phone_to_viseme_ids"], dtype=np.int64)
    n_visemes = len(mapping["visemes"])
    matrix = np.eye(n_visemes, dtype=np.float32)[ids]
    mapped = posteriors @ matrix
    hop = float(mapping["hop_seconds"])
    smoothing = float(mapping["smoothing_seconds"])
    alpha = 1.0 if smoothing == 0 else 1.0 - np.exp(-hop / smoothing)
    state = np.zeros(n_visemes, dtype=np.float32)
    out = np.zeros_like(mapped)
    top_k = int(mapping["top_k"])
    for frame_index, frame in enumerate(mapped):
        state += alpha * (frame - state)
        keep = np.argpartition(state, -top_k)[-top_k:] if 0 < top_k < n_visemes else None
        sparse = np.zeros_like(state) if keep is not None else state.copy()
        if keep is not None:
            sparse[keep] = state[keep]
        sparse /= max(float(sparse.sum()), 1e-12)
        out[frame_index] = sparse
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-onnx", type=Path, required=True)
    parser.add_argument("--phone-onnx", type=Path, required=True)
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    direct_contract = load_model_contract(args.direct_onnx, require_schema=2)
    phone_contract = load_model_contract(args.phone_onnx, require_schema=2)
    mapper = phone_contract.get("phone_to_viseme")
    if not isinstance(mapper, dict):
        raise RuntimeError("phone ONNX has no embedded deterministic mapper")
    index_dir = ROOT / "data" / "phone-tensors" / args.subset
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    rows = sorted(index["utterances"], key=lambda row: row["id"])
    split = max(1, int(round(len(rows) * 0.1)))
    rows = rows[-split:]

    direct_session = ort.InferenceSession(str(args.direct_onnx), providers=["CPUExecutionProvider"])
    phone_session = ort.InferenceSession(str(args.phone_onnx), providers=["CPUExecutionProvider"])
    direct_name = direct_session.get_inputs()[0].name
    phone_name = phone_session.get_inputs()[0].name
    phone_to_viseme = np.asarray(mapper["phone_to_viseme_ids"], dtype=np.int64)
    direct_rows: list[tuple[np.ndarray, np.ndarray]] = []
    phone_rows: list[tuple[np.ndarray, np.ndarray]] = []
    direct_inference_s = phone_inference_s = mapper_s = 0.0
    for row in rows:
        with np.load(ROOT / row["mel_path"]) as mel:
            features = mel["X"].astype(np.float32)
        with np.load(index_dir / row["phone_path"]) as phones:
            reference_phones = phones["y_phone"].astype(np.int64)
        length = min(features.shape[0], reference_phones.shape[0])
        features = features[:length]
        reference = phone_to_viseme[reference_phones[:length]]
        started = time.perf_counter()
        direct_logits = direct_session.run(None, {direct_name: features[None]})[0][0]
        direct_inference_s += time.perf_counter() - started
        started = time.perf_counter()
        phone_logits = phone_session.run(None, {phone_name: features[None]})[0][0]
        phone_inference_s += time.perf_counter() - started
        direct_rows.append((reference, direct_logits.argmax(axis=1)))
        started = time.perf_counter()
        mapped = map_phones(softmax(phone_logits), phone_contract)
        mapper_s += time.perf_counter() - started
        phone_rows.append((reference, mapped.argmax(axis=1)))

    hop_ms = 1000.0 * index["audio"]["hop_length_samples"] / index["audio"]["sample_rate"]
    frames = int(sum(reference.size for reference, _ in direct_rows))
    audio_s = frames * hop_ms / 1000.0
    report = {
        "protocol": {
            "subset": args.subset,
            "held_out_utterances": len(rows),
            "frames": frames,
            "hop_ms": hop_ms,
            "reference": "MFA phone intervals mapped through embedded deterministic table",
        },
        "direct_viseme_tcn": aggregate_visual(direct_rows, hop_ms),
        "phone_tcn_deterministic": aggregate_visual(phone_rows, hop_ms),
        "runtime": {
            "measurement": "single-process CPU wall time; ONNX sessions preloaded",
            "audio_seconds": audio_s,
            "direct_onnx_seconds": direct_inference_s,
            "direct_realtime_factor": direct_inference_s / audio_s,
            "phone_onnx_seconds": phone_inference_s,
            "deterministic_mapper_seconds": mapper_s,
            "phone_pipeline_seconds": phone_inference_s + mapper_s,
            "phone_pipeline_realtime_factor": (phone_inference_s + mapper_s) / audio_s,
            "algorithmic_lookahead_ms": {
                "direct": float(direct_contract.get("lookahead_ms", 0.0)),
                "phone_pipeline": float(phone_contract.get("lookahead_ms", 0.0)),
            },
        },
        "models": {
            "direct": str(args.direct_onnx),
            "phone": str(args.phone_onnx),
            "direct_training_seconds": direct_contract.get("elapsed_s"),
            "phone_training_seconds": phone_contract.get("elapsed_s"),
            "direct_params": direct_contract.get("params"),
            "phone_params": phone_contract.get("params"),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print("DIRECT_PHONE_AB_OK")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
