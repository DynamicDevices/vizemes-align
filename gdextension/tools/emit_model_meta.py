#!/usr/bin/env python3
"""Emit flat model.meta from the canonical ONNX metadata contract.

Usage:
  python3 tools/emit_model_meta.py ../export/ci-smoke/model.onnx ../export/ci-smoke/model.meta
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import load_model_contract  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} model.onnx model.meta", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    data = load_model_contract(src, require_schema=2)
    audio = data.get("audio", {})
    ctx = int(data["context_frames"])
    n_mels = int(audio.get("n_mels", data.get("n_mels", 80)))
    feats = int(data.get("input_features", ctx * n_mels))
    visemes = data.get("visemes", {})
    n_vis = int(data.get("n_visemes", len(visemes)))
    lines = [
        f"# generated from canonical ONNX metadata in {src.name} — do not hand-edit",
        f"context_frames={ctx}",
        f"n_mels={n_mels}",
        f"input_features={feats}",
        f"n_visemes={n_vis}",
        f"sample_rate={int(audio.get('sample_rate', 16000))}",
        f"hop_length_samples={int(audio.get('hop_length_samples', 160))}",
        f"window_length_samples={int(audio.get('window_length_samples', 400))}",
        f"n_fft={int(audio.get('n_fft', 1024))}",
        f"fmin={float(audio.get('fmin', 50.0))}",
        f"fmax={float(audio.get('fmax', 8000.0))}",
    ]
    # id → name
    by_id: dict[int, str] = {}
    for name, idx in visemes.items():
        by_id[int(idx)] = str(name)
    for i in range(n_vis):
        lines.append(f"viseme_{i}={by_id.get(i, str(i))}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
