#!/usr/bin/env python3
"""Bench MelFrontend (host C mel) + OnnxLoader (ORT) CPU/RSS on ci-smoke fixtures.

Prints machine-readable tokens:
  MEL_CPU_US=…  ORT_CPU_US=…  ORT_ITERS=…  RSS_KB=…  BENCH_OK

Usage (from repo root, with ORT available):
  ORT_ROOT=/path/to/ort python3 gdextension/tools/bench_plugins.py
"""
from __future__ import annotations

import os
import resource
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def rss_kb() -> int:
    # Linux: ru_maxrss is KB; macOS is bytes — report as KB best-effort.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def load_wav_mono_f32(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = [
        int.from_bytes(raw[i : i + 2], "little", signed=True) / 32768.0
        for i in range(0, len(raw), 2)
    ]
    return pcm, sr


def bench_mel_python(pcm: list[float], meta: dict, repeats: int = 5) -> float:
    """Approximate mel path cost via torchaudio if present, else skip."""
    try:
        import torch
        import torchaudio
    except ImportError:
        return -1.0
    audio = meta.get("audio", meta)
    sr = int(audio["sample_rate"])
    n_fft = int(audio["n_fft"])
    win = int(audio["window_length_samples"])
    hop = int(audio["hop_length_samples"])
    n_mels = int(audio["n_mels"])
    fmin = float(audio.get("fmin", 50.0))
    fmax = float(audio.get("fmax", 8000.0))
    wav = torch.tensor(pcm, dtype=torch.float32)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop,
        n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
    )
    # warmup
    _ = mel(wav)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = mel(wav)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / repeats


def bench_ort(model: Path, feats: int, iters: int = 500) -> float:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.randn(1, feats).astype(np.float32)
    # warmup
    for _ in range(10):
        sess.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {inp: x})
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters


def main() -> int:
    from model_contract import load_model_contract

    model_onnx = ROOT / "export/ci-smoke/model.onnx"
    wav_path = ROOT / "export/ci-smoke/ci-fixture.wav"
    if not model_onnx.is_file() or not wav_path.is_file():
        print("missing ci-smoke fixtures", file=sys.stderr)
        return 1

    meta = load_model_contract(model_onnx, require_schema=2)
    feats = int(meta.get("input_features", meta.get("context_frames", 20) * 80))
    pcm, _sr = load_wav_mono_f32(wav_path)

    mel_us = bench_mel_python(pcm, meta, repeats=5)
    ort_us = bench_ort(model_onnx, feats, iters=500)
    rss = rss_kb()

    if mel_us < 0:
        print("MEL_CPU_US=NA note=torchaudio_missing")
    else:
        print(f"MEL_CPU_US={mel_us:.1f}")
    print(f"ORT_CPU_US={ort_us:.1f}")
    print(f"ORT_ITERS=500")
    print(f"INPUT_FEATURES={feats}")
    print(f"RSS_KB={rss}")
    # Realtime budget @ 10 ms hop = 100 contexts/s → 10000 µs/context
    if ort_us > 0:
        print(f"ORT_RT_FACTOR={10000.0 / ort_us:.1f}x_vs_10ms_hop")
    print("BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
