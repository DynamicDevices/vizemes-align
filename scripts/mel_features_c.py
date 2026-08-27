#!/usr/bin/env python3
"""Host C mel features (same path as MelFrontend batch / dump_mel)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUMP_MEL = ROOT / "gdextension/build/dump_mel"
DEFAULT_MODEL_JSON = ROOT / "export/ci-smoke/model.json"


def _ensure_dump_mel() -> Path:
    if not DUMP_MEL.is_file():
        subprocess.run(["make", "-C", str(ROOT / "gdextension"), "dump-mel"], check=True)
    return DUMP_MEL


def mel_features_c(wav_path: Path, model_json: Path | None = None) -> np.ndarray:
    """Return (T, n_mels) float32 log-mel from host C (torchaudio-compatible batch path)."""
    model_json = model_json or DEFAULT_MODEL_JSON
    dump = _ensure_dump_mel()
    proc = subprocess.run(
        [str(dump), str(model_json), str(wav_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.strip().splitlines()
    if not lines or not lines[0].startswith("frames "):
        raise RuntimeError(f"unexpected dump_mel output: {lines[:2]!r}")
    nframes = int(lines[0].split()[1])
    nmels = int(lines[0].split()[3])
    rows = []
    for line in lines[1:]:
        if line.strip():
            rows.append([float(x) for x in line.split()])
    X = np.asarray(rows, dtype=np.float32)
    if X.shape != (nframes, nmels):
        raise RuntimeError(f"C mel shape {X.shape} != ({nframes}, {nmels})")
    return X


if __name__ == "__main__":
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "export/ci-smoke/ci-fixture.wav"
    X = mel_features_c(wav)
    print(f"MEL_C_OK frames={X.shape[0]} n_mels={X.shape[1]} absmax={float(np.abs(X).max()):.4f}")
