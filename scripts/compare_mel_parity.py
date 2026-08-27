#!/usr/bin/env python3
"""Compare C mel (dump_mel) vs torchaudio on a mono wav — parity gate for MelFrontend.

Usage:
  python3 scripts/compare_mel_parity.py export/ci-smoke/ci-fixture.wav
  python3 scripts/compare_mel_parity.py --wav ... --model-json export/ci-smoke/model.json

Exit 0 when max abs diff <= --tol (default 1e-3).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_train_tensors import AUDIO, mel_features  # noqa: E402


def run_c_mel(model_json: Path, wav: Path, dump_bin: Path) -> np.ndarray:
    proc = subprocess.run(
        [str(dump_bin), str(model_json), str(wav)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.strip().splitlines()
    if not lines or not lines[0].startswith("frames "):
        raise RuntimeError(f"unexpected dump_mel header: {lines[:3]!r}")
    nframes = int(lines[0].split()[1])
    nmels = int(lines[0].split()[3])
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        rows.append([float(x) for x in line.split()])
    X = np.asarray(rows, dtype=np.float32)
    if X.shape != (nframes, nmels):
        raise RuntimeError(f"C mel shape {X.shape} != ({nframes}, {nmels})")
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", nargs="?", type=Path, default=ROOT / "export/ci-smoke/ci-fixture.wav")
    ap.add_argument("--model-json", type=Path, default=ROOT / "export/ci-smoke/model.json")
    ap.add_argument("--dump-mel", type=Path, default=ROOT / "gdextension/build/dump_mel")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs diff pass threshold")
    ap.add_argument("--build", action="store_true", help="run make dump-mel first")
    args = ap.parse_args()

    if args.build or not args.dump_mel.is_file():
        subprocess.run(["make", "-C", str(ROOT / "gdextension"), "dump-mel"], check=True)

    meta = json.loads(args.model_json.read_text(encoding="utf-8"))
    py = mel_features(args.wav)
    c = run_c_mel(args.model_json, args.wav, args.dump_mel)

    n = min(py.shape[0], c.shape[0])
    if n == 0:
        print("FAIL: zero frames", file=sys.stderr)
        return 1
    if py.shape[0] != c.shape[0]:
        print(f"WARN frame count py={py.shape[0]} c={c.shape[0]} (compare first {n})")

    diff = np.abs(py[:n] - c[:n])
    mx = float(diff.max())
    mean = float(diff.mean())
    print(f"MEL_PARITY frames={n} n_mels={meta['n_mels']} max_abs_diff={mx:.6g} mean_abs_diff={mean:.6g} tol={args.tol}")
    if mx > args.tol:
        ti, mi = np.unravel_index(int(diff.argmax()), diff.shape)
        print(
            f"FAIL worst frame={ti} mel={mi} py={py[ti, mi]:.6g} c={c[ti, mi]:.6g}",
            file=sys.stderr,
        )
        return 1
    print("MEL_PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
