#!/usr/bin/env python3
"""Compare C mel (dump_mel) vs mel_features_c wrapper — must match exactly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mel_features_c import ensure_model_meta, mel_features_c  # noqa: E402


def run_dump_mel(model_onnx: Path, wav: Path, dump_bin: Path) -> np.ndarray:
    import subprocess

    meta = ensure_model_meta(model_onnx)
    proc = subprocess.run(
        [str(dump_bin), str(meta), str(wav)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.strip().splitlines()
    if not lines or not lines[0].startswith("frames "):
        raise RuntimeError(f"unexpected dump_mel header: {lines[:3]!r}")
    nframes = int(lines[0].split()[1])
    nmels = int(lines[0].split()[3])
    rows = [[float(x) for x in line.split()] for line in lines[1:] if line.strip()]
    X = np.asarray(rows, dtype=np.float32)
    if X.shape != (nframes, nmels):
        raise RuntimeError(f"C mel shape {X.shape} != ({nframes}, {nmels})")
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", nargs="?", type=Path, default=ROOT / "export/ci-smoke/ci-fixture.wav")
    ap.add_argument("--model-onnx", type=Path, default=ROOT / "export/ci-smoke/model.onnx")
    ap.add_argument("--dump-mel", type=Path, default=ROOT / "gdextension/build/dump_mel")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    if args.build or not args.dump_mel.is_file():
        import subprocess

        subprocess.run(["make", "-C", str(ROOT / "gdextension"), "dump-mel"], check=True)

    via_api = mel_features_c(args.wav, args.model_onnx)
    via_dump = run_dump_mel(args.model_onnx, args.wav, args.dump_mel)

    if via_api.shape != via_dump.shape:
        print(f"FAIL shape api={via_api.shape} dump={via_dump.shape}", file=sys.stderr)
        return 1

    diff = np.abs(via_api - via_dump)
    mx = float(diff.max())
    mean = float(diff.mean())
    print(f"MEL_PARITY frames={via_api.shape[0]} n_mels={via_api.shape[1]} max_abs_diff={mx:.6g} mean={mean:.6g} tol={args.tol}")
    if mx > args.tol:
        ti, mi = np.unravel_index(int(diff.argmax()), diff.shape)
        print(f"FAIL worst frame={ti} mel={mi}", file=sys.stderr)
        return 1
    print("MEL_PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
