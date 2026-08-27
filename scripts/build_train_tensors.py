#!/usr/bin/env python3
"""Build mel feature tensors + viseme frame labels (OpenLipSync audio recipe).

Stage: feature extraction / dataset build for classifier training.
Reads data/{prepared,aligned}/<subset>, writes data/tensors/<subset>/…
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_godot_package import load_viseme_map, phones_from_textgrid  # noqa: E402
from mel_features_c import mel_features_c  # noqa: E402

# Match OpenLipSync training/recipes/tcn_config.toml defaults (also in model.json sidecar).
AUDIO = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "window_length_samples": 400,  # 25 ms
    "hop_length_samples": 160,  # 10 ms → 100 fps
    "n_mels": 80,
    "fmin": 50.0,
    "fmax": 8000.0,
}


def mel_features(wav_path: Path) -> np.ndarray:
    """Return (T, n_mels) float32 log-mel from host C (runtime SoT)."""
    model_json = ROOT / "export/ci-smoke/model.json"
    return mel_features_c(wav_path, model_json)


def frame_viseme_ids(
    phones: list[dict],
    n_frames: int,
    hop_s: float,
    phone_to_idx: dict[str, int],
) -> np.ndarray:
    """Hard label per mel frame from TextGrid phone intervals."""
    y = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        t = (i + 0.5) * hop_s
        lab = 0
        for p in phones:
            if p["start"] <= t < p["end"] or (i == n_frames - 1 and p["start"] <= t <= p["end"]):
                key = p["phone"]
                base = "".join(c for c in key if not c.isdigit())
                lab = phone_to_idx.get(
                    key,
                    phone_to_idx.get(base, phone_to_idx.get(base.upper(), 0)),
                )
                break
        y[i] = lab
    return y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="ci-fixture")
    ap.add_argument(
        "--viseme-map",
        type=Path,
        default=ROOT / "configs" / "viseme_map_en_us_arpa.json",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prepared = ROOT / "data" / "prepared" / args.subset
    aligned = ROOT / "data" / "aligned" / args.subset
    out_dir = ROOT / "data" / "tensors" / args.subset
    if not aligned.exists():
        print(f"Missing alignments: {aligned}", file=sys.stderr)
        return 1

    name_to_idx, phone_to_idx = load_viseme_map(args.viseme_map)
    hop_s = AUDIO["hop_length_samples"] / AUDIO["sample_rate"]
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = sorted(aligned.rglob("*.TextGrid"))
    if args.limit:
        grids = grids[: args.limit]

    index = {
        "subset": args.subset,
        "audio": AUDIO,
        "viseme_map": args.viseme_map.name,
        "visemes": name_to_idx,
        "fps": 1.0 / hop_s,
        "utterances": [],
    }

    for tg_path in grids:
        stem = tg_path.stem
        wav = tg_path.with_suffix(".wav")
        if not wav.exists():
            cands = list(prepared.rglob(f"{stem}.wav"))
            wav = cands[0] if cands else None
        if wav is None or not wav.exists():
            print(f"skip {stem}: no wav", file=sys.stderr)
            continue

        phones = phones_from_textgrid(tg_path)
        X = mel_features(wav)
        # per-utterance normalize (OpenLipSync default)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-5
        Xn = (X - mu) / sd
        y = frame_viseme_ids(phones, Xn.shape[0], hop_s, phone_to_idx)

        npz = out_dir / f"{stem}.npz"
        np.savez_compressed(npz, X=Xn, y=y, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
        index["utterances"].append(
            {
                "id": stem,
                "path": npz.name,
                "frames": int(Xn.shape[0]),
                "n_mels": int(Xn.shape[1]),
            }
        )
        print(f"{stem}: X{Xn.shape} y{y.shape} → {npz}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(index['utterances'])} utterances → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
