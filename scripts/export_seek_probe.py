#!/usr/bin/env python3
"""Export seek_probe.json for Godot editor side-by-side checks.

Picks one aligned utterance (default: ci-fixture / first in subset), several
phone midpoints as seek times, and writes expected viseme + training mel
context so Godot can compare MelFrontend + ONNX against the training path.
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
from model_contract import load_model_contract  # noqa: E402
from train_viseme_smoke import windows  # noqa: E402


def pick_seeks(phones: list[dict], n: int) -> list[dict]:
    """Phone midpoints spread across the utterance (prefer non-silence)."""
    scored = []
    for p in phones:
        mid = 0.5 * (float(p["start"]) + float(p["end"]))
        lab = str(p["phone"]).strip().lower()
        silence = lab in ("sil", "sp", "spn", "")
        scored.append((silence, mid, p))
    scored.sort(key=lambda x: (x[0], x[1]))
    non_sil = [s for s in scored if not s[0]]
    pool = non_sil if len(non_sil) >= max(3, n // 2) else scored
    if not pool:
        return []
    if len(pool) <= n:
        chosen = pool
    else:
        idxs = np.linspace(0, len(pool) - 1, n, dtype=int)
        chosen = [pool[i] for i in sorted(set(int(i) for i in idxs))]
    return [{"t_sec": float(mid), "phone": p["phone"]} for _, mid, p in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="ci-fixture")
    ap.add_argument("--stem", default="", help="Utterance stem (default: first TextGrid)")
    ap.add_argument("--seeks", type=int, default=6)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "seek_probe.json",
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "model.onnx",
        help="Canonical model contract and inference graph",
    )
    ap.add_argument(
        "--viseme-map",
        type=Path,
        default=ROOT / "configs" / "viseme_map_en_us_arpa.json",
    )
    args = ap.parse_args()

    aligned = ROOT / "data" / "aligned" / args.subset
    prepared = ROOT / "data" / "prepared" / args.subset
    tdir = ROOT / "data" / "tensors" / args.subset
    if not aligned.is_dir():
        print(f"Missing alignments: {aligned}", file=sys.stderr)
        print("For LibriSpeech: prepare + MFA + build_train_tensors first.", file=sys.stderr)
        return 1

    grids = sorted(aligned.rglob("*.TextGrid"))
    if not grids:
        print(f"No TextGrids in {aligned}", file=sys.stderr)
        return 1
    if args.stem:
        grids = [g for g in grids if g.stem == args.stem]
        if not grids:
            print(f"Stem not found: {args.stem}", file=sys.stderr)
            return 1
    tg_path = grids[0]
    stem = tg_path.stem

    wav = tg_path.with_suffix(".wav")
    if not wav.exists():
        cands = list(prepared.rglob(f"{stem}.wav"))
        wav = cands[0] if cands else None
    if wav is None or not wav.exists():
        print(f"No wav for {stem}", file=sys.stderr)
        return 1

    npz_path = tdir / f"{stem}.npz"
    meta = load_model_contract(args.onnx, require_schema=2)
    ctx = int(meta["context_frames"])
    hop = float(meta["audio"]["hop_length_samples"]) / float(meta["audio"]["sample_rate"])
    id_to_name = {int(v): str(k) for k, v in meta["visemes"].items()}
    name_to_idx, phone_to_idx = load_viseme_map(args.viseme_map)

    if npz_path.exists():
        z = np.load(npz_path)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
    else:
        # Same training path as build_train_tensors — lets editor Load work before
        # a full tensor cache exists for the stem.
        from build_train_tensors import frame_viseme_ids, mel_features

        print(f"no {npz_path}; computing mel via mel_features_c", file=sys.stderr)
        X = mel_features(wav).astype(np.float32)
        phones_for_y = phones_from_textgrid(tg_path)
        y = frame_viseme_ids(phones_for_y, X.shape[0], hop, phone_to_idx)

    Xw, yw = windows(X, y, ctx)

    phones = phones_from_textgrid(tg_path)
    seeks_raw = pick_seeks(phones, args.seeks)
    seeks = []
    for s in seeks_raw:
        t = float(s["t_sec"])
        frame = int(t / hop)
        frame = max(0, min(frame, int(y.shape[0]) - 1))
        # Window row index in causal stack (first window ends at ctx-1)
        win_i = frame - (ctx - 1)
        if win_i < 0:
            win_i = 0
        if win_i >= Xw.shape[0]:
            win_i = int(Xw.shape[0]) - 1
        expect_id = int(yw[win_i])
        mel = Xw[win_i].astype(np.float32)
        seeks.append(
            {
                "t_sec": round(t, 4),
                "frame": int(frame),
                "window": int(win_i),
                "phone": s["phone"],
                "expect_id": expect_id,
                "expect_name": id_to_name.get(expect_id, str(expect_id)),
                "mel_l2_norm": float(np.linalg.norm(mel)),
                "mel_context": [float(f"{v:.6g}") for v in mel.tolist()],
            }
        )

    # Prefer repo-relative wav under export/ci-smoke for portable Godot opens.
    wav_rel = str(wav.relative_to(ROOT))
    ci_wav = ROOT / "export" / "ci-smoke" / "ci-fixture.wav"
    if args.subset == "ci-fixture" and ci_wav.exists():
        wav_rel = "export/ci-smoke/ci-fixture.wav"
    elif args.subset != "ci-fixture":
        dest = ROOT / "export" / "ci-smoke" / f"seek_{stem}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_mtime < wav.stat().st_mtime:
            dest.write_bytes(wav.read_bytes())
        wav_rel = f"export/ci-smoke/seek_{stem}.wav"

    out = {
        "subset": args.subset,
        "stem": stem,
        "wav": wav_rel,
        "model_onnx": str(args.onnx.resolve().relative_to(ROOT)),
        "onnx": str(args.onnx.resolve().relative_to(ROOT)),
        "context_frames": ctx,
        "hop_s": hop,
        "visemes": name_to_idx,
        "note": (
            "mel_context is the training-path flattened causal window. "
            "Godot MelFrontend must match within mel_l2_max."
        ),
        "mel_l2_max": 0.05,
        "seeks": seeks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} stem={stem} seeks={len(seeks)}")
    for s in seeks:
        print(
            f"  t={s['t_sec']:.3f}s frame={s['frame']} "
            f"expect={s['expect_name']} phone={s['phone']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
