#!/usr/bin/env python3
"""Export MFA TextGrids + wavs to a Godot-readable package with coded visemes."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import textgrid
except ImportError:
    print("pip install textgrid", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]


def load_viseme_map(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    data = json.loads(path.read_text())
    name_to_idx = {k: int(v) for k, v in data["viseme_set"]["visemes"].items()}
    phone_to_name = data["phoneme_to_viseme"]
    phone_to_idx = {}
    for phone, name in phone_to_name.items():
        phone_to_idx[phone] = name_to_idx.get(name, 0)
        phone_to_idx[phone.lower()] = name_to_idx.get(name, 0)
        phone_to_idx[phone.upper()] = name_to_idx.get(name, 0)
    return name_to_idx, phone_to_idx


def phones_from_textgrid(tg_path: Path) -> list[dict]:
    tg = textgrid.TextGrid.fromFile(str(tg_path))
    tier = None
    for t in tg.tiers:
        if t.name.lower() in ("phones", "phone"):
            tier = t
            break
    if tier is None:
        # fall back to first interval tier
        for t in tg.tiers:
            if hasattr(t, "intervals"):
                tier = t
                break
    if tier is None:
        return []
    out = []
    for iv in tier.intervals:
        lab = (iv.mark or "").strip()
        if not lab:
            continue
        out.append({"start": float(iv.minTime), "end": float(iv.maxTime), "phone": lab})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument(
        "--viseme-map",
        type=Path,
        default=ROOT / "configs" / "viseme_map_en_us_arpa.json",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional max utterances (0=all)")
    args = ap.parse_args()

    prepared = ROOT / "data" / "prepared" / args.subset
    aligned = ROOT / "data" / "aligned" / args.subset
    export = ROOT / "data" / "export" / args.subset
    if not aligned.exists():
        print(f"Missing alignments: {aligned}", file=sys.stderr)
        return 1

    name_to_idx, phone_to_idx = load_viseme_map(args.viseme_map)
    export.mkdir(parents=True, exist_ok=True)
    manifest = {
        "subset": args.subset,
        "viseme_map": str(args.viseme_map.name),
        "visemes": name_to_idx,
        "clips": [],
    }

    grids = sorted(aligned.rglob("*.TextGrid"))
    if args.limit:
        grids = grids[: args.limit]

    for tg_path in grids:
        stem = tg_path.stem
        # wav may sit next to TextGrid in aligned tree or in prepared
        wav = tg_path.with_suffix(".wav")
        if not wav.exists():
            candidates = list(prepared.rglob(f"{stem}.wav"))
            wav = candidates[0] if candidates else None
        if wav is None or not wav.exists():
            print(f"skip {stem}: no wav", file=sys.stderr)
            continue

        phones = phones_from_textgrid(tg_path)
        intervals = []
        for p in phones:
            key = p["phone"]
            # strip stress digits e.g. AH0
            base = "".join(c for c in key if not c.isdigit())
            vid = phone_to_idx.get(key, phone_to_idx.get(base, phone_to_idx.get(base.upper(), 0)))
            intervals.append(
                {
                    "start": p["start"],
                    "end": p["end"],
                    "phone": key,
                    "viseme_id": vid,
                }
            )

        clip_dir = export / stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, clip_dir / "audio.wav")
        label_path = clip_dir / "visemes.json"
        label_path.write_text(
            json.dumps(
                {
                    "id": stem,
                    "audio": "audio.wav",
                    "intervals": intervals,
                },
                indent=2,
            )
            + "\n"
        )
        # also CSV for easy Godot import
        with (clip_dir / "visemes.csv").open("w", encoding="utf-8") as f:
            f.write("start,end,viseme_id,phone\n")
            for iv in intervals:
                f.write(f"{iv['start']:.4f},{iv['end']:.4f},{iv['viseme_id']},{iv['phone']}\n")

        manifest["clips"].append({"id": stem, "path": stem})

    (export / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {len(manifest['clips'])} clips → {export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
