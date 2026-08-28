#!/usr/bin/env python3
"""Export viseme_timeline.json for Godot editor overlay plot.

Writes MFA/trained viseme boxes (time spans) for one utterance so Godot can
draw 15 ONNX weight curves vs those labels on the same time axis.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_godot_package import load_viseme_map, phones_from_textgrid, words_from_textgrid  # noqa: E402


def merge_viseme_boxes(
    phones: list[dict], phone_to_idx: dict[str, int], id_to_name: dict[int, str]
) -> list[dict]:
    """Collapse consecutive phones that map to the same viseme id."""
    boxes: list[dict] = []
    for p in phones:
        start = float(p["start"])
        end = float(p["end"])
        if end <= start:
            continue
        lab = str(p["phone"]).strip()
        key = lab.lower()
        if key in ("sil", "sp", "spn", ""):
            vid = 0
        else:
            vid = int(phone_to_idx.get(lab, phone_to_idx.get(key, 0)))
        name = id_to_name.get(vid, str(vid))
        if boxes and boxes[-1]["expect_id"] == vid and abs(boxes[-1]["end"] - start) < 1e-4:
            boxes[-1]["end"] = end
            boxes[-1]["phones"].append(lab)
        else:
            boxes.append(
                {
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "expect_id": vid,
                    "expect_name": name,
                    "phones": [lab],
                }
            )
    return boxes


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return float(w.getnframes()) / float(w.getframerate())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="ci-fixture")
    ap.add_argument("--stem", default="", help="Utterance stem (default: first TextGrid)")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "viseme_timeline.json",
    )
    ap.add_argument(
        "--model-json",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "model.json",
    )
    ap.add_argument(
        "--viseme-map",
        type=Path,
        default=ROOT / "configs" / "viseme_map_en_us_arpa.json",
    )
    args = ap.parse_args()

    aligned = ROOT / "data" / "aligned" / args.subset
    prepared = ROOT / "data" / "prepared" / args.subset
    if not aligned.is_dir():
        print(f"Missing alignments: {aligned}", file=sys.stderr)
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

    meta = json.loads(args.model_json.read_text(encoding="utf-8"))
    hop = float(meta["audio"]["hop_length_samples"]) / float(meta["audio"]["sample_rate"])
    id_to_name = {int(v): str(k) for k, v in meta["visemes"].items()}
    name_to_idx, phone_to_idx = load_viseme_map(args.viseme_map)

    phones = phones_from_textgrid(tg_path)
    boxes = merge_viseme_boxes(phones, phone_to_idx, id_to_name)
    words = words_from_textgrid(tg_path)

    # Prefer a portable copy under export/ci-smoke for Godot (esp. test-clean).
    wav_rel = str(wav.relative_to(ROOT))
    ci_wav = ROOT / "export" / "ci-smoke" / "ci-fixture.wav"
    if args.subset == "ci-fixture" and ci_wav.exists():
        wav_rel = "export/ci-smoke/ci-fixture.wav"
        duration_s = wav_duration_s(ci_wav)
    else:
        dest = ROOT / "export" / "ci-smoke" / f"timeline_{stem}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_mtime < wav.stat().st_mtime:
            dest.write_bytes(wav.read_bytes())
        wav_rel = f"export/ci-smoke/timeline_{stem}.wav"
        duration_s = wav_duration_s(dest)

    names = [id_to_name[i] for i in range(len(id_to_name))]

    out = {
        "subset": args.subset,
        "stem": stem,
        "wav": wav_rel,
        "model_json": "export/ci-smoke/model.json",
        "onnx": "export/ci-smoke/model.onnx",
        "onnx_b": "",
        "label_a": "A:hidden64",
        "label_b": "B",
        "context_frames": int(meta["context_frames"]),
        "hop_s": hop,
        "duration_s": round(duration_s, 4),
        "viseme_names": names,
        "visemes": name_to_idx,
        "boxes": boxes,
        "words": words,
        "note": (
            "Godot runs MelFrontend+ONNX across the wav and draws 15 weight curves; "
            "boxes are MFA phones collapsed to trained viseme labels; "
            "words are MFA word intervals (when the TextGrid has a words tier). "
            "Optional onnx_b overlays a second model (toggle A/B/D in editor)."
        ),
    }
    # Prefer a thinner sibling model when present (timeline A/B).
    model_b = ROOT / "export" / "ci-smoke" / "model_b.onnx"
    if model_b.is_file():
        out["onnx_b"] = "export/ci-smoke/model_b.onnx"
        out["label_b"] = "B:hidden16"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} stem={stem} duration={duration_s:.2f}s boxes={len(boxes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
