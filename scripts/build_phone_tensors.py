#!/usr/bin/env python3
"""Build frame-level phone targets from MFA while reusing canonical mel tensors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_godot_package import phones_from_textgrid  # noqa: E402


def load_phone_inventory(path: Path) -> tuple[list[str], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = ["silence"] + [str(phone) for phone in data["phones"]]
    return labels, {phone: index + 1 for index, phone in enumerate(data["phones"])}


def frame_phone_ids(
    intervals: list[dict], n_frames: int, hop_seconds: float, phone_to_id: dict[str, int]
) -> np.ndarray:
    """Assign the phone covering each mel-frame midpoint; gaps are silence id 0."""
    targets = np.zeros(n_frames, dtype=np.int64)
    interval_index = 0
    for frame in range(n_frames):
        time_s = (frame + 0.5) * hop_seconds
        while interval_index < len(intervals) and float(intervals[interval_index]["end"]) <= time_s:
            interval_index += 1
        if interval_index >= len(intervals):
            continue
        interval = intervals[interval_index]
        if float(interval["start"]) <= time_s < float(interval["end"]):
            targets[frame] = phone_to_id.get(str(interval["phone"]), 0)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "configs" / "phone_inventory_en_us_arpa.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    mel_dir = ROOT / "data" / "tensors" / args.subset
    aligned_dir = ROOT / "data" / "aligned" / args.subset
    out_dir = ROOT / "data" / "phone-tensors" / args.subset
    mel_index = json.loads((mel_dir / "index.json").read_text(encoding="utf-8"))
    labels, phone_to_id = load_phone_inventory(args.inventory)
    hop_seconds = float(mel_index["audio"]["hop_length_samples"]) / float(
        mel_index["audio"]["sample_rate"]
    )
    utterances = mel_index["utterances"][: args.limit or None]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for utterance in utterances:
        stem = str(utterance["id"])
        grid = aligned_dir / f"{stem}.TextGrid"
        if not grid.is_file():
            matches = list(aligned_dir.rglob(f"{stem}.TextGrid"))
            if not matches:
                print(f"skip {stem}: no TextGrid", file=sys.stderr)
                continue
            grid = matches[0]
        mel_npz = mel_dir / str(utterance["path"])
        with np.load(mel_npz) as mel:
            n_frames = int(mel["X"].shape[0])
        targets = frame_phone_ids(phones_from_textgrid(grid), n_frames, hop_seconds, phone_to_id)
        target_path = out_dir / f"{stem}.npz"
        np.savez_compressed(target_path, y_phone=targets)
        rows.append(
            {
                "id": stem,
                "mel_path": str(mel_npz.relative_to(ROOT)),
                "phone_path": target_path.name,
                "frames": n_frames,
            }
        )
    index = {
        "subset": args.subset,
        "audio": mel_index["audio"],
        "inventory": args.inventory.name,
        "phones": {label: index for index, label in enumerate(labels)},
        "evaluation": "stress-collapsed ARPA; silence excluded from PER",
        "utterances": rows,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"PHONE_TENSORS_OK utterances={len(rows)} labels={len(labels)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
