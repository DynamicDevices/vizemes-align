#!/usr/bin/env python3
"""Select a deterministic sex-balanced speaker subset from a prepared LibriSpeech corpus."""
from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path


def speaker_sexes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(";"):
            continue
        fields = [value.strip() for value in line.split("|")]
        if len(fields) >= 2:
            result[fields[0]] = fields[1]
    return result


def spread(values: list[str], count: int) -> list[str]:
    if count >= len(values):
        return values
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--speakers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speaker-count", type=int, default=10)
    args = parser.parse_args()

    sexes = speaker_sexes(args.speakers)
    available = sorted({path.stem.split("-")[0] for path in args.prepared.glob("*.wav")})
    female = [speaker for speaker in available if sexes.get(speaker) == "F"]
    male = [speaker for speaker in available if sexes.get(speaker) == "M"]
    female_count = args.speaker_count // 2
    selected = sorted(spread(female, female_count) + spread(male, args.speaker_count - female_count))
    if len(selected) != args.speaker_count:
        raise RuntimeError(f"could only select {len(selected)} speakers from {len(available)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    utterances = []
    total_seconds = 0.0
    for wav_path in sorted(args.prepared.glob("*.wav")):
        speaker = wav_path.stem.split("-")[0]
        if speaker not in selected:
            continue
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            raise FileNotFoundError(lab_path)
        seconds = duration(wav_path)
        for source in (wav_path, lab_path):
            speaker_dir = args.out_dir / speaker
            speaker_dir.mkdir(parents=True, exist_ok=True)
            destination = speaker_dir / source.name
            if not destination.exists():
                os.link(source, destination)
        utterances.append({"id": wav_path.stem, "speaker": speaker, "seconds": seconds})
        total_seconds += seconds

    manifest = {
        "source": str(args.prepared),
        "speaker_metadata": str(args.speakers),
        "selection": "evenly-spaced speaker IDs within LibriSpeech sex groups",
        "selected_speakers": selected,
        "female_speakers": [speaker for speaker in selected if sexes.get(speaker) == "F"],
        "male_speakers": [speaker for speaker in selected if sexes.get(speaker) == "M"],
        "utterances": utterances,
        "utterance_count": len(utterances),
        "duration_seconds": total_seconds,
    }
    (args.out_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PILOT_OK speakers={len(selected)} utterances={len(utterances)} minutes={total_seconds / 60:.1f}")
    print("speakers=" + ",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
