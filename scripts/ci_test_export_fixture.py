#!/usr/bin/env python3
"""CI: build a tiny TextGrid+wav fixture and run export_godot_package."""
from __future__ import annotations

import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def write_silent_wav(path: Path, seconds: float = 0.5, sr: int = 16000) -> None:
    n = int(seconds * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def write_minimal_textgrid(path: Path, duration: float = 0.5) -> None:
    # Praat long TextGrid — class must be IntervalTier (singular)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {duration}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = {duration}
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "sil"
        intervals [2]:
            xmin = 0.2
            xmax = {duration}
            text = "AH0"
""",
        encoding="utf-8",
    )


def main() -> int:
    subset = "ci-fixture"
    prepared = ROOT / "data" / "prepared" / subset
    aligned = ROOT / "data" / "aligned" / subset
    prepared.mkdir(parents=True, exist_ok=True)
    aligned.mkdir(parents=True, exist_ok=True)
    utt = "ci-0001"
    write_silent_wav(prepared / f"{utt}.wav")
    write_minimal_textgrid(aligned / f"{utt}.TextGrid")
    # also put wav next to TextGrid for exporter fallback
    write_silent_wav(aligned / f"{utt}.wav")

    import export_godot_package as egp

    # call main via argv
    sys.argv = [
        "export_godot_package.py",
        "--subset",
        subset,
        "--viseme-map",
        str(ROOT / "configs" / "viseme_map_en_us_arpa.json"),
    ]
    rc = egp.main()
    if rc != 0:
        return rc
    export = ROOT / "data" / "export" / subset
    assert (export / "manifest.json").is_file(), "manifest missing"
    assert (export / utt / "visemes.json").is_file(), "visemes.json missing"
    print("ci_test_export_fixture OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
