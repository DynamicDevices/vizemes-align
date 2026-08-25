#!/usr/bin/env python3
"""CI: tiny multi-viseme fixture (audible tones + phones covering all 15 visemes).

Previously used a silent wav → all-zero mel tensors → useless ONNX sanity table.
"""
from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# One ARPA phone per viseme class (see configs/viseme_map_en_us_arpa.json).
PHONE_SEGMENTS = [
    "sil",
    "P",
    "F",
    "TH",
    "T",
    "K",
    "CH",
    "S",
    "N",
    "R",
    "AA0",
    "EH0",
    "IH0",
    "OW0",
    "UW0",
]
SEG_S = 0.12  # seconds per phone
SR = 16000


def write_tone_wav(path: Path) -> float:
    """Non-silent mono wav: distinct sine per phone segment → non-zero mels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[int] = []
    for i, _phone in enumerate(PHONE_SEGMENTS):
        freq = 180.0 + i * 70.0
        n = int(SEG_S * SR)
        for t in range(n):
            # Mild amplitude envelope so segments are separable in mel space.
            env = 0.35 * (0.55 + 0.45 * math.sin(math.pi * t / max(n - 1, 1)))
            sample = env * math.sin(2.0 * math.pi * freq * t / SR)
            frames.append(int(max(-1.0, min(1.0, sample)) * 32767.0))
    duration = len(frames) / SR
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", s) for s in frames))
    return duration


def write_multiviseme_textgrid(path: Path, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(PHONE_SEGMENTS)
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "phones"',
        "        xmin = 0",
        f"        xmax = {duration}",
        f"        intervals: size = {n}",
    ]
    for i, phone in enumerate(PHONE_SEGMENTS):
        xmin = i * SEG_S
        xmax = duration if i == n - 1 else (i + 1) * SEG_S
        lines += [
            f"        intervals [{i + 1}]:",
            f"            xmin = {xmin}",
            f"            xmax = {xmax}",
            f'            text = "{phone}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    subset = "ci-fixture"
    prepared = ROOT / "data" / "prepared" / subset
    aligned = ROOT / "data" / "aligned" / subset
    prepared.mkdir(parents=True, exist_ok=True)
    aligned.mkdir(parents=True, exist_ok=True)
    utt = "ci-0001"
    duration = write_tone_wav(prepared / f"{utt}.wav")
    write_multiviseme_textgrid(aligned / f"{utt}.TextGrid", duration)
    write_tone_wav(aligned / f"{utt}.wav")

    import export_godot_package as egp

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
