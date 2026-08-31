#!/usr/bin/env python3
"""Copy a plain MFA dictionary and append reviewed pronunciation reductions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_entry(line: str) -> tuple[str, tuple[str, ...]] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    fields = text.split("\t")
    if len(fields) == 2:
        word, pronunciation = fields
    elif len(fields) >= 3:
        word, pronunciation = fields[0], fields[-1]
    else:
        raise ValueError(f"dictionary entries must be tab-separated: {line!r}")
    return word.casefold(), tuple(pronunciation.split())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dictionary", type=Path)
    parser.add_argument("output_dictionary", type=Path)
    parser.add_argument(
        "--reductions",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "pronunciation_reductions_en_us_arpa.tsv",
    )
    args = parser.parse_args()
    if args.output_dictionary.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dictionary}")

    base_text = args.base_dictionary.read_text(encoding="utf-8")
    known = {entry for line in base_text.splitlines() if (entry := parse_entry(line))}
    additions: list[str] = []
    for raw in args.reductions.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 3:
            raise ValueError(f"reduction entries require word, probability, pronunciation: {raw!r}")
        word, probability, pronunciation = fields
        float_probability = float(probability)
        if not 0.01 <= float_probability <= 1.0:
            raise ValueError(f"probability outside MFA range 0.01..1.0: {raw!r}")
        key = (word.casefold(), tuple(pronunciation.split()))
        if key not in known:
            additions.append(f"{word}\t{probability}\t{pronunciation}")
            known.add(key)

    args.output_dictionary.parent.mkdir(parents=True, exist_ok=True)
    payload = base_text.rstrip("\n") + "\n" + "\n".join(additions) + "\n"
    args.output_dictionary.write_text(payload, encoding="utf-8")
    manifest = {
        "schema": 1,
        "base_dictionary": str(args.base_dictionary),
        "base_sha256": digest(args.base_dictionary),
        "reductions": str(args.reductions),
        "reductions_sha256": digest(args.reductions),
        "output_dictionary": str(args.output_dictionary),
        "output_sha256": digest(args.output_dictionary),
        "added_entries": additions,
    }
    args.output_dictionary.with_suffix(args.output_dictionary.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REDUCED_DICTIONARY_OK added={len(additions)} out={args.output_dictionary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
