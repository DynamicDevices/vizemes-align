#!/usr/bin/env python3
"""Write auditable per-phone quality sidecars and exclusion spans beside MFA TextGrids."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_godot_package import phones_from_textgrid  # noqa: E402


def canonical(label: str) -> str:
    return "".join(c for c in label.strip().upper() if not c.isdigit())


def overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--aligned-dir", type=Path)
    parser.add_argument("--manual-exclusions", type=Path, default=ROOT / "configs/alignment_manual_exclusions.json")
    parser.add_argument("--duration-z-threshold", type=float, default=4.0)
    parser.add_argument("--utterance-duration-z-threshold", type=float, default=3.0)
    parser.add_argument("--min-phone-samples", type=int, default=20)
    args = parser.parse_args()
    aligned = args.aligned_dir or ROOT / "data" / "aligned" / args.subset
    grids = sorted(aligned.rglob("*.TextGrid"))
    if not grids:
        raise FileNotFoundError(f"no TextGrids under {aligned}")

    manual = json.loads(args.manual_exclusions.read_text()) if args.manual_exclusions.exists() else {"utterances": {}}
    durations: dict[str, list[float]] = defaultdict(list)
    parsed: dict[Path, list[dict]] = {}
    for grid in grids:
        phones = phones_from_textgrid(grid)
        parsed[grid] = phones
        for phone in phones:
            label = canonical(str(phone["phone"]))
            duration = float(phone["end"]) - float(phone["start"])
            if label and duration > 0:
                durations[label].append(duration)

    stats = {}
    for label, values in durations.items():
        data = np.asarray(values, dtype=np.float64)
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        stats[label] = {"count": int(data.size), "median_s": median, "mad_s": mad}

    analysis_rows: dict[str, dict] = {}
    analysis_path = aligned / "alignment_analysis.csv"
    if analysis_path.exists():
        with analysis_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = Path(row.get("file", row.get("file_name", ""))).stem
                if key:
                    analysis_rows[key] = row

    summary_rows = []
    for grid, phones in parsed.items():
        stem = grid.stem
        exclusions = []
        for item in manual.get("utterances", {}).get(stem, []):
            allowed = {canonical(x) for x in item.get("phones", [])}
            if not allowed:
                exclusions.append(dict(item, source="manual"))
                continue
            for phone in phones:
                start, end = float(phone["start"]), float(phone["end"])
                if canonical(str(phone["phone"])) in allowed and overlap(
                    start, end, float(item["start"]), float(item["end"])
                ):
                    exclusions.append({
                        "start": start, "end": end, "phone": str(phone["phone"]),
                        "source": "manual", "reason": item["reason"],
                    })
        phone_rows = []
        utterance_z_values = []
        for phone in phones:
            start, end = float(phone["start"]), float(phone["end"])
            label = canonical(str(phone["phone"]))
            duration = end - start
            stat = stats.get(label, {})
            robust_z = None
            if stat.get("count", 0) >= args.min_phone_samples and stat.get("mad_s", 0.0) > 0:
                robust_z = 0.67448975 * (duration - stat["median_s"]) / stat["mad_s"]
                utterance_z_values.append(abs(robust_z))
                if abs(robust_z) >= args.duration_z_threshold:
                    exclusions.append({
                        "start": start, "end": end, "source": "phone_duration_outlier",
                        "reason": f"{label} robust duration z={robust_z:.3f} exceeds {args.duration_z_threshold}",
                    })
            phone_rows.append({
                "start": start, "end": end, "phone": str(phone["phone"]),
                "duration_ms": round(duration * 1000.0, 3),
                "robust_duration_z": None if robust_z is None else round(float(robust_z), 4),
            })
        utterance_duration_z = float(np.mean(utterance_z_values)) if utterance_z_values else None
        if utterance_duration_z is not None and utterance_duration_z >= args.utterance_duration_z_threshold:
            exclusions.append({
                "start": min((float(p["start"]) for p in phones), default=0.0),
                "end": max((float(p["end"]) for p in phones), default=0.0),
                "source": "utterance_duration_outlier",
                "reason": f"mean absolute robust phone-duration z={utterance_duration_z:.3f}",
            })
        for row in phone_rows:
            row["valid"] = not any(overlap(row["start"], row["end"], float(x["start"]), float(x["end"])) for x in exclusions)
        payload = {
            "schema": 1, "stem": stem, "textgrid": grid.name,
            "mfa_alignment_analysis": analysis_rows.get(stem),
            "quality_method": {
                "phone_duration": "median/MAD robust z within canonical phone across this alignment set",
                "duration_z_threshold": args.duration_z_threshold,
                "utterance_duration_z_threshold": args.utterance_duration_z_threshold,
                "manual_exclusions": str(args.manual_exclusions),
                "training_semantics": "frames overlapping exclusions have valid=0 and are omitted from loss and metrics",
            },
            "utterance_mean_abs_duration_z": None if utterance_duration_z is None else round(utterance_duration_z, 4),
            "phones": phone_rows, "excluded_intervals": exclusions,
        }
        sidecar = grid.with_suffix(".quality.json")
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary_rows.append({"stem": stem, "excluded_intervals": len(exclusions), "quality_file": sidecar.name})
    (aligned / "alignment_quality_summary.json").write_text(json.dumps({
        "schema": 1, "subset": args.subset, "textgrids": len(grids),
        "utterances_with_exclusions": sum(bool(x["excluded_intervals"]) for x in summary_rows),
        "utterances": summary_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"ALIGNMENT_QUALITY_OK grids={len(grids)} excluded_utterances={sum(bool(x['excluded_intervals']) for x in summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
