#!/usr/bin/env python3
"""Summarise MFA phone interval durations and render one distribution panel per phone."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from export_godot_package import phones_from_textgrid  # noqa: E402


def canonical_phone(label: str, keep_stress: bool) -> str:
    value = label.strip()
    return value if keep_stress else "".join(char for char in value if not char.isdigit())


def write_plot(samples: dict[str, list[float]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    labels = sorted(samples)
    columns = 4
    rows = max(1, (len(labels) + columns - 1) // columns)
    figure, axes = pyplot.subplots(rows, columns, figsize=(14, max(3, rows * 2.5)))
    for axis, label in zip(np.ravel(axes), labels):
        durations_ms = np.asarray(samples[label], dtype=np.float64) * 1000.0
        bins = min(20, max(5, int(np.sqrt(durations_ms.size))))
        axis.hist(durations_ms, bins=bins, color="#1f4e79", edgecolor="white")
        axis.axvline(durations_ms.mean(), color="#c45c26", linewidth=1.2)
        axis.set_title(f"{label} (n={durations_ms.size})", fontsize=9)
        axis.set_xlabel("ms", fontsize=8)
        axis.tick_params(labelsize=8)
    for axis in list(np.ravel(axes))[len(labels) :]:
        axis.set_visible(False)
    figure.suptitle("MFA phone duration distributions (orange = mean)", fontsize=13)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    pyplot.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "export" / "phone-duration-report")
    parser.add_argument("--keep-stress", action="store_true")
    args = parser.parse_args()

    aligned = ROOT / "data" / "aligned" / args.subset
    grids = sorted(aligned.rglob("*.TextGrid"))
    if not grids:
        raise FileNotFoundError(f"no MFA TextGrids under {aligned}")

    samples: dict[str, list[float]] = defaultdict(list)
    for grid in grids:
        for phone in phones_from_textgrid(grid):
            duration = float(phone["end"]) - float(phone["start"])
            label = canonical_phone(str(phone["phone"]), args.keep_stress)
            if label and duration > 0.0:
                samples[label].append(duration)

    rows: list[dict[str, object]] = []
    for label in sorted(samples):
        durations_ms = np.asarray(samples[label], dtype=np.float64) * 1000.0
        rows.append(
            {
                "phone": label,
                "count": int(durations_ms.size),
                "mean_ms": round(float(durations_ms.mean()), 3),
                "median_ms": round(float(np.median(durations_ms)), 3),
                "stddev_ms": round(float(durations_ms.std()), 3),
                "min_ms": round(float(durations_ms.min()), 3),
                "p90_ms": round(float(np.percentile(durations_ms, 90)), 3),
                "max_ms": round(float(durations_ms.max()), 3),
            }
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phone_durations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(
            report, fieldnames=list(rows[0]) if rows else ["phone"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "subset": args.subset,
        "stress_collapsed": not args.keep_stress,
        "textgrids": len(grids),
        "phones": rows,
        "plot": "phone_duration_distributions.png",
        "method": "MFA phone TextGrid interval duration; this measures aligned acoustic-phone spans, not visual transition durations.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_plot(samples, out_dir / "phone_duration_distributions.png")
    lines = [
        "# Phone duration report",
        "",
        f"MFA `TextGrid` intervals from `{args.subset}`; {len(grids)} clips; "
        + ("stress kept." if args.keep_stress else "ARPA stress digits collapsed."),
        "",
        "![Phone duration distributions](phone_duration_distributions.png)",
        "",
        "| Phone | Count | Mean ms | Median ms | Stddev ms | P90 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['phone']} | {row['count']} | {row['mean_ms']} | {row['median_ms']} | "
        f"{row['stddev_ms']} | {row['p90_ms']} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "These are alignment durations. They are useful evidence for phone-specific transition policies, "
            "but do not alone establish the duration of the corresponding visible mouth movement.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PHONE_DURATION_REPORT_OK phones={len(rows)} clips={len(grids)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
