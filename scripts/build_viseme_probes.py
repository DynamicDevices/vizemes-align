#!/usr/bin/env python3
"""Build hard-coded viseme probe bank for ONNX sanity (human-readable quality).

Writes data/tensors/ci-fixture/viseme_probes.npz with ~20 fixed inputs:
  - real labelled windows from ci-fixture where available (silence / aa today)
  - deterministic synthetic mel templates for every viseme id (stable, no RNG)

A strong ONNX should map probes toward their expected labels; a weak/smoke
model trained on little data will miss most classes — that gap is the point.

  python3 scripts/build_viseme_probes.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def windows(X: np.ndarray, y: np.ndarray, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] < ctx:
        pad = np.repeat(X[:1], ctx - X.shape[0], axis=0)
        X = np.concatenate([pad, X], axis=0)
        y = np.concatenate([np.repeat(y[:1], ctx - y.shape[0]), y], axis=0)
    xs, ys = [], []
    for i in range(ctx - 1, X.shape[0]):
        xs.append(X[i - ctx + 1 : i + 1].reshape(-1))
        ys.append(y[i])
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def synthetic_template(vid: int, ctx: int, n_mels: int) -> np.ndarray:
    """Deterministic mel context meant as a class cue (not speech-derived)."""
    x = np.full((ctx, n_mels), -45.0, dtype=np.float32)
    band = (vid * 5) % max(1, n_mels - 4)
    x[:, band : band + 3] = -12.0 - 0.3 * vid
    # late-context accent so causal window cares about "now"
    x[-4:, band] = -5.0 - 0.2 * vid
    x[-2:, (band + 10) % n_mels] = -8.0
    return x.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "export" / "ci-smoke" / "model.json",
    )
    ap.add_argument(
        "--subset",
        default="ci-fixture",
        help="data/tensors/<subset> for real labelled windows",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "tensors" / "ci-fixture" / "viseme_probes.npz",
    )
    ap.add_argument("--target", type=int, default=20, help="Approx probe count")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    ctx = int(meta["context_frames"])
    n_mels = int(meta["n_mels"])
    n_visemes = int(meta["n_visemes"])
    id_to_name = {int(v): k for k, v in meta["visemes"].items()}

    by: dict[int, list[np.ndarray]] = defaultdict(list)
    tdir = ROOT / "data" / "tensors" / args.subset
    index_path = tdir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for u in index["utterances"]:
            z = np.load(tdir / u["path"])
            xw, yw = windows(z["X"], z["y"], ctx)
            for x, yi in zip(xw, yw):
                by[int(yi)].append(x)

    probes_x: list[np.ndarray] = []
    probes_y: list[int] = []
    tags: list[str] = []
    sources: list[str] = []

    # One synthetic template per viseme (15) — hard-coded, stable.
    for vid in range(n_visemes):
        probes_x.append(synthetic_template(vid, ctx, n_mels))
        probes_y.append(vid)
        tags.append(f"{id_to_name[vid]}_synth")
        sources.append("synth")

    # Fill toward ~20 with real labelled windows (diverse frames).
    for vid in sorted(by.keys()):
        # stride through available frames for variety
        real = by[vid]
        step = max(1, len(real) // 3)
        for j, x in enumerate(real[::step][:3]):
            if len(probes_x) >= args.target:
                break
            probes_x.append(x.astype(np.float32))
            probes_y.append(vid)
            tags.append(f"{id_to_name[vid]}_real{j}")
            sources.append("real")
        if len(probes_x) >= args.target:
            break

    X = np.stack(probes_x).astype(np.float32)
    y = np.asarray(probes_y, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        tags=np.asarray(tags),
        source=np.asarray(sources),
        viseme_names=np.asarray([id_to_name[i] for i in range(n_visemes)]),
        context_frames=np.asarray([ctx]),
        n_mels=np.asarray([n_mels]),
    )
    print(
        f"wrote {args.out} n={len(y)} "
        f"synth={sum(s == 'synth' for s in sources)} "
        f"real={sum(s == 'real' for s in sources)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
