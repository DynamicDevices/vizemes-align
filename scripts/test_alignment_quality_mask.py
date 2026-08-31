#!/usr/bin/env python3
"""Regression checks for reduction merging and frame exclusion semantics."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from build_train_tensors import frame_validity
from train_viseme_smoke import windows


def main() -> int:
    valid = frame_validity([{"start": 0.02, "end": 0.04}], 6, 0.01)
    assert valid.tolist() == [1, 1, 0, 0, 1, 1], valid
    X = np.arange(18, dtype=np.float32).reshape(6, 3)
    y = np.arange(6, dtype=np.int64)
    xw, yw = windows(X, y, 2, valid)
    assert yw.tolist() == [1, 4, 5], yw
    assert xw.shape == (3, 6), xw.shape

    from build_reduced_dictionary import main as build_dictionary
    import sys

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / "base.dict"
        reductions = root / "reductions.tsv"
        output = root / "derived.dict"
        base.write_text("and\tAE1 N D\n", encoding="utf-8")
        reductions.write_text("and\t0.7\tAE1 N\nand\t0.2\tAE1 N D\n", encoding="utf-8")
        previous = sys.argv
        try:
            sys.argv = ["build_reduced_dictionary.py", str(base), str(output), "--reductions", str(reductions)]
            assert build_dictionary() == 0
        finally:
            sys.argv = previous
        text = output.read_text()
        assert text.count("AE1 N D") == 1
        assert "and\t0.7\tAE1 N\n" in text
    print("ALIGNMENT_QUALITY_MASK_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
