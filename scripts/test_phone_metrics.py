#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from build_phone_tensors import frame_phone_ids
from phone_metrics import boundary_metrics, collapse_phone_frames, phone_error_rate


def main() -> int:
    labels = ["silence", "AA0", "AA1", "B", "F"]
    intervals = [
        {"start": 0.00, "end": 0.03, "phone": "AA0"},
        {"start": 0.03, "end": 0.06, "phone": "B"},
        {"start": 0.06, "end": 0.09, "phone": "F"},
    ]
    reference = frame_phone_ids(intervals, 9, 0.01, {"AA0": 1, "B": 3, "F": 4})
    hypothesis = np.array([1, 1, 1, 1, 3, 3, 4, 4, 4])
    assert collapse_phone_frames(np.array([1, 1, 2, 2, 3]), labels) == ["AA", "B"]
    assert np.isclose(phone_error_rate(reference, reference, labels), 0.0)
    assert phone_error_rate(reference, hypothesis, labels) == 0.0  # same phones, shifted boundary
    boundaries = boundary_metrics(reference, hypothesis)
    assert boundaries.matched == 2
    assert np.isclose(boundaries.mean_absolute_ms, 5.0)
    print(
        "PHONE_METRICS_OK "
        f"per=0 boundary_mean_ms={boundaries.mean_absolute_ms:.1f} "
        f"boundary_p95_ms={boundaries.p95_absolute_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
