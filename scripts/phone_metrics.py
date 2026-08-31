#!/usr/bin/env python3
"""Phone recognition sequence and boundary metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def collapse_phone_frames(ids: np.ndarray, labels: list[str]) -> list[str]:
    """Collapse repeats, remove silence, and strip ARPA stress for PER."""
    sequence: list[str] = []
    previous = -1
    for raw in np.asarray(ids).reshape(-1):
        index = int(raw)
        if index == previous:
            continue
        previous = index
        if index <= 0 or index >= len(labels):
            continue
        label = labels[index].rstrip("012")
        if label and (not sequence or sequence[-1] != label):
            sequence.append(label)
    return sequence


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        nxt = [i]
        for j, hyp in enumerate(hypothesis, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ref != hyp)))
        row = nxt
    return row[-1]


def phone_error_rate(reference_ids: np.ndarray, hypothesis_ids: np.ndarray, labels: list[str]) -> float:
    reference = collapse_phone_frames(reference_ids, labels)
    hypothesis = collapse_phone_frames(hypothesis_ids, labels)
    return float(edit_distance(reference, hypothesis)) / float(max(1, len(reference)))


def transition_frames(ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids).reshape(-1)
    return np.flatnonzero(values[1:] != values[:-1]) + 1


@dataclass(frozen=True)
class BoundaryMetrics:
    mean_absolute_ms: float
    p95_absolute_ms: float
    matched: int


def boundary_metrics(reference_ids: np.ndarray, hypothesis_ids: np.ndarray, hop_ms: float = 10.0) -> BoundaryMetrics:
    """Greedily match ordered transition boundaries and report timing error."""
    reference = transition_frames(reference_ids)
    hypothesis = transition_frames(hypothesis_ids)
    matched = min(reference.size, hypothesis.size)
    if matched == 0:
        return BoundaryMetrics(float("inf"), float("inf"), 0)
    errors = np.abs(reference[:matched] - hypothesis[:matched]).astype(np.float64) * hop_ms
    return BoundaryMetrics(float(errors.mean()), float(np.percentile(errors, 95)), int(matched))

