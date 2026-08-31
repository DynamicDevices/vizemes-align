#!/usr/bin/env python3
"""Explainable stage-B baseline: phone posteriors → smoothed viseme weights."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhoneVisemeMap:
    phone_labels: tuple[str, ...]
    viseme_labels: tuple[str, ...]
    matrix: np.ndarray  # [phones, visemes], one-hot rows

    @classmethod
    def from_config(cls, path: Path, phone_labels: list[str]) -> "PhoneVisemeMap":
        data = json.loads(path.read_text(encoding="utf-8"))
        visemes = {str(k): int(v) for k, v in data["viseme_set"]["visemes"].items()}
        by_id = tuple(name for name, _ in sorted(visemes.items(), key=lambda item: item[1]))
        phone_to_name = {str(k).upper(): str(v) for k, v in data["phoneme_to_viseme"].items()}
        matrix = np.zeros((len(phone_labels), len(by_id)), dtype=np.float32)
        for row, raw_phone in enumerate(phone_labels):
            phone = raw_phone.upper()
            base = phone.rstrip("012")
            name = phone_to_name.get(phone, phone_to_name.get(base, "silence"))
            matrix[row, visemes.get(name, 0)] = 1.0
        return cls(tuple(phone_labels), by_id, matrix)

    def map_posteriors(
        self,
        phone_posteriors: np.ndarray,
        *,
        hop_seconds: float = 0.01,
        smoothing_seconds: float = 0.06,
        top_k: int = 2,
    ) -> np.ndarray:
        """Map `[time, phones]` to causal, sparse `[time, visemes]` weights.

        Phone probabilities that share a viseme are summed. A causal exponential
        smoother models transition inertia; retaining the two strongest outputs
        makes every frame explainable as one pose or a crossfade between two.
        """
        p = np.asarray(phone_posteriors, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != self.matrix.shape[0]:
            raise ValueError(f"phone posterior shape {p.shape} incompatible with {self.matrix.shape}")
        if hop_seconds <= 0 or smoothing_seconds < 0:
            raise ValueError("invalid timing")
        row_sum = p.sum(axis=1, keepdims=True)
        p = np.divide(p, row_sum, out=np.zeros_like(p), where=row_sum > 0)
        mapped = p @ self.matrix
        alpha = 1.0 if smoothing_seconds == 0 else 1.0 - math.exp(-hop_seconds / smoothing_seconds)
        out = np.zeros_like(mapped)
        state = np.zeros(mapped.shape[1], dtype=np.float32)
        for t, frame in enumerate(mapped):
            state += alpha * (frame - state)
            if top_k > 0 and top_k < state.size:
                keep = np.argpartition(state, -top_k)[-top_k:]
                sparse = np.zeros_like(state)
                sparse[keep] = state[keep]
            else:
                sparse = state.copy()
            total = float(sparse.sum())
            if total > 0:
                sparse /= total
            out[t] = sparse
        return out

