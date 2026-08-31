#!/usr/bin/env python3
"""One-byte stateful viseme transition contract for each 20 ms Opus packet."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FRAME_SECONDS = 0.020
MAX_VISEME_ID = 15
MAX_BLEND = 15


def pack_transition(target_id: int, progress: float) -> int:
    """Pack target viseme and transition progress (0=start, 1=target reached)."""
    if not 0 <= target_id <= MAX_VISEME_ID:
        raise ValueError("target_id must fit four bits")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    blend = int(round(progress * MAX_BLEND))
    return (target_id << 4) | blend


def unpack_transition(packet: int) -> tuple[int, float]:
    if not 0 <= packet <= 0xFF:
        raise ValueError("packet must be one byte")
    return (packet >> 4) & 0x0F, float(packet & 0x0F) / MAX_BLEND


@dataclass
class TransitionDecoder:
    """Receiver state; the previous completed target is implicit in the stream."""

    viseme_count: int = 15
    base_id: int = 0

    def decode(self, packet: int) -> np.ndarray:
        target, progress = unpack_transition(packet)
        if target >= self.viseme_count:
            raise ValueError(f"target {target} outside vocabulary")
        weights = np.zeros(self.viseme_count, dtype=np.float32)
        weights[self.base_id] = 1.0 - progress
        weights[target] += progress
        if progress >= 1.0:
            self.base_id = target
        return weights

