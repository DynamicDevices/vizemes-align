#!/usr/bin/env python3
"""Sparse one-byte events carried once per 20 ms Opus packet.

The audio receiver buffers packets before playback, so transition boundaries can
arrive after their corresponding audio packet and still be rendered on time.
Boundary events encode how many packets ago the boundary occurred. Idle packets
carry a no-op; periodic resync packets make packet loss self-healing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


FRAME_SECONDS = 0.020
MAX_VISEME_ID = 14  # IDs 0..14; high nibble 0xF is reserved for control.
MAX_PACKETS_AGO = 7
NO_EVENT = 0xF0
RESYNC_BASE = 0xF1


class BoundaryKind(IntEnum):
    START = 0
    END = 1


@dataclass(frozen=True)
class BoundaryEvent:
    target_id: int
    kind: BoundaryKind
    packets_ago: int


def _check_byte(packet: int) -> None:
    if not 0 <= packet <= 0xFF:
        raise ValueError("packet must be one byte")


def pack_boundary(target_id: int, kind: BoundaryKind, packets_ago: int) -> int:
    """Encode a transition boundary relative to the packet carrying this byte."""
    if not 0 <= target_id <= MAX_VISEME_ID:
        raise ValueError("target_id must be in the 15-viseme vocabulary (0..14)")
    if not 0 <= packets_ago <= MAX_PACKETS_AGO:
        raise ValueError("packets_ago must be in [0, 7]")
    low = packets_ago | (0x08 if BoundaryKind(kind) == BoundaryKind.END else 0)
    return (target_id << 4) | low


def unpack_boundary(packet: int) -> BoundaryEvent:
    """Decode a boundary byte; control bytes are rejected."""
    _check_byte(packet)
    target = packet >> 4
    if target == 0x0F:
        raise ValueError("control byte is not a boundary event")
    low = packet & 0x0F
    kind = BoundaryKind.END if low & 0x08 else BoundaryKind.START
    return BoundaryEvent(target, kind, low & 0x07)


def pack_resync(current_id: int) -> int:
    """Encode an absolute dominant-viseme recovery point."""
    if not 0 <= current_id <= MAX_VISEME_ID:
        raise ValueError("current_id must be in the 15-viseme vocabulary (0..14)")
    return RESYNC_BASE + current_id


def unpack_resync(packet: int) -> int:
    _check_byte(packet)
    if not RESYNC_BASE <= packet <= 0xFF:
        raise ValueError("packet is not a resync control")
    return packet - RESYNC_BASE


@dataclass
class _Transition:
    base_id: int
    target_id: int
    start_packet: int | None = None
    end_packet: int | None = None


@dataclass
class EventDecoder:
    """Stateful buffered receiver for boundary events and loss recovery."""

    viseme_count: int = 15
    base_id: int = 0
    transition: _Transition | None = None

    def ingest(self, packet: int, stream_packet: int) -> None:
        """Ingest metadata attached to ``stream_packet`` on the audio timeline."""
        _check_byte(packet)
        if packet == NO_EVENT:
            return
        if packet >> 4 == 0x0F:
            self.base_id = unpack_resync(packet)
            if self.base_id >= self.viseme_count:
                raise ValueError(f"resync viseme {self.base_id} outside vocabulary")
            self.transition = None
            return

        event = unpack_boundary(packet)
        if event.target_id >= self.viseme_count:
            raise ValueError(f"target {event.target_id} outside vocabulary")
        boundary_packet = stream_packet - event.packets_ago
        if event.kind == BoundaryKind.START:
            self.transition = _Transition(self.base_id, event.target_id, start_packet=boundary_packet)
            return
        if self.transition is None or self.transition.target_id != event.target_id:
            # A lost START cannot be reconstructed safely. The next periodic
            # resync restores the dominant state without inventing a boundary.
            return
        self.transition.end_packet = boundary_packet
        if self.transition.start_packet is not None and boundary_packet <= self.transition.start_packet:
            raise ValueError("transition END must follow START")

    def weights_at(self, audio_packet: float) -> np.ndarray:
        """Render weights for a buffered audio packet (fractional indices allowed)."""
        weights = np.zeros(self.viseme_count, dtype=np.float32)
        transition = self.transition
        if transition is None or transition.start_packet is None or transition.end_packet is None:
            weights[self.base_id] = 1.0
            return weights

        if audio_packet <= transition.start_packet:
            progress = 0.0
        elif audio_packet >= transition.end_packet:
            progress = 1.0
        else:
            progress = (audio_packet - transition.start_packet) / (
                transition.end_packet - transition.start_packet
            )
        weights[transition.base_id] = 1.0 - progress
        weights[transition.target_id] += progress
        if progress >= 1.0:
            self.base_id = transition.target_id
            self.transition = None
        return weights
