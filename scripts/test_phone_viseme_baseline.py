#!/usr/bin/env python3
"""Deterministic stage-B and sparse 20 ms side-channel regression tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from phone_viseme_mapper import PhoneVisemeMap
from viseme_sidechannel import (
    FRAME_SECONDS,
    MAX_PACKETS_AGO,
    NO_EVENT,
    BoundaryKind,
    EventDecoder,
    pack_boundary,
    pack_resync,
    unpack_boundary,
    unpack_resync,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    phones = ["sil", "P", "B", "F", "AA1", "IY0"]
    mapper = PhoneVisemeMap.from_config(ROOT / "configs/viseme_map_en_us_arpa.json", phones)
    posterior = np.zeros((12, len(phones)), dtype=np.float32)
    posterior[:4, 1] = 1.0  # P → PP
    posterior[4:8, 2] = 1.0  # B → same PP (must not create a visual transition)
    posterior[8:, 3] = 1.0  # F → FF
    weights = mapper.map_posteriors(posterior, smoothing_seconds=0.03)
    assert weights.shape == (12, 15)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.count_nonzero(weights > 0, axis=1).max() <= 2
    assert int(weights[3].argmax()) == 1
    assert int(weights[7].argmax()) == 1
    assert int(weights[-1].argmax()) == 2

    # Julian's example: at packet 5 report that target 1 started three packets
    # ago; at packet 6 report its end one packet ago. A buffered receiver can
    # reconstruct the transition on audio packets 2..5 without repeated levels.
    decoder = EventDecoder()
    start = pack_boundary(1, BoundaryKind.START, packets_ago=3)
    end = pack_boundary(1, BoundaryKind.END, packets_ago=1)
    decoder.ingest(start, stream_packet=5)
    decoder.ingest(end, stream_packet=6)
    assert unpack_boundary(start).packets_ago == 3
    assert decoder.weights_at(2)[0] == 1.0
    assert np.allclose(decoder.weights_at(3.5)[[0, 1]], [0.5, 0.5])
    assert decoder.weights_at(5)[1] == 1.0

    # A missing boundary is bounded damage: ignore an orphan END, then recover
    # the absolute state with one periodic resync byte.
    lost = EventDecoder()
    lost.ingest(end, stream_packet=6)
    assert lost.weights_at(5)[0] == 1.0
    recovery = pack_resync(1)
    lost.ingest(recovery, stream_packet=10)
    assert unpack_resync(recovery) == 1
    assert lost.weights_at(10)[1] == 1.0

    packets = [NO_EVENT, start, end, recovery]
    assert len(bytes(packets)) == len(packets)
    assert MAX_PACKETS_AGO * FRAME_SECONDS == 0.14
    assert int(round(1.0 / FRAME_SECONDS)) == 50
    print("PHONE_VISEME_BASELINE_OK frames=12 sparse_top_k=2")
    print(
        "VISEME_SIDECHANNEL_OK bytes_per_frame=1 frames_per_second=50 "
        "bitrate=400bps relative_history_ms=140 loss_recovery=resync"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
