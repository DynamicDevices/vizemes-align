#!/usr/bin/env python3
"""Deterministic stage-B and 20 ms side-channel regression tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from phone_viseme_mapper import PhoneVisemeMap
from viseme_sidechannel import FRAME_SECONDS, TransitionDecoder, pack_transition, unpack_transition

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

    decoder = TransitionDecoder()
    packets = [pack_transition(1, x) for x in (0.0, 0.5, 1.0)]
    decoded = [decoder.decode(packet) for packet in packets]
    assert unpack_transition(packets[1]) == (1, 8 / 15)
    assert decoded[0][0] == 1.0
    assert np.isclose(decoded[1].sum(), 1.0)
    assert decoder.base_id == 1
    assert len(bytes(packets)) == len(packets)
    assert int(round(1.0 / FRAME_SECONDS)) == 50
    print("PHONE_VISEME_BASELINE_OK frames=12 sparse_top_k=2")
    print("VISEME_SIDECHANNEL_OK bytes_per_frame=1 frames_per_second=50 bitrate=400bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
