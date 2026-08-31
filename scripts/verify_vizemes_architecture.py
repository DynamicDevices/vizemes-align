#!/usr/bin/env python3
"""Gate the shipped self-describing Vizemes architecture and A/B evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import load_model_contract  # noqa: E402
from viseme_sidechannel import (  # noqa: E402
    FRAME_SECONDS,
    MAX_PACKETS_AGO,
    MAX_VISEME_ID,
    NO_EVENT,
    RESYNC_BASE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_metrics(row: dict, label: str) -> None:
    for key in (
        "frame_accuracy",
        "boundary_mean_absolute_ms",
        "boundary_p95_absolute_ms",
        "reference_transitions",
        "hypothesis_transitions",
        "transition_ratio",
        "excess_transition_jitter_hz",
    ):
        require(key in row, f"{label}: missing {key}")


def main() -> int:
    pack_root = ROOT / "godot-demo" / "addons" / "vizeme-onnxmodels"
    sidecars = sorted(pack_root.rglob("model*.json"))
    require(not sidecars, f"runtime model JSON sidecars returned: {sidecars}")

    direct_path = pack_root / "tier-b-tcn" / "model_final.onnx"
    phone_path = pack_root / "tier-c" / "model_final.onnx"
    direct = load_model_contract(direct_path, require_schema=2)
    phone = load_model_contract(phone_path, require_schema=2)
    require(direct["model"] == "viseme_tcn", "direct baseline identity changed")
    require(phone["model"] == "phone_tcn", "phone model identity changed")
    for contract, label in ((direct, "direct"), (phone, "phone")):
        require(contract.get("normalization"), f"{label}: missing normalization")
        require("lookahead_ms" in contract, f"{label}: missing lookahead")
        require(contract.get("checkpoint"), f"{label}: missing checkpoint")
        require(contract.get("params", 0) > 0, f"{label}: missing parameter provenance")
        require(contract.get("elapsed_s", 0) >= 590, f"{label}: not the controlled run")

    mapping = phone.get("phone_to_viseme")
    require(isinstance(mapping, dict), "phone ONNX lacks deterministic mapper")
    require(mapping.get("type") == "posterior_sum_causal_exponential_top_k", "mapper type")
    require(len(mapping.get("visemes", {})) == 15, "mapper must expose 15 visemes")
    require(
        len(mapping.get("phone_to_viseme_ids", [])) == len(phone["phones"]),
        "mapper column count disagrees with phone vocabulary",
    )
    require(mapping.get("smoothing_seconds") == 0.06, "mapper smoothing changed")
    require(mapping.get("top_k") == 2, "mapper sparsity changed")
    require(len(mapping.get("config_sha256", "")) == 64, "mapper source hash missing")

    report = json.loads(
        (ROOT / "docs" / "benchmarks" / "direct_phone_ab.json").read_text(encoding="utf-8")
    )
    require(report["protocol"]["held_out_utterances"] == 262, "A/B held-out split changed")
    require(report["protocol"]["frames"] == 190492, "A/B frame population changed")
    require_metrics(report["direct_viseme_tcn"], "direct A/B")
    require_metrics(report["phone_tcn_deterministic"], "phone A/B")
    runtime = report.get("runtime", {})
    for key in (
        "direct_onnx_seconds",
        "phone_onnx_seconds",
        "deterministic_mapper_seconds",
        "direct_realtime_factor",
        "phone_pipeline_realtime_factor",
    ):
        require(float(runtime.get(key, 0)) > 0, f"runtime: missing {key}")

    learned = json.loads(
        (ROOT / "docs" / "benchmarks" / "learned_mapper_ab.json").read_text(encoding="utf-8")
    )
    require_metrics(learned["deterministic"], "deterministic challenger baseline")
    require_metrics(learned["learned"], "learned challenger")
    require(
        learned["protocol"]["decision"] == "reject_learned_mapper_keep_deterministic",
        "learned mapper decision is absent or ambiguous",
    )

    require(FRAME_SECONDS == 0.020, "sidechannel cadence changed")
    require(MAX_VISEME_ID == 14, "sidechannel viseme range changed")
    require(MAX_PACKETS_AGO == 7, "sidechannel history changed")
    require(NO_EVENT == 0xF0 and RESYNC_BASE == 0xF1, "sidechannel controls changed")
    godot_utils = (ROOT / "godot-demo" / "viseme_utils.gd").read_text(encoding="utf-8")
    for token in (
        "SIDECHANNEL_FRAME_SECONDS := 0.02",
        "SIDECHANNEL_MAX_PACKETS_AGO := 7",
        "SIDECHANNEL_NO_EVENT := 0xF0",
        "SIDECHANNEL_RESYNC_BASE := 0xF1",
        "class SidechannelDecoder",
        "phone_logits_to_viseme_series",
    ):
        require(token in godot_utils, f"Godot runtime lacks {token}")

    print(
        "VIZEMES_ARCHITECTURE_CONTRACT_OK "
        "onnx_only=1 direct_phone_ab=262 mapper=deterministic "
        "sidechannel=1byte@20ms godot_runtime=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
