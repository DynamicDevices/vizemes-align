#!/usr/bin/env python3
"""Regression gate: runtime model configuration is complete inside ONNX."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import id_to_label, load_model_contract  # noqa: E402


def main() -> int:
    model = ROOT / "export" / "ci-smoke" / "model.onnx"
    contract = load_model_contract(model, require_schema=2)
    labels = id_to_label(contract)
    assert contract["_schema"] == 2
    assert contract["_normalization"] == "per_utterance_per_mel_mean_std"
    assert int(contract["input_features"]) == 1600
    assert int(contract["audio"]["sample_rate"]) == 16000
    assert labels[0] == "silence"
    assert len(labels) == 15
    print(
        "MODEL_CONTRACT_OK "
        f"schema={contract['_schema']} model={contract['model']} "
        f"features={contract['input_features']} labels={len(labels)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
