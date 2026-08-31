#!/usr/bin/env python3
"""Gate shipped model packs: schema-2 ONNX contracts and no model sidecars."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_contract import load_model_contract  # noqa: E402


def main() -> int:
    roots = [
        ROOT / "export" / "ci-smoke",
        ROOT / "godot-demo" / "addons" / "vizeme-onnxmodels",
    ]
    sidecars = sorted(path for root in roots for path in root.rglob("model*.json"))
    if sidecars:
        raise AssertionError(f"model JSON sidecars are forbidden: {sidecars}")
    models = sorted(path for root in roots for path in root.rglob("model*.onnx"))
    if not models:
        raise AssertionError("no ONNX models found")
    for model in models:
        contract = load_model_contract(model, require_schema=2)
        for key in ("model", "audio", "normalization"):
            if not contract.get(key):
                raise AssertionError(f"{model}: missing {key}")
        if not (contract.get("visemes") or contract.get("phones")):
            raise AssertionError(f"{model}: missing vocabulary")
    print(f"ONNX_ONLY_PACKS_OK models={len(models)} sidecars=0 schema=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
