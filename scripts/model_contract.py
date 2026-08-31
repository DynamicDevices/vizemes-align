#!/usr/bin/env python3
"""Read and validate the canonical Vizemes contract embedded in an ONNX model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelContractError(ValueError):
    """The ONNX model is missing or contains inconsistent Vizemes metadata."""


def custom_metadata(onnx_path: Path) -> dict[str, str]:
    """Return ONNX custom metadata without requiring the training-only ``onnx`` package."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


def load_model_contract(onnx_path: Path, *, require_schema: int = 1) -> dict[str, Any]:
    """Load the full embedded contract and verify the flat bootstrap keys agree."""
    props = custom_metadata(onnx_path)
    try:
        schema = int(props.get("vizemes_schema", "0"))
    except ValueError as exc:
        raise ModelContractError("vizemes_schema is not an integer") from exc
    if schema < require_schema:
        raise ModelContractError(f"metadata schema {schema} < required {require_schema}")

    blob = props.get("vizemes_meta_json", "")
    if not blob:
        raise ModelContractError("missing vizemes_meta_json")
    try:
        contract = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ModelContractError("invalid vizemes_meta_json") from exc
    if not isinstance(contract, dict):
        raise ModelContractError("vizemes_meta_json must be an object")

    model = str(contract.get("model", ""))
    if not model or props.get("vizemes_model", model) != model:
        raise ModelContractError("model identity missing or inconsistent")
    audio = contract.get("audio")
    if not isinstance(audio, dict):
        raise ModelContractError("missing audio contract")
    for key in ("sample_rate", "hop_length_samples", "window_length_samples", "n_fft", "n_mels"):
        if float(audio.get(key, 0)) <= 0:
            raise ModelContractError(f"invalid audio.{key}")

    vocabulary = contract.get("visemes") or contract.get("phones")
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ModelContractError("missing phone/viseme vocabulary")
    flat_vocab = props.get("vizemes_vocabulary_json")
    if schema >= 2:
        if not flat_vocab:
            raise ModelContractError("schema 2 requires vizemes_vocabulary_json")
        if json.loads(flat_vocab) != vocabulary:
            raise ModelContractError("flat vocabulary disagrees with full contract")
        if not props.get("vizemes_normalization"):
            raise ModelContractError("schema 2 requires vizemes_normalization")

    contract["_onnx_path"] = str(onnx_path)
    contract["_schema"] = schema
    contract["_normalization"] = props.get(
        "vizemes_normalization", str(contract.get("normalization", ""))
    )
    contract["_provenance"] = json.loads(props.get("vizemes_provenance_json", "{}"))
    return contract


def id_to_label(contract: dict[str, Any]) -> dict[int, str]:
    vocabulary = contract.get("visemes") or contract.get("phones") or {}
    return {int(index): str(label) for label, index in vocabulary.items()}

