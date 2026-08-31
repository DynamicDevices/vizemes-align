#!/usr/bin/env python3
"""Migrate a model sidecar into canonical ONNX ``metadata_props``.

The sibling JSON is accepted as a compatibility input while older training and
evaluation tools are migrated.  After this command succeeds, the ONNX metadata
is the runtime source of truth and is verified by re-reading the saved model.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate_latency(audio: dict, context_frames: int, lookahead_ms: float) -> dict:
    """Document the three latency layers: Speex DSP, mel context, label lookahead."""
    sr = float(audio.get("sample_rate", 16000))
    win = float(audio.get("window_length_samples", 400))
    hop = float(audio.get("hop_length_samples", 160))
    # Time spanned by a full mel context ending at "now".
    context_s = (win + hop * max(0, context_frames - 1)) / sr
    return {
        "hop_seconds": hop / sr,
        "window_seconds": win / sr,
        "mel_context_seconds": round(context_s, 6),
        "label_lookahead_ms": lookahead_ms,
        "label_lookahead_seconds": round(float(lookahead_ms) / 1000.0, 6),
        "speex_dsp_seconds": (
            "runtime: MelFrontend.get_dsp_latency_seconds() "
            "(resampler + optional AEC/preprocess frames; often ~0 if bypassed)"
        ),
        "playback_compensation_hint_seconds": round(
            context_s + float(lookahead_ms) / 1000.0, 6
        ),
        "note": (
            "Feed mono float PCM at sample_rate with n_mels/hop/window matching audio{}. "
            "ONNX input is flattened context_frames * n_mels. "
            "ONNX metadata_props are the runtime source of truth for MelFrontend.configure."
        ),
    }


def load_quality(hit_rate_path: Path | None, model_key: str, onnx_path: Path | None = None) -> dict | None:
    if hit_rate_path is None or not hit_rate_path.is_file():
        return None
    data = json.loads(hit_rate_path.read_text(encoding="utf-8"))
    models = (data.get("overall") or {}).get("models") or {}
    aliases = {
        "model_final": "tier_b_final",
        "model_10m": "tier_b_10m",
        "model_20m": "tier_b_20m",
        "model_b": "ci_smoke",
    }
    key = aliases.get(model_key)
    if key is None and model_key == "model":
        # Ambiguous stem: prefer path hint
        path_s = str(onnx_path or "")
        key = "ci_smoke" if "ci-smoke" in path_s else "tier_b_final"
    if key is None:
        key = model_key
    row = models.get(key)
    if not row:
        return None
    try:
        source = str(hit_rate_path.relative_to(ROOT))
    except ValueError:
        source = str(hit_rate_path)
    return {
        "metric": "mfa_vs_predict_frame_acc",
        "all_acc": row.get("acc"),
        "non_silence_acc": row.get("mid_acc"),
        "n_frames": row.get("n"),
        "source": source,
        "caveat": "Soft-label + lookahead eval; not production lip-sync quality.",
    }


def embed_onnx(onnx_path: Path, meta: dict) -> None:
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path))
    # Clear prior vizemes_* keys then rewrite
    keep = [p for p in model.metadata_props if not p.key.startswith("vizemes_")]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)

    vocabulary = meta.get("visemes") or meta.get("phones") or {}
    provenance_keys = (
        "subset",
        "checkpoint",
        "step",
        "elapsed_s",
        "train_utterances",
        "val_utterances",
        "params",
    )
    provenance = {k: meta[k] for k in provenance_keys if k in meta}
    normalization = str(meta.get("normalization", "per_utterance_per_mel_mean_std"))
    flat = {
        "vizemes_schema": "2",
        "vizemes_doc": str(meta.get("doc", meta.get("note", ""))),
        "vizemes_model": str(meta.get("model", "")),
        "vizemes_subset": str(meta.get("subset", "")),
        "vizemes_checkpoint": str(meta.get("checkpoint", "")),
        "vizemes_normalization": normalization,
        "vizemes_vocabulary_json": json.dumps(vocabulary, separators=(",", ":")),
        "vizemes_provenance_json": json.dumps(provenance, separators=(",", ":")),
        "vizemes_context_frames": str(meta.get("context_frames", "")),
        "vizemes_n_mels": str(meta.get("n_mels", "")),
        "vizemes_input_features": str(meta.get("input_features", "")),
        "vizemes_n_visemes": str(meta.get("n_visemes", "")),
        "vizemes_sample_rate": str((meta.get("audio") or {}).get("sample_rate", "")),
        "vizemes_hop_length_samples": str((meta.get("audio") or {}).get("hop_length_samples", "")),
        "vizemes_window_length_samples": str((meta.get("audio") or {}).get("window_length_samples", "")),
        "vizemes_n_fft": str((meta.get("audio") or {}).get("n_fft", "")),
        "vizemes_fmin": str((meta.get("audio") or {}).get("fmin", "")),
        "vizemes_fmax": str((meta.get("audio") or {}).get("fmax", "")),
        "vizemes_lookahead_ms": str(meta.get("lookahead_ms", "")),
        "vizemes_blend_ms": str(meta.get("blend_ms", "")),
        "vizemes_soft_boundaries": str(meta.get("soft_boundaries", "")),
        "vizemes_mel_context_seconds": str((meta.get("latency") or {}).get("mel_context_seconds", "")),
        "vizemes_playback_compensation_hint_seconds": str(
            (meta.get("latency") or {}).get("playback_compensation_hint_seconds", "")
        ),
        "vizemes_quality_all_acc": str((meta.get("quality") or {}).get("all_acc", "")),
        "vizemes_quality_non_silence_acc": str((meta.get("quality") or {}).get("non_silence_acc", "")),
        "vizemes_updated_utc": meta.get("metadata_updated_utc", ""),
        "vizemes_meta_json": json.dumps(meta, separators=(",", ":")),
    }
    for k, v in flat.items():
        if v == "" or v == "None":
            continue
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v if len(v) < 60000 else v[:60000]
    onnx.save(model, str(onnx_path))

    # The saved ONNX, not the migration sidecar, is authoritative from here.
    check = onnx.load(str(onnx_path))
    saved = {p.key: p.value for p in check.metadata_props}
    required = {
        "vizemes_schema",
        "vizemes_model",
        "vizemes_normalization",
        "vizemes_vocabulary_json",
        "vizemes_meta_json",
    }
    missing = sorted(k for k in required if not saved.get(k))
    if missing:
        raise RuntimeError(f"ONNX metadata verification failed; missing {missing}")


def refresh_one(onnx_path: Path, hit_rate: Path | None) -> Path:
    json_path = onnx_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    audio = meta.get("audio") or {}
    ctx = int(meta.get("context_frames", 20))
    meta["latency"] = estimate_latency(audio, ctx, float(meta.get("lookahead_ms", 0.0)))
    q = load_quality(hit_rate, onnx_path.stem, onnx_path)
    if q:
        meta["quality"] = q
    meta["metadata_updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["onnx"] = onnx_path.name
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    embed_onnx(onnx_path, meta)
    print(
        f"META_OK {onnx_path} mel_context_s={meta['latency']['mel_context_seconds']} "
        f"quality={meta.get('quality')}"
    )
    return json_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "onnx",
        nargs="+",
        type=Path,
        help="ONNX path(s); sibling .json required",
    )
    ap.add_argument(
        "--hit-rate",
        type=Path,
        default=ROOT / "export" / "tier-b" / "hit_rate_summary.json",
    )
    args = ap.parse_args()
    hit = args.hit_rate if args.hit_rate.is_file() else None
    for p in args.onnx:
        if p.is_symlink():
            p = p.resolve()
        refresh_one(p, hit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
