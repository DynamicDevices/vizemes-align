# Two-stage: phones → visemes (design)

Julian (Telegram 890): keep acoustic discrimination on a richer label set, then
a second net (with history) produces clean viseme weights — one active, or a
pair during a smooth transition.

## Why

OpenLipSync collapses phones→visemes in the *dataset* and trains a deep TCN
straight to 15 classes. That works when the backbone is strong. Our shallow MLP
plateaued; throwing away phone identity also blocks later remapping / blend
policy without retraining.

## Stages

| Stage | Input | Output | File |
|-------|--------|--------|------|
| A | mel context (`context_frames × n_mels`) | phone (or phone-group) logits / posteriors per hop | `phone.onnx` |
| B | history of phone posteriors (`H` hops × `P` phones) | 15 viseme weights (sum≈1; sparse) | `viseme_map.onnx` |

MelFrontend + OnnxLoader stay shared. Stage A uses the same mel config embedded
in `phone.onnx` metadata (`vizemes_*` / `phone_*` props). Stage B metadata
describes `history_hops`, `n_phones`, `n_visemes`, and the phone↔viseme table
hash/version.

## Stage B behaviour

- Prefer **sparse multi-label**: at most two visemes with mass during transitions
  (crossfade), else one-hot.
- History length: start `H = 20` hops (200 ms at 10 ms hop) — enough for local
  coarticulation without a huge MLP.
- Train B on MFA phone timelines → mapped soft viseme targets (same 60 ms blend
  as OpenLipSync), *or* on rule-based crossfade from hard phone maps as a
  bootstrap, then refine.
- Inference graph in Godot: mel → A → ring buffer of posteriors → B → face /
  timeline.

## Interface sketch (GDScript)

```gdscript
# After MelFrontend yields a context:
var phone_logits := phone_loader.predict(ctx)          # PackedFloat32Array P
phone_hist.push_back(softmax(phone_logits))            # keep last H
var flat := flatten(phone_hist)                        # H*P
var viseme_w := viseme_loader.predict(flat)            # 15 weights
```

## Near-term implementation order

1. ONNX-only Mel configure (metadata on A’s file) — in progress.
2. Export phone-aligned tensors from MFA (keep ARPA inventory; no early collapse).
3. Train small phone head (TCN or wider MLP) on `test-clean` → `export/tier-c/phone.onnx`.
4. Train stage-B mapper on phone soft-histories → `export/tier-c/viseme_map.onnx`.
5. Timeline A/B compare: direct viseme MLP vs two-stage.

## Non-goals (yet)

- Full OpenLipSync TCN reimplementation in-repo (can vendor weights later).
- Replacing MFA with ASR CTC in stage A (optional upgrade).
