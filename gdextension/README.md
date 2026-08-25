# Godot GDExtension (planned)

**Status:** scaffold only — not a loadable extension yet.

## Runtime contract (matching smoke ONNX)

| Piece | SoT |
|-------|-----|
| Mel (C) | `OpenLipSync/audio/mel_spectrogram.c` (+ `.h`) — torchaudio-compatible |
| Model | `export/ci-smoke/model.onnx` (smoke MLP) + `model.json` |
| Input | Flattened causal mel context `(batch, context * n_mels)` — see JSON |
| Output | Viseme logits `(batch, n_visemes)` |

## Build plan

1. GDExtension skeleton (Godot 4.x) linking OpenLipSync mel C.
2. ONNX Runtime (or Godot ONNX addon) loading `model.onnx`.
3. Node/API: push PCM frames → viseme weights at 100 fps.

Until that lands, use OpenLipSync’s C# inference path against the same mel recipe, or Python + onnxruntime for harness tests.

## Generate a fresh smoke ONNX

```bash
nix develop .#train
python3 scripts/train_viseme_smoke.py --subset ci-fixture --export-onnx export/ci-smoke/model.onnx
# moderate laptop run:
python3 scripts/build_train_tensors.py --subset test-clean --limit 40
python3 scripts/train_viseme_smoke.py --subset test-clean --limit 40 --max-frames 80000 \
  --export-onnx export/smoke-test-clean/model.onnx
```
