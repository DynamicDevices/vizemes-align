# Godot GDExtension (in progress)

**Status:** C runtime scaffold + vendored mel frontend. **Not** a loadable
`.gdextension` yet (no godot-cpp / ONNX Runtime link).

OpenLipSync’s C mel is a **starting frontend**, not a compatibility lock.
Waveform→features stays swappable (mel, LPC, …). Each exported ONNX keeps its
own sidecar contract (`export/ci-smoke/model.json`).

## Layout

| Path | Role |
|------|------|
| `include/feature_frontend.h` | Pluggable frontend ops |
| `include/viseme_runtime.h` | PCM → viseme weights (Godot-facing SoT) |
| `frontends/mel/` | First impl (vendored mel C + ORIGIN.md) |
| `src/mel_frontend.c` | Mel → `VizemesFrontendOps` |
| `src/viseme_runtime_stub.c` | API stub until ONNX Runtime linked |
| `../export/ci-smoke/` | Smoke `model.onnx` + `model.json` |
| `../scripts/sanity_check_onnx.py` | Python ORT harness (works today) |

## Target Godot shape ([goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync))

Demo drives Ready Player Me blendshapes via `VisemeSystem.set_visemes(vv)` with
OVR-style names (`sil`, `PP`, …, `U`, optional `LA`). Today that comes from
OVRLipSync through two-voip. Our runtime should emit the same kind of float
weight vector so the avatar path can stay.

Smoke MLP: 15 classes (see `export/ci-smoke/model.json`); map `silence`↔`sil`,
`ih`/`oh`/`ou`↔`I`/`O`/`U` as needed. No laughter channel yet.

## Host build (no Godot)

```bash
cd gdextension
make
# → build/libvizemes_runtime.a  (mel + stub; push_pcm returns -1 until ORT)
```

## Next

1. Link ONNX Runtime in `vizeme_runtime` (load `model.onnx`, causal context).
2. godot-cpp GDExtension node: push mic/PCM chunks → `PackedFloat32Array` weights.
3. Drop into a lipsync-style scene calling `set_visemes`.

Until then:

```bash
python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx
```

## Generate a fresh smoke ONNX

```bash
nix develop .#train
python3 scripts/train_viseme_smoke.py --subset ci-fixture --export-onnx export/ci-smoke/model.onnx
```
