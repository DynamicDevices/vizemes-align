# Godot GDExtension (in progress)

**Status:** C runtime + mel frontend + **ORT** when `ORT_ROOT` is set
(nix `.#train` or official release tarball). Host smokes: `make smoke`
(row 0) and `make smoke-csv` (all CSV windows ≡ Python). GDExtension
sources under `godot/` — **not** loadable until godot-cpp build is wired.

OpenLipSync’s C mel is a **starting frontend**, not a compatibility lock.
Waveform→features stays swappable (mel, LPC, …). Each exported ONNX keeps its
own sidecar contract (`export/ci-smoke/model.json`).

## Layout

| Path | Role |
|------|------|
| `include/feature_frontend.h` | Pluggable frontend ops |
| `include/viseme_runtime.h` | PCM → viseme weights (Godot-facing SoT) |
| `include/sidecar_json.h` | Tiny `model.json` reader (no cJSON) |
| `frontends/mel/` | First impl (vendored mel C + ORIGIN.md) |
| `src/mel_frontend.c` | Mel → `VizemesFrontendOps` |
| `src/viseme_runtime.c` | ORT-backed runtime (`ORT_ROOT` required) |
| `src/viseme_runtime_stub.c` | API stub when built without ORT |
| `tools/smoke_context.c` | Host smoke: one flat mel context → softmax |
| `tools/smoke_csv.c` | Host smoke: all `demo_inputs.csv` rows (≡ Python table) |
| `godot/` | GDExtension scaffold (`VizemesOnnx.load_model` / `predict`) |
| `../export/ci-smoke/` | Smoke `model.onnx` + `model.json` + demo inputs |
| `../scripts/sanity_check_onnx.py` | Python ORT harness |

## Target Godot shape ([goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync))

Demo drives Ready Player Me blendshapes via `VisemeSystem.set_visemes(vv)` with
OVR-style names (`sil`, `PP`, …, `U`, optional `LA`). Today that comes from
OVRLipSync through two-voip. Our runtime should emit the same kind of float
weight vector so the avatar path can stay.

Smoke MLP: 15 classes (see `export/ci-smoke/model.json`); map `silence`↔`sil`,
`ih`/`oh`/`ou`↔`I`/`O`/`U` as needed. No laughter channel yet.

## Host build

```bash
cd gdextension
make                    # stub archive (no ORT)
# Preferred (Hydra-cached C lib on nixos-25.11):
nix develop .#train --command bash -c 'cd gdextension && make smoke'
# Laptop without Nix — official CPU tarball:
#   curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
#   tar -C /tmp -xzf /tmp/ort.tgz
#   make ORT_ROOT=/tmp/onnxruntime-linux-x64-1.20.1 smoke
```

`make smoke` dumps `demo_inputs.npz` row 0 → `demo_row0.f32`, runs
`smoke_context`, checks argmax vs label.

## CSV parity (C ↔ Python)

Same windows as `scripts/sanity_check_onnx.py`:

```bash
nix develop .#train --command bash -c 'cd gdextension && make smoke-csv'
```

## Next

1. godot-cpp GDExtension node: `load_model(json, onnx)` + `predict(PackedFloat32Array)` (see `godot/`).
2. Drop into a lipsync-style scene calling `set_visemes`.
3. Push mic/PCM via `vizemes_runtime_push_pcm`.

Python path (unchanged):

```bash
nix develop .#train --command python3 scripts/sanity_check_onnx.py
```

## Generate a fresh smoke ONNX

```bash
nix develop .#train
python3 scripts/train_viseme_smoke.py --subset ci-fixture --export-onnx export/ci-smoke/model.onnx
```
