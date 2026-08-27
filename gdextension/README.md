# C audio runtime + Godot inference via shared loader

**Status:** Host C runtime (mel frontend + ORT when `ORT_ROOT` set). **Godot
mel → tensor** via `MelFrontend` GDExtension (`scons`). **Godot inference** uses
the shared [godot-onnx-loader](https://github.com/DynamicDevices/godot-onnx-loader)
addon — not a bespoke ORT GDExtension in this repo.

OpenLipSync’s C mel is a **starting frontend**, not a compatibility lock.
Waveform→features stays swappable (mel, LPC, …). Each exported ONNX keeps its
own sidecar contract (`export/ci-smoke/model.json`).

## Layout

| Path | Role |
|------|------|
| `include/feature_frontend.h` | Pluggable frontend ops |
| `include/viseme_runtime.h` | PCM → viseme weights (C API) |
| `include/sidecar_json.h` | Tiny `model.json` reader (no cJSON) |
| `frontends/mel/` | First impl (vendored mel C + ORIGIN.md) |
| `src/mel_frontend.c` | Mel → `VizemesFrontendOps` |
| `src/viseme_runtime.c` | Host ORT runtime (`ORT_ROOT` required) |
| `src/viseme_runtime_stub.c` | API stub when built without ORT |
| `tools/smoke_context.c` | Host smoke: one flat mel context → softmax |
| `tools/smoke_csv.c` | Host smoke: all `demo_inputs.csv` rows (≡ Python table) |
| `../godot-demo/` | Godot 4.5+ dev project — **OnnxLoader** + **MelFrontend** + smokes |
| `../export/ci-smoke/` | Smoke `model.onnx` + `model.json` + demo inputs |
| `../scripts/sanity_check_onnx.py` | Python ORT harness |

Removed: `VizemesOnnx` GDExtension (monolithic ORT in this repo). ONNX in
Godot is delegated to **godot-onnx-loader**; this tree keeps the audio/mel C
path plus **MelFrontend** (PCM → flat mel context for `OnnxLoader.predict`).

## Godot MelFrontend

```bash
cd gdextension
git submodule update --init --recursive
scons platform=linux target=template_debug
# .so → gdextension/godot/bin/libvizemes_mel.linux.template_debug.x86_64.so
```

Open `../godot-demo/` → run `mel_smoke.tscn` → expect `GODOT_MEL_ONNX_SMOKE_OK`
(wav → mel context → OnnxLoader). Requires both addons built (OnnxLoader + MelFrontend).

**Streaming vs batch:** `push_pcm_contexts` accumulates PCM and runs the same batch mel +
per-utterance normalize as `build_utterance_contexts` (chunked vs one-shot only).

## Target Godot shape ([goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync))

Demo drives Ready Player Me blendshapes via `VisemeSystem.set_visemes(vv)` with
OVR-style names (`sil`, `PP`, …, `U`, optional `LA`). Smoke MLP: 15 classes;
map `silence`↔`sil`, `ih`/`oh`/`ou`↔`I`/`O`/`U` as needed. Godot helpers:
`godot-demo/viseme_utils.gd`, `viseme_system_stub.gd`, `lipsync_smoke.tscn`.

## Host build (C runtime)

```bash
cd gdextension
make                    # stub archive (no ORT)
# Preferred (Hydra-cached C lib on nixos-25.11):
nix develop .#train --command bash -c 'cd gdextension && make smoke && make smoke-csv'
# Laptop without Nix — official CPU tarball:
#   curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
#   tar -C /tmp -xzf /tmp/ort.tgz
#   make ORT_ROOT=/tmp/onnxruntime-linux-x64-1.20.1 smoke smoke-csv
```

## Godot (shared OnnxLoader)

See [`../godot-demo/README.md`](../godot-demo/README.md):

```bash
git clone --recurse-submodules https://github.com/DynamicDevices/godot-onnx-loader.git
cd godot-onnx-loader && nix develop && scons platform=linux target=template_debug
Open `../godot-demo/` in Godot 4.5+ → `mel_smoke.tscn` / `csv_smoke.tscn`.
```

Host CSV parity (no Godot): `scons smoke-csv` in **godot-onnx-loader**.

## Generate a fresh smoke ONNX

```bash
nix develop .#train
python3 scripts/train_viseme_smoke.py --subset ci-fixture --export-onnx export/ci-smoke/model.onnx
```
