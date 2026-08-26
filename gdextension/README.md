# Godot GDExtension (in progress)

**Status:** C runtime + mel frontend + **ORT** when `ORT_ROOT` is set.
Host smokes: `make smoke` / `make smoke-csv`. Godot 4.3 GDExtension:
`scons` → `bin/libvizemes_onnx.*.so` + `demo/` CSV scene (needs Godot editor
to run; host has no Godot binary).

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
| `godot/` | `VizemesOnnx` GDExtension sources |
| `SConstruct` | Builds `bin/libvizemes_onnx.*.so` via godot-cpp |
| `demo/` | Minimal Godot 4.3 project (CSV smoke scene) |
| `godot-cpp/` | Submodule (branch 4.3) |
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
nix develop .#train --command bash -c 'cd gdextension && make smoke && make smoke-csv'
# Laptop without Nix — official CPU tarball:
#   curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
#   tar -C /tmp -xzf /tmp/ort.tgz
#   make ORT_ROOT=/tmp/onnxruntime-linux-x64-1.20.1 smoke smoke-csv
```

`make` uses `-fPIC` so `libvizemes_runtime.a` can link into the Godot `.so`.

## Godot GDExtension (.so)

See [godot/README.md](godot/README.md). Short form:

```bash
git submodule update --init --recursive
cd gdextension
make ORT_ROOT="$ORT_ROOT"
scons platform=linux target=template_debug
cp bin/libvizemes_onnx.linux.template_debug.x86_64.so demo/bin/
# Open demo/ in Godot 4.3+
```

## Generate a fresh smoke ONNX

```bash
nix develop .#train
python3 scripts/train_viseme_smoke.py --subset ci-fixture --export-onnx export/ci-smoke/model.onnx
```
