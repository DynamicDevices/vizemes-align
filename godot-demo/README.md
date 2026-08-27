# godot-demo — shared OnnxLoader + Vizemes MelFrontend

Uses **godot-onnx-loader** and **vizemes MelFrontend** via symlinks (not vendored).

## Layout (sibling repos)

```text
/data_drive/julian/godot-onnx-loader/   # or ~/work/godot-onnx-loader
/data_drive/julian/vizemes-align/
  gdextension/                          # MelFrontend GDExtension (scons)
  godot-demo/
    addons/onnx_loader -> ../../../godot-onnx-loader/addons/onnx_loader
    addons/vizemes_mel -> ../../gdextension/godot
  export/ci-smoke/                      # ONNX + CSV + ci-fixture.wav
```

## Build addons

```bash
cd ../godot-onnx-loader
nix develop   # or export ORT_ROOT=...
scons platform=linux target=template_debug

cd ../gdextension
git submodule update --init --recursive
scons platform=linux target=template_debug
```

## Godot 4.5+

Open `godot-demo/`:

| Scene | Expect |
|-------|--------|
| `csv_smoke.tscn` | `GODOT_ONNX_CSV_SMOKE_OK` |
| `mel_smoke.tscn` | `GODOT_MEL_ONNX_SMOKE_OK` + `hit_rate=` (wav → mel → ONNX vs demo_inputs) |
| `lipsync_smoke.tscn` | `GODOT_LIPSYNC_SMOKE_OK` (mel → ONNX → OVR → `VisemeSystemStub.set_visemes`) |
| `streaming_smoke.tscn` | `GODOT_STREAMING_SMOKE_OK` (chunked `push_pcm_contexts` → ONNX) |
| `mic_lipsync.tscn` | Live mic GUI demo (not headless) — `AudioStreamMicrophone` → streaming mel → visemes |

Host smoke (no Godot): `scons smoke-csv` in godot-onnx-loader; `make smoke-csv` in gdextension.

Headless Godot (both GDExtensions): `bash godot-demo/tools/godot_mel_smoke.sh`

NixOS CI (`nixos-self-hosted`) runs the same script after building MelFrontend + cloning/building godot-onnx-loader.

### Live mic (`mic_lipsync.tscn`)

Open in Godot (not headless): captures mic via `AudioEffectCapture`, resamples to 16 kHz,
feeds `push_pcm_contexts`, drives `VisemeSystemStub`. Swap stub for goatchurchprime/lipsync
`VisemeSystem` in production.

## Note

Monolithic `VizemesOnnx` (ORT inside vizemes GDExtension) is removed. Split:
**MelFrontend** (PCM → tensor) + **OnnxLoader** (tensor → logits).
