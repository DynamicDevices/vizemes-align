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

## Godot 4.3

Open `godot-demo/`:

| Scene | Expect |
|-------|--------|
| `csv_smoke.tscn` | `GODOT_ONNX_CSV_SMOKE_OK` |
| `mel_smoke.tscn` | `GODOT_MEL_ONNX_SMOKE_OK` (wav → mel → ONNX) |

Host smoke (no Godot): `scons smoke-csv` in godot-onnx-loader; `make smoke-csv` in gdextension.

## Note

Monolithic `VizemesOnnx` (ORT inside vizemes GDExtension) is removed. Split:
**MelFrontend** (PCM → tensor) + **OnnxLoader** (tensor → logits).
