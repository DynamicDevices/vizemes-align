# godot-demo — shared OnnxLoader dev project

Uses the **godot-onnx-loader** addon via symlink (not vendored in vizemes git).

## Layout (sibling repos)

```text
/data_drive/julian/godot-onnx-loader/   # or ~/work/godot-onnx-loader
/data_drive/julian/vizemes-align/
  godot-demo/
    addons/onnx_loader -> ../../../godot-onnx-loader/addons/onnx_loader
  export/ci-smoke/                      # ONNX + CSV probes (this repo)
```

## Build addon

```bash
cd ../godot-onnx-loader
nix develop   # or export ORT_ROOT=...
scons platform=linux target=template_debug
```

## Godot 4.3

Open `godot-demo/` → run `csv_smoke.tscn` → expect `GODOT_ONNX_CSV_SMOKE_OK`.

Host smoke (no Godot): `scons smoke-csv` in godot-onnx-loader repo.

## Note

VizemesOnnx (bespoke ORT GDExtension under `gdextension/`) has been removed;
Godot inference uses the shared loader only. Host C smokes (`make smoke-csv`)
remain for mel/ORT parity during frontend work.
