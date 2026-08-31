# Godot `onnxmodels/` layout

Ship trained ONNX checkpoints with the game under a single project folder the
export template can include.

## Layout

```text
res://onnxmodels/
  tier-b-tcn/
    model_final.onnx
    model_final.json          # until ONNX-only Mel configure is in your build
  tier-b/
    model_final.onnx
    model_10m.onnx
    …
  manifest.json               # optional: default model id, labels for UI
```

Download release zips from GitHub into this tree (or CI copies them at pack time).

## Export

In the Godot export preset, include `onnxmodels/` (Resources → Filters to export
non-resource files / keep folder). Do not put large checkpoints under `addons/`
unless the addon owns them.

## Runtime

```gdscript
var root := "res://onnxmodels/tier-b-tcn"
var onnx := root.path_join("model_final.onnx")
# Prefer OnnxLoader metadata → MelFrontend.configure; JSON fallback while migrating.
```

## Releases

| Tag | Contents |
|-----|----------|
| `v0.2.0-tier-b-models` | MLP tier-B + ci-smoke |
| `v0.2.1-tier-b-tcn` | TCN 10‑min + MLP compare |

See [MODELS.md](MODELS.md).
