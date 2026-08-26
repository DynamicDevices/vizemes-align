# Godot 4 GDExtension (scaffold)

Wraps `vizemes_runtime_*` as `VizemesOnnx`:

| Method | Maps to |
|--------|---------|
| `load_model(json, onnx)` | `vizemes_runtime_create` |
| `predict(PackedFloat32Array)` | `vizemes_runtime_run_context` |
| `get_n_visemes` / `get_input_features` | C accessors |

**Status:** sources present; **not built in CI yet**. Needs a checked-out
[godot-cpp](https://github.com/godotengine/godot-cpp) matching your Godot minor
(4.2+), plus `ORT_ROOT`, then an SConstruct (or CMake) that links
`libvizemes_runtime.a` + `libonnxruntime`.

Until that lands, host parity is:

```bash
cd gdextension && make ORT_ROOT=… smoke-csv
```

Same CSV windows / expect↔predict table as `scripts/sanity_check_onnx.py`.

## Intended GDScript smoke (after `.so` exists)

```gdscript
var m := VizemesOnnx.new()
assert(m.load_model("res://export/ci-smoke/model.json", "res://export/ci-smoke/model.onnx"))
var ctx := PackedFloat32Array() # 1600 floats from one CSV row
var w := m.predict(ctx)
print(w.size(), " ", w[0])
```
