# Godot 4 GDExtension

Wraps `vizemes_runtime_*` as `VizemesOnnx`:

| Method | Maps to |
|--------|---------|
| `load_model(json, onnx)` | `vizemes_runtime_create` |
| `predict(PackedFloat32Array)` | `vizemes_runtime_run_context` |
| `get_n_visemes` / `get_input_features` | C accessors |

## Build (Linux x86_64, Godot 4.3)

```bash
git submodule update --init --recursive   # gdextension/godot-cpp @ 4.3
nix develop .#train                       # ORT_ROOT + scons/cmake/ninja/autotools
cd gdextension
scons platform=linux target=template_debug
scons smoke-csv                           # optional: CSV parity table
cp bin/libvizemes_onnx.linux.template_debug.x86_64.so demo/bin/
```

Open `gdextension/demo/` in Godot 4.3+ and run the main scene — prints the
CSV expect/argmax table (`GODOT_ONNX_CSV_SMOKE_OK`).

Host parity without Godot:

```bash
scons smoke-csv
```

Multi-platform layout follows patterns from
[godot_debug_draw_3d](https://github.com/DmitriySalnikov/godot_debug_draw_3d/blob/master/SConstruct)
(next step).
