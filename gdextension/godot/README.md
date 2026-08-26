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
# ORT: nix develop .#train, or:
#   curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
#   tar -C /tmp -xzf /tmp/ort.tgz && export ORT_ROOT=/tmp/onnxruntime-linux-x64-1.20.1
cd gdextension
make ORT_ROOT="$ORT_ROOT"                 # -fPIC static lib
pip install scons                         # if needed
scons platform=linux target=template_debug
cp bin/libvizemes_onnx.linux.template_debug.x86_64.so demo/bin/
```

Open `gdextension/demo/` in Godot 4.3+ and run the main scene — prints the
CSV expect/argmax table (`GODOT_ONNX_CSV_SMOKE_OK`).

Host parity without Godot:

```bash
make ORT_ROOT="$ORT_ROOT" smoke-csv
```
