# vizeme-onnxmodels

Godot-distributable store for viseme ONNX packs. **All demo / mic / timeline
runtime I/O for models stays under this addon** (`res://addons/vizeme-onnxmodels`).

Julian (Telegram 944): the Godot project must not climb out of `res://` via
`ProjectSettings.globalize_path("res://")/../..` into the monorepo `export/`
tree. Native loaders may `globalize_path` a `res://…` path to open the file on
disk — that still stays inside the project.

## Layout

```text
addons/vizeme-onnxmodels/
  tier-b/           # Mel → viseme MLP (default mic)
  tier-b-tcn/       # sequence TCN when supported
  ci-smoke/         # small CI / fallback pack
  fixtures/         # wav + optional timeline JSON for offline quality UI
```

Each pack should include a `model_final.onnx` (or `model.onnx`) with **embedded**
metadata: mel settings, lag, viseme names, quality/history, date. Sidecar JSON
is optional fallback only.

## Sync from training export

From the repo root (after training / downloading models):

```bash
bash scripts/sync_vizeme_onnxmodels.sh
```

That copies `export/tier-b*`, `export/ci-smoke` ONNX/JSON/timeline JSON and
wav fixtures into this addon.

## Godot export

Include `addons/vizeme-onnxmodels/**` in export filters so shipped builds carry
the models. Future: this addon can grow a downloader / package-manager UI
without changing MelFrontend / OnnxLoader.
