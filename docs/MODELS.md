# Using pre-built viseme models

How to download and experiment with models already trained for Vizemes
(MelFrontend + OnnxLoader). Training from scratch: [TRAINING.md](TRAINING.md).

## What a “model” is

Each checkpoint is a pair:

| File | Role |
|------|------|
| `*.onnx` | MLP: flattened mel context → 15 viseme logits |
| `*.json` | MelFrontend config + viseme names + latency/quality notes |

Always keep the pair together. Godot loads JSON first (`configure`), then ONNX.

ONNX files also carry `vizemes_*` entries in `metadata_props` (schema, mel hop /
window, lookahead, quality estimates). Inspect with:

```bash
python3 -c "import onnx,sys; m=onnx.load(sys.argv[1]);
print({p.key:p.value[:120] for p in m.metadata_props if p.key.startswith('vizemes_') and p.key!='vizemes_meta_json'})" model.onnx
```

## Download

**GitHub release (preferred for experimenting)**

Releases under [DynamicDevices/vizemes-align](https://github.com/DynamicDevices/vizemes-align/releases)
ship zipped bundles, e.g. `vizemes-models-tier-b.zip`:

```bash
# example — use the tag shown in the release notes
gh release download <tag> -R DynamicDevices/vizemes-align -p 'vizemes-models-*.zip'
unzip vizemes-models-tier-b.zip -d export/
```

**From a clone** (if you already have the tree that trained them):

```text
export/ci-smoke/model.onnx + model.json     # small smoke / CI
export/tier-b/model_10m.onnx|.json          # ~10 wall-minutes
export/tier-b/model_20m.onnx|.json
export/tier-b/model_final.onnx|.json        # end of timed run (also model.onnx)
```

## Which model to try

| Bundle | Intent | Notes |
|--------|--------|-------|
| `ci-smoke` | Loader / mel smoke | Tiny; not for lip quality |
| `tier-b` `10m` / `20m` / `final` | Longer Libri `test-clean` train | Soft boundaries + 50 ms lookahead; still early quality (~20–35% MFA frame hit-rate on held-out stems) |

A/B in the timeline UI: primary = final, secondary = 10m (or final vs smoke via
`export/tier-b/viseme_timeline_vs_smoke.json`).

## Feed the right audio

From the sidecar `audio` block (also mirrored in ONNX metadata):

- mono float PCM  
- `sample_rate` (16000)  
- mel: `n_mels`, `n_fft`, `window_length_samples`, `hop_length_samples`, `fmin`, `fmax`  
- ONNX input length = `context_frames * n_mels` (e.g. 20×80 = 1600)

Latency layers (see `latency` in JSON):

1. **Speex / MelFrontend DSP** — `mel.get_dsp_latency_seconds()` at runtime  
2. **Mel context** — time spanned by `context_frames`  
3. **Label lookahead** — training lag (`lookahead_ms`); compensate in playback if you align to MFA boxes  

## Godot quick path

```bash
cd godot-demo
# Default Load prefers export/tier-b/viseme_timeline.json when present
godot --path . res://viseme_timeline.tscn
```

Or export a stem yourself:

```bash
python3 scripts/export_viseme_timeline.py \
  --subset test-clean --stem 8555-284447-0002 \
  --model-json export/tier-b/model.json \
  --onnx export/tier-b/model_final.onnx \
  --onnx-b export/tier-b/model_10m.onnx \
  --label-a "A:tier-b-final" --label-b "B:tier-b-10m" \
  --out export/tier-b/viseme_timeline.json
```

Mic / live preview uses `_default_model_paths()` (tier-b if staged, else ci-smoke).

## Host / Python check

```bash
python3 scripts/eval_tier_b_hit_rate.py --model-dir export/tier-b
```

## Refreshing embedded metadata

```bash
python3 scripts/embed_model_metadata.py export/tier-b/model_final.onnx \
  --hit-rate export/tier-b/hit_rate_summary.json
```
