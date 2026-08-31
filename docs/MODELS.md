# Using pre-built viseme models

How to download and experiment with models already trained for Vizemes
(MelFrontend + OnnxLoader). Training from scratch: [TRAINING.md](TRAINING.md).

## What a “model” is

Each checkpoint is one self-describing file:

| File | Role |
|------|------|
| `*.onnx` | graph plus embedded model/audio/normalization/vocabulary/latency/provenance contract |

Godot loads the ONNX and configures MelFrontend from its `vizemes_*`
`metadata_props`. Inspect them with:

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
export/ci-smoke/model.onnx          # small smoke / CI
export/tier-b/model_10m.onnx        # ~10 wall-minutes
export/tier-b/model_20m.onnx
export/tier-b/model_final.onnx      # end of timed run (also model.onnx)
```

## Which model to try

| Bundle | Intent | Notes |
|--------|--------|-------|
| `ci-smoke` | Loader / mel smoke | Tiny; not for lip quality |
| `tier-b` `10m` / `20m` / `final` | Longer Libri `test-clean` train | Soft boundaries + 50 ms lookahead; still early quality (~20–35% MFA frame hit-rate on held-out stems) |

A/B in the timeline UI: primary = final, secondary = 10m (or final vs smoke via
`export/tier-b/viseme_timeline_vs_smoke.json`).

## Feed the right audio

From the embedded ONNX `audio` block:

- mono float PCM  
- `sample_rate` (16000)  
- mel: `n_mels`, `n_fft`, `window_length_samples`, `hop_length_samples`, `fmin`, `fmax`  
- ONNX input length = `context_frames * n_mels` (e.g. 20×80 = 1600)

### Live Mel path

`MelFrontend.push_pcm()` is causal and bounded: it performs one Mel transform
for each newly available hop, keeps only one analysis window of PCM, and keeps
only the model's `context_frames` of Mel vectors before flattening them for the
MLP. Its per-bin normalization is running mean/std over frames received so far.
This is necessarily different from the offline per-utterance normalization used
for timeline export and training, which needs the full utterance. The load log
prints the window, hop, overlap, tensor shape, and (for a TCN) its receptive
history so the selected model's live contract is visible.

Latency layers (see `latency` in embedded metadata):

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

## Migrating a legacy checkpoint

```bash
python3 scripts/embed_model_metadata.py legacy/model_final.onnx \
  --hit-rate export/tier-b/hit_rate_summary.json
```

This one-time compatibility tool consumes the legacy sibling JSON, verifies the
saved ONNX, and then the sidecar can be removed. New trainers embed directly.
