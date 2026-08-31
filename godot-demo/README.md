# godot-demo — shared OnnxLoader + Vizemes MelFrontend

Uses **godot-onnx-loader** and **vizemes MelFrontend** via symlinks (not vendored).

## Layout (sibling repos)

```text
/data_drive/julian/godot-onnx-loader/   # or ~/work/godot-onnx-loader
/data_drive/julian/vizemes-align/
  gdextension/                          # MelFrontend GDExtension (scons)
  godot-demo/
    addons/onnx_loader -> ../../../godot-onnx-loader/addons/onnx_loader
    addons/vizemes_mel -> ../../addons/vizemes_mel
    addons/vizeme-onnxmodels/           # ONNX packs + wav fixtures (res:// only)
  export/                               # training outputs; sync into the addon
```

Runtime stays inside the Godot project (`res://addons/vizeme-onnxmodels`). After
training or downloading models:

```bash
bash scripts/sync_vizeme_onnxmodels.sh
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

## Godot 4.6+

Open `godot-demo/` (minimum **Godot 4.6** — MelFrontend `compatibility_minimum` and
project features are 4.6). On Nix, prefer sibling `godot-onnx-loader` Godot 4.6 MS-ORT
path (`bash tools/godot_46_ms_ort.sh`) or a stock 4.6 editor binary.

| Scene | Expect |
|-------|--------|
| `csv_smoke.tscn` | `GODOT_ONNX_CSV_SMOKE_OK` |
| `mel_smoke.tscn` | `GODOT_MEL_ONNX_SMOKE_OK` + `hit_rate=` (wav → mel → ONNX vs demo_inputs) |
| `lipsync_smoke.tscn` | `GODOT_LIPSYNC_SMOKE_OK` (mel → ONNX → OVR → VisemeSystem or Stub) |
| `streaming_smoke.tscn` | `GODOT_STREAMING_SMOKE_OK` (chunked `push_pcm_contexts` → ONNX) |
| `seek_probe.tscn` | `GODOT_SEEK_PROBE_OK` — seek times vs alignment; mel L2 vs training path (editor OK) |
| `viseme_timeline.tscn` | Quality UI: top face (mouth zoom + orbit + blend bars) + graph / **Record 3s** (editor F6) |
| `mic_lipsync.tscn` | Live mic GUI demo (not headless) — Godot 4.6 `AudioServer.get_input_frames` → streaming mel → visemes |

Host smoke (no Godot): `scons smoke-csv` in godot-onnx-loader; `make smoke-csv` in gdextension.

GDScript editor parse gate (Godot `--check-only`, catches `:=` inference footguns):

```bash
export GODOT_BIN=~/Downloads/Godot_v4.6.1-stable_linux.x86_64   # or nix godot_4_6
bash godot-demo/tools/gdscript_check_only.sh
```

Headless Godot (both GDExtensions): `bash godot-demo/tools/godot_mel_smoke.sh`

**Julian Nix one-liner** (builds MelFrontend + OnnxLoader via Godot **4.6** MS-ORT, then smokes):

```bash
bash godot-demo/tools/julian_vizemes_smoke.sh
```

That script calls sibling `godot-onnx-loader/tools/godot_46_ms_ort.sh` (the path that
already prints `GODOT_46_MS_ORT_SMOKE_OK`). Do **not** rely on bare
`nix develop` `GODOT_BIN` from godot-onnx-loader for vizemes — that shell still
pins nixpkgs Godot **4.5.1**.

NixOS CI (`nixos-self-hosted`) runs the same script after building MelFrontend + cloning/building godot-onnx-loader.

### Editor (Godot 4.6)

From `vizemes-align` root, with sibling `godot-onnx-loader` checked out and built
(`bash tools/godot_46_ms_ort.sh` there). Symlink targets are relative to
`godot-demo/addons/`, so use three `../` segments (not `../godot-onnx-loader`
from the repo root):

```bash
ln -sfn ../../../godot-onnx-loader/addons/onnx_loader godot-demo/addons/onnx_loader
# MelFrontend (usually already present in the tree):
#   godot-demo/addons/vizemes_mel -> ../../addons/vizemes_mel

nix shell github:nixos/nixpkgs/nixos-26.05#godot_4_6 --command bash -c '
  unset ONNX_ORT_BIN
  # ReleaseSession skip defaults on in recent onnx-loader; =1 still fine.
  export ONNX_LOADER_SKIP_SESSION_RELEASE=1
  godot4 --editor --path '"$PWD"'/godot-demo
'
```

Then open `viseme_timeline.tscn` (F6) — primary quality UI — or `seek_probe.tscn` for the table.

### Quality UI (Julian) — start here

**Primary:** `viseme_timeline.tscn` (project main scene). F6 in Godot **4.6+**.

- Type a `test-clean` stem id (default `1320-122617-0010`) → **Load** to re-export MFA boxes + reload.
- **Seek table…** opens expect/got + MEL dumps; **Mic…** opens live capture.
- Wheel zoom, middle-drag pan, left-drag select, Space/P play, Esc clear, R reset.
- A/B toggle models (when a secondary pack is set); D toggles the argmax-disagreement ribbon; **H** toggles the local ID+confidence preview ribbon. That preview byte is not the sparse Opus boundary-event wire contract.
- **T** drives the top face from MFA/training expect boxes (quality ceiling) instead of ONNX A; **[** / **]** adjust crossfade width at box boundaries (default 60 ms).
- **Sample / primary / overlay OptionButtons** (Julian 947): pick a stem or mic recording; primary drives the top face (Ground truth / Model A / Model B / Hidden); overlay is the faded compare series. Ground truth is disabled when the sample has no MFA timeline.

**Companion:** `seek_probe.tscn` — side-by-side expect vs got; full MEL dumps under
`export/debug/seek_mel_<stem>_<t>.json` (written on each run; **Dump MEL** rewrites).

```bash
python3 scripts/export_viseme_timeline.py --subset test-clean --stem 1320-122617-0010
python3 scripts/export_seek_probe.py --subset test-clean --stem 1320-122617-0010 --seeks 8
# CI fixture still available:
python3 scripts/export_seek_probe.py --subset ci-fixture --out export/ci-smoke/seek_probe_ci_fixture.json
```

Headless markers: `GODOT_VISEME_TIMELINE_OK`, `GODOT_SEEK_PROBE_OK` (seek also in `godot_mel_smoke.sh`).

### Direct-viseme vs phone-to-viseme review

The shipped `tier-c` pack is the controlled 600-second phone TCN. Start the
editor with it as Model B; Model A remains the direct `tier-b-tcn` baseline:

```bash
export VISEMES_MODEL_B_DIR="$PWD/godot-demo/addons/vizeme-onnxmodels/tier-c"
godot4 --editor --path "$PWD/godot-demo"
```

Open `viseme_timeline.tscn`, then use the **primary** and **overlay** selectors
to swap Model A and Model B while replaying the same selection. Model B's phone
posterior table, smoothing and viseme vocabulary come entirely from its ONNX
metadata. For a headless load/inference gate, run the scene with the same
environment variable and expect both model manifests plus
`GODOT_VISEME_TIMELINE_OK`.

### Live mic (`mic_lipsync.tscn`)

Open in Godot (not headless): captures mic via `AudioServer.get_input_frames()` /
`get_input_mix_rate()`, resamples to 16 kHz with `VisemeUtils.resample_pcm`,
feeds `push_pcm_contexts`, drives `VisemeSystem` when present else `VisemeSystemStub`.
Also reachable from the timeline/seek **Mic…** buttons.

### Real VisemeSystem (optional)

Default scenes use `VisemeSystemStub`. To drive [goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync)
`VisemeSystem` instead: instance that node as a child named **`VisemeSystem`**
(sibling to or replacing the stub). `VisemeTarget.resolve` prefers it when present.
Weights are OVR order including trailing `LA` (always 0 from our MLP).

### Hit-rate / richer data

CI smokes use `ci-fixture`. For LibriSpeech `test-clean` after MFA + tensors:

```bash
python3 scripts/build_train_tensors.py --subset test-clean
python3 scripts/train_viseme_smoke.py --subset test-clean --context 20 --export-onnx export/ci-smoke/model.onnx
python3 scripts/build_demo_inputs.py --subset test-clean
python3 scripts/export_seek_probe.py --subset test-clean --stem <utt_id> --seeks 8
```

Then re-run `mel_smoke` / `seek_probe` for hit-rate on real speech.

## Note

Monolithic `VizemesOnnx` (ORT inside vizemes GDExtension) is removed. Split:
**MelFrontend** (PCM → tensor) + **OnnxLoader** (tensor → logits).
