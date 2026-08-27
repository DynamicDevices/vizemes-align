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
| `mic_lipsync.tscn` | Live mic GUI demo (not headless) — `AudioStreamMicrophone` → streaming mel → visemes |

Host smoke (no Godot): `scons smoke-csv` in godot-onnx-loader; `make smoke-csv` in gdextension.

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

### Seek probe (`seek_probe.tscn`) — editor-friendly

Open `seek_probe.tscn` in Godot and run (F6). Output panel shows for each seek:

- expected viseme (from MFA alignment / training labels)
- got viseme (ONNX argmax)
- mel L2 vs the training-path context (`mel_features_c` → causal window)

Regenerate the probe JSON (ci-fixture by default; use a LibriSpeech clip when aligned):

```bash
python3 scripts/export_seek_probe.py --subset ci-fixture
# later, with MFA done:
python3 scripts/export_seek_probe.py --subset test-clean --stem <utt_id> --seeks 8
```

Headless marker: `GODOT_SEEK_PROBE_OK` (also run by `godot_mel_smoke.sh`).

### Live mic (`mic_lipsync.tscn`)

Open in Godot (not headless): captures mic via `AudioEffectCapture`, resamples to 16 kHz,
feeds `push_pcm_contexts`, drives `VisemeSystem` when present else `VisemeSystemStub`.

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
