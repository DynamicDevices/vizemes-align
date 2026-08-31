# Vizemes Mel + Onnx — Julian plan (msgs 821–830)

SoT for next engineering slice after quality UI. Rev A — 2026-08-29.

## Goals (in order)

1. **Ship-layout** — `addons/vizemes_mel/` as the Godot ship unit (like onnx_loader), sources *outside* the addon.
2. **MelFrontend API** — GDScript-owned config; stream queue API; mic stereo ingest + resample.
3. **Minimal mic demo** — few lines: mic → contexts → print numbers.
4. **Train scale-up** — better ONNX on more data; compute budget on Alex CI.
5. **Perf harness** — CPU + memory for `vizemes_mel` and `onnx_loader`.

AudioEffect Capture-style hook is **next** after (2)–(3), not this slice.

---

## 1. Network shape (current smoke) — answers mid=821

| Item | Value |
|------|--------|
| Type | 2-layer MLP |
| Input | flattened causal mel context `context_frames × n_mels` = **20 × 80 = 1600** |
| Hidden | **64** ReLU |
| Output | **15** viseme logits |
| Loss | CrossEntropy; Adam lr=1e-2 |
| Hop | 10 ms (160 @ 16 kHz) |
| Train recipe | `train_viseme_smoke.py` — CI often **40 epochs**, subset `ci-fixture` |
| ONNX size | ~400 KB (`export/ci-smoke/model.onnx`) |
| Convergence | Smoke val accuracy logged every 10 epochs; fixture is toy — **not** production fit |

### Bigger model / more data (proposal)

| Tier | Data | Hidden / ctx | Where | Rough compute |
|------|------|--------------|-------|----------------|
| A (now) | ci-fixture | 64 / 20 | GitHub ubuntu + Nix self-hosted | **minutes** CPU |
| B | test-clean (~subset) | 128–256 / 20–40 | Alex spare CI (Nix / Alien GPU if torch-cuda) | **tens of minutes–few hours** CPU; **minutes** GPU |
| C | fuller LibriSpeech aligned | 256–512 + optional TCN | scheduled CI / Alien | **hours** — need explicit budget + early-stop |

Accountables before kicking B/C: wall-clock + CPU-hours per run, peak RAM, ONNX size, val accuracy vs MFA boxes on held-out stems.

---

## 2. Directory layout — msgs 822–824

**Target (vizemes-align):**

```text
repo/
  addons/vizemes_mel/          # SHIP ONLY: .gdextension + bin/ (+ README)
  gdextension/                 # BUILD: src/, include/, frontends/, SConstruct, godot-cpp/
  godot-demo/addons/vizemes_mel → ../../addons/vizemes_mel
```

Move today’s `gdextension/godot/` → `addons/vizemes_mel/`. SConstruct writes binaries into `addons/vizemes_mel/bin/`.

**godot-onnx-loader (separate PR):** Julian: `addons/onnx_loader/` should not carry `src/` for consumers — ship unit = `.gdextension` + `bin/` (+ deps); sources live at repo `src/`. Track as follow-up in that repo.

---

## 3. MelFrontend API — msgs 825–828

### 3a configure (msg 826)

- **Done:** dropped hand-rolled `sidecar_json.c`. Python `emit_model_meta.py`
  writes flat `model.meta` from canonical ONNX metadata; C loads `key=value`.
  Godot reads the same metadata through OnnxLoader.
- Prefer:
  ```gdscript
  mel.configure(context_frames, n_mels, sample_rate, hop, n_fft, win, fmin, fmax, …)
  ```
  GDScript loads `res://…/model.onnx`, then passes its embedded fields.
- `configure_from_json` and model sidecar fallbacks have been removed.

### 3b mic ingest (msg 827)

- Native **resampler** — SpeexDSP (`godot-speexdsp` + MelFrontend mic path); ear-validated, not linear interp.
- **AGC / VAD / denoise** — `MelFrontend.configure_preprocess(...)` (Speex preprocess); optional `gate_on_vad`.
- `push_pcm_stereo(PackedVector2Array frames, int mix_rate)` → mono + resample + queue.

### 3c stream queue (msg 828)

Replace “return Array of all new contexts” with pull API:

| Method | Role |
|--------|------|
| `begin_stream()` | clear buffers |
| `push_pcm(PackedFloat32Array)` | append mono 16 kHz PCM |
| `count_available_contexts() -> int` | ready contexts |
| `get_next_context() -> PackedFloat32Array` | pop one flattened context |
| `last_context_time_offset() -> float` | seconds from newest PCM front back to last emitted context |

Deprecate / remove `build_utterance_contexts` and `push_pcm_contexts` once demos use the queue (batch = push all + drain).

Prepares **AudioEffect** that emits contexts like `AudioEffectCapture` (next change).

---

## 4. Minimal mic example — msg 829

`godot-demo/mic_context_print.tscn` + short `.gd`:

1. enable input
2. `configure(…)` from loaded ONNX metadata (`res://`)
3. `begin_stream()`
4. each frame: `get_input_frames` → `push_pcm_stereo` → while `count_available_contexts()`: print RMS / argmax-ish stats of context

Shout into mic → numbers move. No face / timeline required.

---

## 5. Perf plan — msg 830

Measure **both** plugins under the same harness:

| Metric | How |
|--------|-----|
| CPU | `python3 gdextension/tools/bench_plugins.py` → `MEL_CPU_US` / `ORT_CPU_US` |
| Memory | `RSS_KB` from same script |
| Realtime factor | `ORT_RT_FACTOR` vs 10 ms hop budget |

Improve only after numbers exist (no blind “make faster”).

### Train tier B quote (2026-08-29)

| Item | Estimate |
|------|----------|
| Data | `data/aligned/test-clean` ≈ **2620** TextGrids; prepared audio ~610 MB |
| Model | hidden **128–256**, context **20**, 40–80 epochs |
| Host | Alex Nix self-hosted / Alien GPU |
| Wall-clock (CPU) | **30–120 min** typical for tier B; GPU **5–20 min** if CUDA torch |
| Peak RAM | **4–12 GB** tensors+torch (subset if OOM) |
| Gate | Do **not** start until Alex confirms budget; log val accuracy vs MFA boxes |

---

## 6. Execution order (this chat)

1. Plan doc (this file) + Telegram ack covering 821–830  
2. Layout move `addons/vizemes_mel`  
3. `configure(...)` + GDScript JSON; stream queue + time offset  
4. Stereo mic + resampler  
5. Minimal mic print demo  
6. Perf harness stub  
7. Train tier B proposal with compute quote (separate MCQ before burning CI hours)

---

## Out of scope here

- AssetLib submit (held for Julian)  
- Fancy hard-encode (Julian: models first)  
- Mel-as-AudioEffect (explicitly “next change”)
