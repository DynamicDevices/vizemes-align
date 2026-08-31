# Training viseme models

How we train the mel→viseme ONNX models used by MelFrontend + OnnxLoader.
For downloading and running already-built checkpoints, see [MODELS.md](MODELS.md).

## Overview

1. LibriSpeech subset → wav + transcript  
2. Montreal Forced Aligner (MFA) → phone TextGrids  
3. Phonemes → 15 coded visemes (`configs/viseme_map_*.json`)  
4. Mel tensors (`scripts/build_train_tensors.py`)  
5. Train MLP / TCN → self-describing ONNX

Godot never trains; it loads the graph and Mel contract from the same ONNX.

## Environment

**NixOS / flake**

```bash
git pull
nix develop .#train
python3 scripts/build_train_tensors.py --subset test-clean
```

Train uses `nixpkgs-train` (`nixos-25.11`) so torch/triton come from the Hydra
cache. Non-NixOS / CI: `pip install -r requirements.txt` and optionally
`requirements-train.txt`.

**MFA on NixOS** — do **not** use nixpkgs `micromamba` (`.mamba-wrapped`). Prefer:

```bash
./scripts/bootstrap_mfa_micromamba.sh
./scripts/run_mfa.sh test-clean
```

One-time MFA env (upstream micromamba):

```bash
micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
micromamba run -n mfa mfa model download acoustic english_us_arpa
micromamba run -n mfa mfa model download dictionary english_us_arpa
micromamba run -n mfa mfa model download g2p english_us_arpa
```

## Data pipeline (small subset)

```bash
python scripts/download_librispeech.py --dataset test-clean
python scripts/prepare_corpus.py --subset test-clean
./scripts/run_mfa.sh test-clean
python scripts/export_godot_package.py --subset test-clean
# or: ./scripts/pipeline.sh test-clean
```

Tensors for training:

```bash
python3 scripts/build_train_tensors.py --subset test-clean
```

## Smoke train (CI fixture)

Fast, small model for loader/mel smoke:

```bash
python3 scripts/train_viseme_smoke.py \
  --subset test-clean --context 20 \
  --export-onnx export/ci-smoke/model.onnx
```

Writes `model.onnx` with embedded Mel parameters, vocabulary and provenance.

## Tier-B train (longer Libri run)

Soft phone boundaries + causal mel lookahead (defaults: 60 ms blend, 50 ms lag):

```bash
python3 scripts/train_viseme_tier_b.py \
  --subset test-clean \
  --out-dir export/tier-b \
  --hours 1
```

Timed checkpoints: `model_10m.onnx`, `model_20m.onnx`, `model_final.onnx`.
Convergence log/plot: `convergence.csv` / `convergence.png`.

## Scoring quality

Training `val_acc` under soft labels is a weak signal. Prefer MFA vs predict:

```bash
python3 scripts/eval_tier_b_hit_rate.py \
  --stems 8555-284447-0002 8463-287645-0003
```

Eye-check curves vs MFA boxes in Godot:

```bash
python3 scripts/export_viseme_timeline.py \
  --subset test-clean --stem 8555-284447-0002 \
  --onnx export/tier-b/model_final.onnx \
  --onnx-b export/tier-b/model_10m.onnx \
  --out export/tier-b/viseme_timeline.json
# open godot-demo/viseme_timeline.tscn
```

## Single-clip capacity check

This deliberately trains and scores on the same aligned clip. It answers whether
the architecture can reproduce the simple case; it is **not** a quality or
generalisation result. The trainer stores the fit accuracy in the ONNX metadata.

```bash
STEM=1089-134686-0000
python3 scripts/train_viseme_tcn.py \
  --subset test-clean --overfit-stem "$STEM" --dropout 0 \
  --wall-seconds 60 --out-dir export/one-stem-overfit
python3 scripts/export_viseme_timeline.py \
  --subset test-clean --stem "$STEM" \
  --onnx export/ci-smoke/model.onnx \
  --onnx-b export/one-stem-overfit/model_final.onnx \
  --label-a "A: ci-smoke baseline" --label-b "B: one-clip fit" \
  --out export/one-stem-overfit/viseme_timeline.json
bash scripts/sync_vizeme_onnxmodels.sh
```

Open the previewer with the pack as Model B:

```bash
export VISEMES_MODEL_B_DIR="$PWD/godot-demo/addons/vizeme-onnxmodels/one-stem-overfit"
export VISEMES_TIMELINE_JSON="$PWD/godot-demo/addons/vizeme-onnxmodels/one-stem-overfit/viseme_timeline.json"
godot4 --editor --path "$PWD/godot-demo"
```

In `viseme_timeline.tscn`, select `Model B` as the primary source and compare
it with the ground-truth timeline or Model A. The one-clip model is expected to
fail on any other clip.

## Phone duration evidence

Summarise MFA-aligned phone spans and make the distribution plot Julian asked
for:

```bash
python3 scripts/report_phone_durations.py --subset test-clean
```

This writes `export/phone-duration-report/README.md`, `phone_durations.csv`,
`summary.json`, and `phone_duration_distributions.png`. The report is evidence
for transition-policy experiments; it does not measure visual mouth movement.

## Metadata on export

New trainers embed schema-2 metadata during export. For old ONNX+JSON pairs only:

```bash
python3 scripts/embed_model_metadata.py export/tier-b/model_*.onnx
```

The migration tool embeds and verifies `vizemes_*` metadata. Remove the legacy
sidecar after verification; CI rejects model JSON in shipped packs.

## Layout reminders

| Path | Role |
|------|------|
| `data/raw/` | LibriSpeech |
| `data/prepared/<subset>/` | MFA corpus |
| `data/tensors/<subset>/` | Train windows |
| `export/ci-smoke/` | Smoke ONNX |
| `export/tier-b/` | Longer train + timelines |
| `configs/viseme_map_*.json` | Phone → viseme maps |
