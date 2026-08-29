# Training viseme models

How we train the mel→viseme ONNX models used by MelFrontend + OnnxLoader.
For downloading and running already-built checkpoints, see [MODELS.md](MODELS.md).

## Overview

1. LibriSpeech subset → wav + transcript  
2. Montreal Forced Aligner (MFA) → phone TextGrids  
3. Phonemes → 15 coded visemes (`configs/viseme_map_*.json`)  
4. Mel tensors (`scripts/build_train_tensors.py`)  
5. Train MLP / smoke → ONNX + sidecar JSON  

Godot never trains; it only loads ONNX + mel config.

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

Writes `model.onnx` + `model.json` (mel params, viseme table).

## Tier-B train (longer Libri run)

Soft phone boundaries + causal mel lookahead (defaults: 60 ms blend, 50 ms lag):

```bash
python3 scripts/train_viseme_tier_b.py \
  --subset test-clean \
  --out-dir export/tier-b \
  --hours 1
```

Timed checkpoints: `model_10m.onnx`, `model_20m.onnx`, `model_final.onnx` (+ JSON).
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
  --model-json export/tier-b/model.json \
  --onnx export/tier-b/model_final.onnx \
  --onnx-b export/tier-b/model_10m.onnx \
  --out export/tier-b/viseme_timeline.json
# open godot-demo/viseme_timeline.tscn
```

## Metadata on export

After training (or before shipping a release):

```bash
python3 scripts/embed_model_metadata.py export/tier-b/model_*.onnx
```

Refreshes sidecar JSON (`latency`, `quality`) and embeds `vizemes_*` keys in the
ONNX `metadata_props` so a lone `.onnx` still describes mel shape, lookahead,
and estimated hit-rate. Sidecar JSON remains what MelFrontend.configure reads.

## Layout reminders

| Path | Role |
|------|------|
| `data/raw/` | LibriSpeech |
| `data/prepared/<subset>/` | MFA corpus |
| `data/tensors/<subset>/` | Train windows |
| `export/ci-smoke/` | Smoke ONNX |
| `export/tier-b/` | Longer train + timelines |
| `configs/viseme_map_*.json` | Phone → viseme maps |
