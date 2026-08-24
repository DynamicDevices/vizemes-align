# vizemes-align

Download LibriSpeech, force-align with **Montreal Forced Aligner (MFA)**, map phonemes → coded visemes, and export a **Godot-readable** package (audio + time-aligned viseme labels). Spectrogram / ONNX training stays on the Godot feature path (see Telegram Vizemes thread).

Derived from the Dynamic Devices / OpenLipSync training data pipeline.



## MFA on NixOS (important)

`nix develop` must **not** use nixpkgs `micromamba` (it appears as `.mamba-wrapped` and MFA dies).

```bash
git pull
./scripts/run_mfa.sh test-clean
```

`run_mfa.sh` now prefers `~/micromamba/bin/micromamba` and **auto-runs**
`./scripts/bootstrap_mfa_micromamba.sh` if only the broken nix wrapper is present.

## MFA env (one-time)

```bash
micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
micromamba run -n mfa mfa model download acoustic english_us_arpa
micromamba run -n mfa mfa model download dictionary english_us_arpa
micromamba run -n mfa mfa model download g2p english_us_arpa
```

If you see `The given prefix does not exist: ".../envs/mfa"`, the env was never created — run the block above, then `./scripts/run_mfa.sh test-clean` again. Download/prepare need not be re-run.

## NixOS / `nix develop`

```bash
nix develop
# then pipeline as below (MFA still via micromamba inside the shell — see flake shellHook)
```

## Prerequisites

- Python 3.11+
- `ffmpeg`
- `micromamba` env `mfa` with `montreal-forced-aligner`  
  (`micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner`)
- MFA models (US default):  
  `micromamba run -n mfa mfa model download acoustic english_us_arpa`  
  `micromamba run -n mfa mfa model download dictionary english_us_arpa`  
  `micromamba run -n mfa mfa model download g2p english_us_arpa`

## Quick start (small subset)

```bash
cd vizemes-align
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Download + extract LibriSpeech subset
python scripts/download_librispeech.py --dataset test-clean

# 2) Flatten to MFA-ready wav+lab corpus
python scripts/prepare_corpus.py --subset test-clean

# 3) MFA align → TextGrids / JSON
./scripts/run_mfa.sh test-clean

# 4) Phonemes → visemes + Godot package under data/export/
python scripts/export_godot_package.py --subset test-clean
```

Or one shot:

```bash
./scripts/pipeline.sh test-clean
```

## Layout

| Path | Role |
|------|------|
| `data/raw/` | LibriSpeech tarballs / extract |
| `data/prepared/<subset>/` | Flat wav + transcript for MFA |
| `data/cache/` | MFA working dirs |
| `data/export/<subset>/` | Godot-ingestible clips + viseme JSON/CSV |
| `configs/viseme_map_*.json` | Phoneme → 15-viseme maps (US ARPA / UK MFA) |

## Notes

- Start with `test-clean` or `dev-clean` before `train-clean-100`.
- Own-voice fine-tune and noise aug come after this SoT path is solid.
- Default topic for Briar Telegram sends while iterating: `vizemes` (group Alex/Julian/Briar).

## NixOS MFA note

nixpkgs `micromamba` often installs as `.mamba-wrapped` and then fails with
`unknown MAMBA_EXE` / `exec: mfa: not found`. Use the upstream binary instead:

```bash
./scripts/bootstrap_mfa_micromamba.sh
export PATH="$HOME/micromamba/bin:$PATH"
./scripts/run_mfa.sh test-clean
```
