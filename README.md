# vizemes-align

Download LibriSpeech, force-align with **Montreal Forced Aligner (MFA)**, map phonemes → coded visemes, and export a **Godot-readable** package (audio + time-aligned viseme labels).

**Godot minimum: 4.6** (`godot-demo/` + MelFrontend GDExtension). Sibling [godot-onnx-loader](https://github.com/DynamicDevices/godot-onnx-loader) supplies OnnxLoader; on Nix use its Godot 4.6 MS-ORT path when needed.

Derived from the Dynamic Devices / OpenLipSync training data pipeline.

## Docs

| Doc | Contents |
|-----|----------|
| [docs/MODELS.md](docs/MODELS.md) | Download / use pre-built ONNX + sidecar JSON, latency, Godot timeline |
| [docs/TRAINING.md](docs/TRAINING.md) | MFA, tensors, smoke + tier-B train, scoring, embedding metadata |

## Quick start (alignment package)

```bash
cd vizemes-align
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_librispeech.py --dataset test-clean
python scripts/prepare_corpus.py --subset test-clean
./scripts/run_mfa.sh test-clean          # see docs/TRAINING.md for NixOS micromamba
python scripts/export_godot_package.py --subset test-clean
# or: ./scripts/pipeline.sh test-clean
```

## Try a trained model

See [docs/MODELS.md](docs/MODELS.md). Short path once `export/tier-b/` (or a release zip) is present:

```bash
# open godot-demo/viseme_timeline.tscn  — Load prefers tier-b timeline when staged
python3 scripts/eval_tier_b_hit_rate.py --model-dir export/tier-b
```

## Layout

| Path | Role |
|------|------|
| `data/raw/` | LibriSpeech tarballs / extract |
| `data/prepared/<subset>/` | Flat wav + transcript for MFA |
| `data/cache/` | MFA working dirs |
| `data/export/<subset>/` | Godot-ingestible clips + viseme JSON/CSV |
| `export/ci-smoke/` | Smoke ONNX |
| `export/tier-b/` | Longer train checkpoints + timelines |
| `configs/viseme_map_*.json` | Phoneme → 15-viseme maps (US ARPA / UK MFA) |
| `docs/` | Training + model how-to |

## Notes

- Start with `test-clean` or `dev-clean` before `train-clean-100`.
- Own-voice fine-tune and noise aug come after this SoT path is solid.
- Default topic for Briar Telegram sends while iterating: `vizemes` (group Alex/Julian/Briar).
