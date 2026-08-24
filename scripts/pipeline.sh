#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBSET="${1:-test-clean}"
cd "$ROOT"
python3 scripts/download_librispeech.py --dataset "${SUBSET}"
python3 scripts/prepare_corpus.py --subset "${SUBSET}" || python3 scripts/prepare_corpus.py --subset "${SUBSET//-/_}" || true
./scripts/run_mfa.sh "${SUBSET}"
python3 scripts/export_godot_package.py --subset "${SUBSET}" || python3 scripts/export_godot_package.py --subset "${SUBSET}" --viseme-map configs/viseme_map_en_us_arpa.json
