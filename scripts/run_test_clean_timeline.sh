#!/usr/bin/env bash
# LibriSpeech test-clean → prepare → MFA → tensors → viseme_timeline.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs data/raw
LOG="$ROOT/logs/test-clean-pipeline.log"
exec > >(tee -a "$LOG") 2>&1

# Skip download if already extracted.
if [[ ! -d data/raw/LibriSpeech/test-clean ]]; then
  echo "=== DOWNLOAD $(date -Is) ==="
  python3 scripts/download_librispeech.py --datasets test-clean --keep-archives
else
  echo "=== DOWNLOAD skip (data/raw/LibriSpeech/test-clean present) $(date -Is) ==="
fi

echo "=== PREPARE $(date -Is) ==="
python3 scripts/prepare_corpus.py --subsets test-clean

echo "=== MFA $(date -Is) ==="
bash scripts/run_mfa.sh test-clean

echo "=== TENSORS $(date -Is) ==="
# Prefer flake train Python (3.12); fall back to host python3.
if command -v nix >/dev/null 2>&1; then
  nix develop .#train --command python3 scripts/build_train_tensors.py --subset test-clean
else
  python3 scripts/build_train_tensors.py --subset test-clean
fi

echo "=== PICK STEM + TIMELINE $(date -Is) ==="
python3 - <<'PY'
from pathlib import Path
import wave
prep = Path("data/prepared/test-clean")
aligned = Path("data/aligned/test-clean")
cands = []
for wav in prep.glob("*.wav"):
    if not list(aligned.rglob(f"{wav.stem}.TextGrid")):
        continue
    with wave.open(str(wav), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
    cands.append((abs(dur - 10.0), dur, wav.stem))
cands.sort()
if not cands:
    raise SystemExit("no aligned wavs for timeline pick")
stem = cands[0][2]
Path("logs/timeline-stem.txt").write_text(stem + "\n")
print(f"picked stem={stem} dur={cands[0][1]:.2f}s")
PY

STEM="$(cat logs/timeline-stem.txt)"
python3 scripts/export_viseme_timeline.py --subset test-clean --stem "$STEM" \
  --out export/ci-smoke/viseme_timeline.json
cp -a export/ci-smoke/viseme_timeline.json "export/ci-smoke/viseme_timeline_${STEM}.json"
echo "TEST_CLEAN_TIMELINE_OK stem=$STEM"
echo "=== DONE $(date -Is) ==="
