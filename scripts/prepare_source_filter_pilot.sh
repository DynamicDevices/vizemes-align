#!/usr/bin/env bash
# Restore a balanced LibriSpeech pilot, align once, then build matched A/B/C tensors.
set -euo pipefail

finish() {
	code=$?
	printf 'EXIT_CODE=%s FINISHED=%s\n' "$code" "$(date -Is)"
}
trap finish EXIT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
FULL_SUBSET="test-clean"
PILOT_SUBSET="source-filter-pilot"
RAW="$ROOT/data/raw/LibriSpeech"
PREPARED="$ROOT/data/prepared"
ALIGNED="$ROOT/data/aligned"
TENSORS="$ROOT/data/tensors"

cd "$ROOT"

if [[ ! -d "$RAW/$FULL_SUBSET" ]]; then
	"$PYTHON" scripts/download_librispeech.py --datasets "$FULL_SUBSET" --keep-archives
fi

"$PYTHON" scripts/prepare_corpus.py --subsets "$FULL_SUBSET"
"$PYTHON" scripts/select_speaker_pilot.py \
	--prepared "$PREPARED/$FULL_SUBSET" \
	--speakers "$RAW/SPEAKERS.TXT" \
	--out-dir "$PREPARED/$PILOT_SUBSET" \
	--speaker-count 10

bash scripts/run_mfa.sh "$PILOT_SUBSET"

for frontend in mel lpc-filter lpc-source-filter; do
	"$PYTHON" scripts/build_train_tensors.py \
		--subset "$PILOT_SUBSET" \
		--aligned-dir "$ALIGNED/$PILOT_SUBSET" \
		--out-dir "$TENSORS/$PILOT_SUBSET-$frontend" \
		--frontend "$frontend"
done

echo "SOURCE_FILTER_PILOT_TENSORS_OK"
