#!/usr/bin/env bash
# Train the matched Mel / LPC-filter / LPC+source comparison sequentially.
set -euo pipefail

finish() {
	code=$?
	printf 'EXIT_CODE=%s FINISHED=%s\n' "$code" "$(date -Is)"
}
trap finish EXIT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SUBSET="source-filter-pilot"
OUT="$ROOT/export/source-filter-pilot"

cd "$ROOT"
mkdir -p "$OUT"

for frontend in mel lpc-filter lpc-source-filter; do
	"$PYTHON" scripts/train_viseme_tcn.py \
		--subset "$SUBSET" \
		--tensor-dir "$ROOT/data/tensors/$SUBSET-$frontend" \
		--split-by-speaker --val-frac 0.2 \
		--layers 5 --kernel 2 --dilations 1,2,2,2,2 \
		--channels 128 --dropout 0.1 \
		--hard-boundaries --lookahead-ms 10 --blend-ms 0 \
		--wall-seconds 900 --seed 20260905 --device cpu \
		--out-dir "$OUT/$frontend"
done

echo "SOURCE_FILTER_PILOT_TRAINING_OK"
