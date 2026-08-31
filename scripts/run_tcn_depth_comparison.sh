#!/usr/bin/env bash
# Reproducible 3-block (29 hops) vs 5-block (125 hops) causal TCN comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBSET="${SUBSET:-test-clean}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-20260831}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-$ROOT/export/tcn-depth-comparison-seed-$SEED}"

if [[ -e "$OUT_ROOT" ]]; then
	echo "refusing to overwrite existing comparison output: $OUT_ROOT" >&2
	echo "set OUT_ROOT to a fresh directory" >&2
	exit 2
fi

common=(
	--subset "$SUBSET"
	--steps "$STEPS"
	--seed "$SEED"
	--device "$DEVICE"
	--channels 128
	--kernel 3
	--dropout 0.1
	--lr 3e-4
	--batch-utterances 8
	--lookahead-ms 50
	--blend-ms 60
)

cd "$ROOT"
python3 scripts/train_viseme_tcn.py "${common[@]}" \
	--layers 3 --out-dir "$OUT_ROOT/three-block"
python3 scripts/train_viseme_tcn.py "${common[@]}" \
	--layers 5 --out-dir "$OUT_ROOT/five-block"
python3 scripts/report_tcn_depth_comparison.py "$OUT_ROOT"
