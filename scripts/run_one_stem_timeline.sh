#!/usr/bin/env bash
# Fast path: MFA + tensors + viseme_timeline for one ~10s stem (parallel to full MFA).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEM="${1:-$(cat logs/one-stem.txt 2>/dev/null || true)}"
STEM="${STEM:-1320-122617-0010}"
SUBSET="timeline-demo"
LOG="logs/one-stem-timeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ONE-STEM START stem=$STEM $(date -Is) ==="
mkdir -p "data/prepared/${SUBSET}" "data/cache/mfa_corpus_${SUBSET}" "data/aligned/${SUBSET}" "data/tensors/${SUBSET}"
rm -rf "data/cache/mfa_corpus_${SUBSET:?}/"* "data/aligned/${SUBSET:?}/"*
cp -a "data/prepared/test-clean/${STEM}.wav" "data/prepared/test-clean/${STEM}.lab" "data/prepared/${SUBSET}/"
cp -a "data/prepared/${SUBSET}/." "data/cache/mfa_corpus_${SUBSET}/"

# shellcheck source=resolve_micromamba.sh
source "${ROOT}/scripts/resolve_micromamba.sh"
MM=("${ROOT}/scripts/mamba_nixos.sh")
mmfa() { "${MM[@]}" run -n mfa mfa "$@"; }

echo "=== MFA one-stem $(date -Is) ==="
mmfa align "data/cache/mfa_corpus_${SUBSET}" english_us_arpa english_us_arpa \
  "data/aligned/${SUBSET}" --clean --num_jobs 1

echo "=== TENSORS one-stem $(date -Is) ==="
if command -v nix >/dev/null 2>&1; then
  nix develop .#train --command python3 scripts/build_train_tensors.py --subset "$SUBSET"
else
  python3 scripts/build_train_tensors.py --subset "$SUBSET"
fi

echo "=== EXPORT timeline $(date -Is) ==="
# copy wav into export for Godot convenience
mkdir -p export/ci-smoke
cp -a "data/prepared/${SUBSET}/${STEM}.wav" "export/ci-smoke/timeline_${STEM}.wav"
python3 scripts/export_viseme_timeline.py --subset "$SUBSET" --stem "$STEM" \
  --out export/ci-smoke/viseme_timeline.json
python3 - "$STEM" <<'PY'
import json, sys
from pathlib import Path
stem = sys.argv[1]
p = Path("export/ci-smoke/viseme_timeline.json")
d = json.loads(p.read_text())
d["wav"] = f"export/ci-smoke/timeline_{stem}.wav"
d["note"] = (
    (d.get("note") or "")
    + f" one-stem demo from test-clean {stem}"
)
p.write_text(json.dumps(d, indent=2) + "\n")
print("wav ->", d["wav"], "duration_s=", d.get("duration_s"))
PY
cp -a export/ci-smoke/viseme_timeline.json "export/ci-smoke/viseme_timeline_${STEM}.json"
echo "ONE_STEM_TIMELINE_OK stem=$STEM"
echo "=== ONE-STEM DONE $(date -Is) ==="
