#!/usr/bin/env bash
# MFA-align a prepared LibriSpeech subset → TextGrids under data/aligned/<subset>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBSET="${1:-test-clean}"
# LibriSpeech OpenSLR names use hyphens; prepare_corpus may use underscores — accept both
MFA_ACOUSTIC="${MFA_ACOUSTIC:-english_us_arpa}"
MFA_DICTIONARY="${MFA_DICTIONARY:-english_us_arpa}"

PREPARED="${ROOT}/data/prepared/${SUBSET}"
if [[ ! -d "$PREPARED" ]]; then
  # try underscore form
  alt="${SUBSET//-/_}"
  if [[ -d "${ROOT}/data/prepared/${alt}" ]]; then
    PREPARED="${ROOT}/data/prepared/${alt}"
    SUBSET="$alt"
  fi
fi

CORPUS="${ROOT}/data/cache/mfa_corpus_${SUBSET}"
OUT="${ROOT}/data/aligned/${SUBSET}"
OOVS="${ROOT}/data/cache/mfa_${SUBSET}/oovs"

if [[ ! -d "$PREPARED" ]]; then
  echo "Prepared corpus missing: $PREPARED" >&2
  echo "Run: python scripts/prepare_corpus.py --subset ${SUBSET}" >&2
  exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba required (env mfa with montreal-forced-aligner)" >&2
  exit 1
fi

mkdir -p "$CORPUS" "$OUT" "$OOVS"

# Flat corpus: wav + matching lab/txt beside each other for MFA
# Prefer copying prepared layout if already flat; else link wav/lab pairs
rm -rf "${CORPUS:?}/"*
# prepared layout from createDataCorpus: per-utterance dirs or flat — copy tree
cp -a "${PREPARED}/." "$CORPUS/"

echo "[mfa] acoustic=$MFA_ACOUSTIC dictionary=$MFA_DICTIONARY"
echo "[mfa] corpus=$CORPUS → $OUT"

micromamba run -n mfa mfa find_oovs "$CORPUS" "$MFA_DICTIONARY" "$OOVS" || true
OOVS_FILE="${OOVS}/oovs_found_${MFA_DICTIONARY}.txt"
OOVS_DICT="${OOVS}/oovs.dict"
if [[ -f "$OOVS_FILE" && -s "$OOVS_FILE" ]]; then
  micromamba run -n mfa mfa g2p "$OOVS_FILE" "$MFA_DICTIONARY" "$OOVS_DICT" || true
  if [[ -f "$OOVS_DICT" && -s "$OOVS_DICT" ]]; then
    micromamba run -n mfa mfa model add_words "$MFA_DICTIONARY" "$OOVS_DICT" || true
  fi
fi

micromamba run -n mfa mfa align "$CORPUS" "$MFA_DICTIONARY" "$MFA_ACOUSTIC" "$OUT" --clean
echo "[mfa] done → $OUT"
