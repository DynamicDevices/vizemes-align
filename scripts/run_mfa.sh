#!/usr/bin/env bash
# MFA-align a prepared LibriSpeech subset → TextGrids under data/aligned/<subset>
# NixOS: uses upstream ~/micromamba only — never nixpkgs .mamba-wrapped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=resolve_micromamba.sh
source "${ROOT}/scripts/resolve_micromamba.sh"

SUBSET="${1:-test-clean}"
MFA_ACOUSTIC="${MFA_ACOUSTIC:-english_us_arpa}"
MFA_DICTIONARY="${MFA_DICTIONARY:-english_us_arpa}"

PREPARED="${ROOT}/data/prepared/${SUBSET}"
if [[ ! -d "$PREPARED" ]]; then
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

MM=""
if ! MM="$(resolve_micromamba)"; then
  echo "No usable micromamba on PATH." >&2
  if is_broken_nix_micromamba; then
    echo "Detected nixpkgs micromamba (.mamba-wrapped) — MFA cannot use it." >&2
  fi
  echo "Bootstrapping upstream micromamba + MFA env ..." >&2
  "${ROOT}/scripts/bootstrap_mfa_micromamba.sh"
  MM="$(resolve_micromamba)"
fi

export PATH="$(dirname "$MM"):${PATH}"
echo "[mfa] using micromamba=$MM"

if ! "$MM" env list 2>/dev/null | awk '{print $1}' | grep -qx mfa; then
  echo "Creating micromamba env mfa ..." >&2
  "$MM" create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
  "$MM" run -n mfa mfa model download acoustic "${MFA_ACOUSTIC}"
  "$MM" run -n mfa mfa model download dictionary "${MFA_DICTIONARY}"
  "$MM" run -n mfa mfa model download g2p "${MFA_ACOUSTIC}"
fi

mkdir -p "$CORPUS" "$OUT" "$OOVS"
rm -rf "${CORPUS:?}/"*
cp -a "${PREPARED}/." "$CORPUS/"

echo "[mfa] acoustic=$MFA_ACOUSTIC dictionary=$MFA_DICTIONARY"
echo "[mfa] corpus=$CORPUS → $OUT"

mmfa() { "$MM" run -n mfa mfa "$@"; }

mmfa find_oovs "$CORPUS" "$MFA_DICTIONARY" "$OOVS" || true
OOVS_FILE="${OOVS}/oovs_found_${MFA_DICTIONARY}.txt"
OOVS_DICT="${OOVS}/oovs.dict"
if [[ -f "$OOVS_FILE" && -s "$OOVS_FILE" ]]; then
  mmfa g2p "$OOVS_FILE" "$MFA_DICTIONARY" "$OOVS_DICT" || true
  if [[ -f "$OOVS_DICT" && -s "$OOVS_DICT" ]]; then
    mmfa model add_words "$MFA_DICTIONARY" "$OOVS_DICT" || true
  fi
fi

mmfa align "$CORPUS" "$MFA_DICTIONARY" "$MFA_ACOUSTIC" "$OUT" --clean
echo "[mfa] done → $OUT"
