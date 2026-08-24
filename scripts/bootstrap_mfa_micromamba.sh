#!/usr/bin/env bash
# Upstream micromamba + MFA env (NixOS-safe). Avoid nixpkgs micromamba (.mamba-wrapped).
set -euo pipefail
ROOT="${HOME}/micromamba"
BIN="${ROOT}/bin/micromamba"
mkdir -p "${ROOT}"
if [[ ! -x "$BIN" ]]; then
  echo "Installing upstream micromamba into ${ROOT} ..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$ROOT" bin/micromamba
fi
export PATH="${ROOT}/bin:${PATH}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba-root}"
mkdir -p "$MAMBA_ROOT_PREFIX"
echo "Using: $(command -v micromamba) ($(micromamba --version))"
if ! micromamba env list | awk '{print $1}' | grep -qx mfa; then
  micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
fi
micromamba run -n mfa mfa model download acoustic english_us_arpa
micromamba run -n mfa mfa model download dictionary english_us_arpa
micromamba run -n mfa mfa model download g2p english_us_arpa
echo "OK — now: export PATH=\"${ROOT}/bin:\$PATH\" && ./scripts/run_mfa.sh test-clean"
