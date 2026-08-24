#!/usr/bin/env bash
# Upstream micromamba + MFA env (NixOS-safe). Avoid nixpkgs micromamba (.mamba-wrapped).
set -euo pipefail
ROOT="${HOME}/micromamba"
BIN="${ROOT}/bin/micromamba"
mkdir -p "${ROOT}/bin"
if [[ ! -x "$BIN" ]]; then
  echo "Installing upstream micromamba into ${ROOT} ..."
  tmp="$(mktemp -d)"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$tmp" bin/micromamba
  mv "$tmp/bin/micromamba" "$BIN"
  rm -rf "$tmp"
  chmod +x "$BIN"
fi
# Hide nixpkgs wrapper if present earlier on PATH
export PATH="${ROOT}/bin:${PATH}"
# Keep envs under micromamba-root (not ~/.mamba from a broken wrapper)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba-root}"
mkdir -p "$MAMBA_ROOT_PREFIX"
hash -r 2>/dev/null || true
echo "Using: $(command -v micromamba) ($(micromamba --version))"
case "$(command -v micromamba)" in
  *mamba-wrapped*)
    echo "ERROR: still resolving nixpkgs .mamba-wrapped — PATH=$(command -v micromamba)" >&2
    exit 1
    ;;
esac
if ! micromamba env list | awk '{print $1}' | grep -qx mfa; then
  micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
fi
micromamba run -n mfa mfa model download acoustic english_us_arpa
micromamba run -n mfa mfa model download dictionary english_us_arpa
micromamba run -n mfa mfa model download g2p english_us_arpa
echo "OK — PATH already has ${ROOT}/bin. Run: ./scripts/run_mfa.sh test-clean"
