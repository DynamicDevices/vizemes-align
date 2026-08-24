#!/usr/bin/env bash
# Upstream micromamba + MFA env — NixOS-safe (steam-run / nix-ld).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MM=("${HERE}/mamba_nixos.sh")

echo "Bootstrapping MFA via NixOS-safe micromamba launcher ..."
"${MM[@]}" --version
if ! "${MM[@]}" env list 2>/dev/null | awk '{print $1}' | grep -qx mfa; then
  "${MM[@]}" create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
fi
"${MM[@]}" run -n mfa mfa model download acoustic english_us_arpa
"${MM[@]}" run -n mfa mfa model download dictionary english_us_arpa
"${MM[@]}" run -n mfa mfa model download g2p english_us_arpa
echo "OK — align with: ./scripts/run_mfa.sh test-clean"
