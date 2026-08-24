#!/usr/bin/env bash
# Resolve a usable micromamba binary. Prefers ~/micromamba/bin (upstream).
# Rejects nixpkgs .mamba-wrapped (breaks MFA activate/run).
set -euo pipefail

resolve_micromamba() {
  local candidates=(
    "${VIZEMES_MICROMAMBA:-}"
    "${HOME}/micromamba/bin/micromamba"
    "${HOME}/.local/bin/micromamba"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -x "$c" ]] && [[ "$(basename "$c")" == "micromamba" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  # PATH lookup — skip wrappers
  if command -v micromamba >/dev/null 2>&1; then
    c="$(command -v micromamba)"
    if [[ "$c" == *mamba-wrapped* ]] || [[ "$(basename "$c")" == .* ]]; then
      return 1
    fi
    if [[ "$(basename "$c")" == "micromamba" ]] || [[ "$(basename "$c")" == "mamba" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  fi
  return 1
}

is_broken_nix_micromamba() {
  local c
  c="$(command -v micromamba 2>/dev/null || true)"
  [[ -n "$c" && ( "$c" == *mamba-wrapped* || "$(basename "$c")" == .* ) ]]
}

# When sourced: define functions only. When executed: print path or exit 1.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if out="$(resolve_micromamba)"; then
    echo "$out"
    exit 0
  fi
  echo "No usable micromamba (need real binary named micromamba, not nixpkgs .mamba-wrapped)." >&2
  exit 1
fi
