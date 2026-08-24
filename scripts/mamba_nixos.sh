#!/usr/bin/env bash
# Launch upstream micromamba on NixOS (stub-ld safe via steam-run / nix-ld).
set -euo pipefail
ROOT="${HOME}/micromamba"
BIN="${ROOT}/bin/micromamba"

is_nixos() { [[ -f /etc/NIXOS ]] || [[ -n "${NIX_PATH:-}" && -d /nix/store ]]; }

ensure_binary() {
  mkdir -p "${ROOT}/bin"
  if [[ ! -x "$BIN" ]]; then
    echo "Installing upstream micromamba into ${ROOT} ..."
    tmp="$(mktemp -d)"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$tmp" bin/micromamba
    mv "$tmp/bin/micromamba" "$BIN"
    rm -rf "$tmp"
    chmod +x "$BIN"
  fi
}

# Returns 0 if bare binary runs
bare_works() {
  "$BIN" --version >/dev/null 2>&1
}

run_mm() {
  ensure_binary
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba-root}"
  mkdir -p "$MAMBA_ROOT_PREFIX"
  # Prefer PATH that puts our binary first (never nixpkgs .mamba-wrapped)
  export PATH="${ROOT}/bin:${PATH}"

  if bare_works; then
    exec "$BIN" "$@"
  fi

  if is_nixos || ! bare_works; then
    if command -v steam-run >/dev/null 2>&1; then
      echo "[mamba-nixos] using steam-run (NixOS stub-ld)" >&2
      exec steam-run "$BIN" "$@"
    fi
    cat >&2 <<'EOF'
Upstream micromamba cannot start on this NixOS host (stub dynamic linker).

Quick fix (in nix develop / shell that has steam-run):
  steam-run ~/micromamba/bin/micromamba ...

Permanent fix — enable nix-ld in configuration.nix / flake, then rebuild:
  programs.nix-ld.enable = true;
  # optional: programs.nix-ld.libraries = with pkgs; [ stdenv.cc.cc zlib ];

Then re-login and re-run ./scripts/bootstrap_mfa_micromamba.sh

Or enter the project flake (adds steam-run):
  nix develop
  ./scripts/bootstrap_mfa_micromamba.sh
EOF
    exit 1
  fi
}

run_mm "$@"
