{
  description = "vizemes-align: LibriSpeech → MFA → Godot viseme export dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # steam-run → steam-unwrapped (unfree). Allow in-flake so plain
        # `nix develop` works without NIXPKGS_ALLOW_UNFREE + --impure.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        py = pkgs.python312;
        # textgrid is not in nixpkgs; package from PyPI into the store (NixOS-friendly).
        textgrid = py.pkgs.buildPythonPackage rec {
          pname = "textgrid";
          version = "1.6.1";
          pyproject = true;
          src = pkgs.fetchurl {
            # PyPI sdist is TextGrid-*.tar.gz (capital T/G).
            url = "https://files.pythonhosted.org/packages/cf/6f/701ef6aa56cf85c8965b7ff929f0766e0e8311c4478937eeee9441bf9663/TextGrid-1.6.1.tar.gz";
            hash = "sha256-DT+NT1EUdHd84ofS7O8JBXPA5jFBqnPG9ABjfh7KymM=";
          };
          nativeBuildInputs = with py.pkgs; [ setuptools ];
          # No upstream tests in the sdist worth running in the flake.
          doCheck = false;
          pythonImportsCheck = [ "textgrid" ];
        };
        # MFA via micromamba; keep Python export path pure-Nix (no .venv).
        pythonEnv = py.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgrid
          # torch/torchaudio: requirements-train.txt only (heavy; optional).
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pythonEnv
            ffmpeg
            git
            curl
            steam-run  # FHS wrap for upstream micromamba on NixOS
          ];

          shellHook = ''
            export VIZEMES_ALIGN_ROOT="$(pwd)"
            # Durable temp for micromamba/libmamba (nix-shell TMPDIR breaks steam-run)
            export TMPDIR="''${VIZEMES_TMPDIR:-/tmp}"
            export TMP="$TMPDIR"
            export TEMP="$TMPDIR"
            mkdir -p "$TMPDIR"
            echo "vizemes-align nix develop"
            echo "  Python (requests/tqdm/numpy/textgrid) + ffmpeg from the Nix store."
            echo "  No .venv for the export path — prefer store packages on NixOS."
            echo "  MFA on NixOS (stub-ld): steam-run is in this shell; prefer:"
            echo "    ./scripts/mamba_nixos.sh ...   # or ./scripts/bootstrap_mfa_micromamba.sh"
            echo "  Permanent: programs.nix-ld.enable = true; then bare micromamba works."
            echo "  MFA on NixOS: do NOT use nixpkgs micromamba (breaks as .mamba-wrapped)."
            echo "  One-time MFA env (inside this shell):"
            echo "    ./scripts/bootstrap_mfa_micromamba.sh"
            echo "    ./scripts/mamba_nixos.sh create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner"
            echo "    ./scripts/mamba_nixos.sh run -n mfa mfa model download acoustic english_us_arpa"
            echo "    ./scripts/mamba_nixos.sh run -n mfa mfa model download dictionary english_us_arpa"
            echo "    ./scripts/mamba_nixos.sh run -n mfa mfa model download g2p english_us_arpa"
            echo "  Train stack (torch, optional): python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt"
            if ! python3 -c 'import textgrid, numpy' 2>/dev/null; then
              echo "  ERROR: textgrid/numpy missing from flake pythonEnv" >&2
              exit 1
            fi
          '';
        };
      });
}
