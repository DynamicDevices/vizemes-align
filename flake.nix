{
  description = "vizemes-align: LibriSpeech → MFA → Godot viseme export dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Hydra-cached torch/triton (python313) — avoid compiling Triton on laptops.
    nixpkgs-train.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-train, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # steam-run → steam-unwrapped (unfree). Allow in-flake so plain
        # `nix develop` works without NIXPKGS_ALLOW_UNFREE + --impure.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pkgsTrain = import nixpkgs-train {
          inherit system;
          config.allowUnfree = true;
        };
        py = pkgs.python312;
        pyTrain = pkgsTrain.python313;
        mkTextgrid = pyPkgs: pkgsRef: pyPkgs.buildPythonPackage rec {
          pname = "textgrid";
          version = "1.6.1";
          pyproject = true;
          src = pkgsRef.fetchurl {
            # PyPI sdist is TextGrid-*.tar.gz (capital T/G).
            url = "https://files.pythonhosted.org/packages/cf/6f/701ef6aa56cf85c8965b7ff929f0766e0e8311c4478937eeee9441bf9663/TextGrid-1.6.1.tar.gz";
            hash = "sha256-DT+NT1EUdHd84ofS7O8JBXPA5jFBqnPG9ABjfh7KymM=";
          };
          nativeBuildInputs = with pyPkgs; [ setuptools ];
          doCheck = false;
          pythonImportsCheck = [ "textgrid" ];
        };
        textgrid = mkTextgrid py.pkgs pkgs;
        textgridTrain = mkTextgrid pyTrain.pkgs pkgsTrain;
        # MFA via micromamba; keep Python export path pure-Nix (no .venv).
        pythonEnv = py.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgrid
        ]);
        # Train: nixos-26.05 python313 + Hydra torch/triton/torchaudio (no venv).
        pythonTrainEnv = pyTrain.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgridTrain
          soundfile
          torch
          torchaudio
          triton
        ]);
        commonHook = ''
            export VIZEMES_ALIGN_ROOT="$(pwd)"
            export TMPDIR="''${VIZEMES_TMPDIR:-/tmp}"
            export TMP="$TMPDIR"
            export TEMP="$TMPDIR"
            mkdir -p "$TMPDIR"
        '';
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pythonEnv
            ffmpeg
            git
            curl
            steam-run  # FHS wrap for upstream micromamba on NixOS
          ];

          shellHook = commonHook + ''
            echo "vizemes-align nix develop (default = export/MFA)"
            echo "  Python (requests/tqdm/numpy/textgrid) + ffmpeg from the Nix store."
            echo "  Train: nix develop .#train   # nixos-26.05 python313 + Hydra torch/triton"
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
            if ! python3 -c 'import textgrid, numpy' 2>/dev/null; then
              echo "  ERROR: textgrid/numpy missing from flake pythonEnv" >&2
              exit 1
            fi
          '';
        };

        # Store-only train shell (no .venv). Uses nixos-26.05 for Hydra-built torch/triton.
        # First entry should download from cache.nixos.org — if it compiles, check substituters.
        devShells.train = pkgsTrain.mkShell {
          packages = with pkgsTrain; [
            pythonTrainEnv
            ffmpeg
            git
            curl
          ];

          shellHook = commonHook + ''
            echo "vizemes-align nix develop .#train"
            echo "  nixos-26.05 python313 + torch/torchaudio/triton from Hydra (no .venv)."
            echo "  Expect store downloads from cache.nixos.org — not a Triton source build."
            echo "  Cycle:"
            echo "    python3 scripts/build_train_tensors.py --subset test-clean"
            echo "    python3 scripts/train_viseme_smoke.py --subset test-clean --context 20"
            if ! python3 -c 'import torch, torchaudio, triton, soundfile, textgrid, numpy' 2>/dev/null; then
              echo "  ERROR: train pythonEnv missing imports" >&2
              exit 1
            fi
          '';
        };
      });
}
