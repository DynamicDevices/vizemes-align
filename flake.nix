{
  description = "vizemes-align: LibriSpeech → MFA → Godot viseme export dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Train/torch: pin a channel where Hydra has prebuilt torch+triton (no
    # local Triton compile). Verified 2026-08-25: nixos-25.11 substitutes;
    # unstable / 26.05 often rebuild triton on this host.
    nixpkgs-train.url = "github:NixOS/nixpkgs/nixos-25.11";
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
        pyTrain = pkgsTrain.python312;
        mkTextgrid = pyPkgs: pyPkgs.buildPythonPackage rec {
          pname = "textgrid";
          version = "1.6.1";
          pyproject = true;
          src = pkgs.fetchurl {
            # PyPI sdist is TextGrid-*.tar.gz (capital T/G).
            url = "https://files.pythonhosted.org/packages/cf/6f/701ef6aa56cf85c8965b7ff929f0766e0e8311c4478937eeee9441bf9663/TextGrid-1.6.1.tar.gz";
            hash = "sha256-DT+NT1EUdHd84ofS7O8JBXPA5jFBqnPG9ABjfh7KymM=";
          };
          nativeBuildInputs = with pyPkgs; [ setuptools ];
          doCheck = false;
          pythonImportsCheck = [ "textgrid" ];
        };
        textgrid = mkTextgrid py.pkgs;
        textgridTrain = mkTextgrid pyTrain.pkgs;
        # MFA via micromamba; keep Python export path pure-Nix (no .venv).
        pythonEnv = py.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgrid
        ]);
        # Full train stack from nixpkgs-train binary cache — no .venv.
        pythonTrainEnv = pyTrain.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgridTrain
          soundfile
          torch
          torchaudio
          # triton comes in as a torch dep when present on the channel
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
            echo "  Train: nix develop .#train   # torch/torchaudio from nixpkgs-25.11 cache"
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

        # Train shell: all store packages from nixos-25.11 (Hydra cache).
        # No .venv — nixpkgs unstable/26.05 often rebuild Triton locally.
        devShells.train = pkgsTrain.mkShell {
          packages = with pkgsTrain; [
            pythonTrainEnv
            ffmpeg
            git
            curl
            steam-run
          ];

          shellHook = commonHook + ''
            echo "vizemes-align nix develop .#train"
            echo "  Store (nixpkgs nixos-25.11): numpy/textgrid/soundfile/torch/torchaudio"
            echo "  No .venv. First enter may download ~1GiB from cache.nixos.org (not compile)."
            echo "  Cycle:"
            echo "    python3 scripts/build_train_tensors.py --subset test-clean"
            echo "    python3 scripts/train_viseme_smoke.py --subset test-clean --context 20"
            if ! python3 -c 'import torch, torchaudio, soundfile, textgrid, numpy' 2>/dev/null; then
              echo "  ERROR: train pythonEnv missing imports" >&2
              exit 1
            fi
            python3 -c 'import torch; print("  torch", torch.__version__, "cuda", torch.cuda.is_available())'
          '';
        };
      });
}
