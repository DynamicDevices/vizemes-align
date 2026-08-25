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
        # Default shell stays on nixos-unstable for export/MFA tooling.
        # Do NOT put onnxruntime here — unstable often builds it from source.
        # ONNX sanity uses .#train (nixos-25.11 Hydra substitutes).
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
          onnxruntime  # Hydra-cached on 25.11; scripts/sanity_check_onnx.py
          # triton comes in as a torch dep when present on the channel
        ]);
        commonHook = ''
            export VIZEMES_ALIGN_ROOT="$(pwd)"
            export TMPDIR="''${VIZEMES_TMPDIR:-/tmp}"
            export TMP="$TMPDIR"
            export TEMP="$TMPDIR"
            mkdir -p "$TMPDIR"
        '';
        # C/C++ ORT lib for gdextension (python onnxruntime is separate in pythonTrainEnv).
        ortC = pkgsTrain.onnxruntime;
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
            echo "  ONNX sanity (Hydra-cached onnxruntime): nix develop .#train --command \\"
            echo "    python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx --subset ci-fixture --limit 1"
            echo "  Train: nix develop .#train   # torch/torchaudio/onnxruntime from nixos-25.11 cache"
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
        # pkgsTrain.onnxruntime = C/C++ lib (gdextension); python onnxruntime is separate.
        devShells.train = pkgsTrain.mkShell {
          packages = with pkgsTrain; [
            pythonTrainEnv
            onnxruntime # C API headers + libonnxruntime for gdextension/
            ffmpeg
            git
            curl
            steam-run
            gcc
            gnumake
          ];

          shellHook = commonHook + ''
            export ORT_ROOT="${ortC}"
            echo "vizemes-align nix develop .#train"
            echo "  Store (nixos-25.11 Hydra): numpy/textgrid/soundfile/torch/torchaudio/onnxruntime"
            echo "  ORT_ROOT=$ORT_ROOT  # C lib for: cd gdextension && make smoke"
            echo "  No .venv. First enter may download ~1GiB from cache.nixos.org (not compile)."
            echo "  ONNX sanity:"
            echo "    python3 scripts/sanity_check_onnx.py export/ci-smoke/model.onnx --subset ci-fixture --limit 1"
            echo "  Cycle:"
            echo "    python3 scripts/build_train_tensors.py --subset test-clean"
            echo "    python3 scripts/train_viseme_smoke.py --subset test-clean --context 20"
            if ! python3 -c 'import torch, torchaudio, soundfile, textgrid, numpy, onnxruntime' 2>/dev/null; then
              echo "  ERROR: train pythonEnv missing imports" >&2
              exit 1
            fi
            python3 -c 'import torch, onnxruntime as ort; print("  torch", torch.__version__, "cuda", torch.cuda.is_available(), "ort", ort.__version__)'
            if [ ! -f "$ORT_ROOT/include/onnxruntime_c_api.h" ]; then
              echo "  WARN: onnxruntime C headers missing under ORT_ROOT" >&2
            fi
          '';
        };
      });
}
