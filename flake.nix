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
          # torch/torchaudio: see devShells.train (heavy; not in default/CI).
        ]);
        pythonTrainEnv = py.withPackages (ps: with ps; [
          requests
          tqdm
          numpy
          textgrid
          soundfile
          pip
          # Do NOT pull nixpkgs torch/torchaudio here — often builds Triton
          # from source. Train uses precompiled PyTorch wheels via .venv
          # (see shellHook). CI never enters this shell.
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
            echo "  Train: nix develop .#train   # then one-shot PyTorch CPU wheels (no nixpkgs torch)"
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

        # Train shell: store python for mel/export deps; torch via precompiled wheels.
        # (nixpkgs torch often compiles triton — bad on laptops / thin runners.)
        devShells.train = pkgs.mkShell {
          packages = with pkgs; [
            pythonTrainEnv
            ffmpeg
            git
            curl
            steam-run
          ];

          shellHook = commonHook + ''
            echo "vizemes-align nix develop .#train"
            echo "  Store: numpy/textgrid/soundfile. Torch: precompiled CPU wheels (not nixpkgs)."
            _ok() { "$1" -c 'import torch, torchaudio, soundfile, textgrid, numpy' 2>/dev/null; }
            if _ok python3; then
              :
            elif [ -x .venv/bin/python ] && _ok .venv/bin/python; then
              export PATH="$PWD/.venv/bin:$PATH"
              echo "  (activated .venv with PyTorch wheels)"
            else
              echo "  One-time wheel install (precompiled — skips Triton source build):"
              echo "    python3 -m venv .venv"
              echo "    .venv/bin/pip install -U pip"
              echo "    .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
              echo "    .venv/bin/pip install soundfile"
              echo "  Then: source .venv/bin/activate"
              echo "  Cycle:"
              echo "    python3 scripts/build_train_tensors.py --subset test-clean"
              echo "    python3 scripts/train_viseme_smoke.py --subset test-clean --context 20"
            fi
          '';
        };
      });
}
