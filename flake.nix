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
        # MFA is typically installed via micromamba/conda; keep Python export path pure-Nix.
        pythonEnv = py.withPackages (ps: with ps; [
          requests
          tqdm
          pip
          # textgrid may need pip if not in nixpkgs for this channel
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
            echo "vizemes-align nix develop"
            echo "  Python/ffmpeg ready for download + prepare + Godot export."
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
            # Optional: local venv for textgrid if missing from nixpkgs
            if ! python3 -c 'import textgrid' 2>/dev/null; then
              if [ ! -d .venv ]; then
                python3 -m venv .venv
                .venv/bin/pip -q install -r requirements.txt
              fi
              export PATH="$PWD/.venv/bin:$PATH"
              echo "  (activated .venv for textgrid/requirements.txt)"
            fi
          '';
        };
      });
}
