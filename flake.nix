{
  description = "vizemes-align: LibriSpeech → MFA → Godot viseme export dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
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
          ];

          shellHook = ''
            export VIZEMES_ALIGN_ROOT="$(pwd)"
            echo "vizemes-align nix develop"
            echo "  Python/ffmpeg ready for download + prepare + Godot export."
            echo "  MFA on NixOS: do NOT use nixpkgs micromamba (breaks as .mamba-wrapped)."
            echo "  Install upstream micromamba once, then:"
            echo "    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C $HOME/micromamba bin/micromamba"
            echo "    export PATH=\"$HOME/micromamba/bin:\$PATH\""
            echo "    micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner"
            echo "    micromamba run -n mfa mfa model download acoustic english_us_arpa"
            echo "    micromamba run -n mfa mfa model download dictionary english_us_arpa"
            echo "    micromamba run -n mfa mfa model download g2p english_us_arpa"
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
