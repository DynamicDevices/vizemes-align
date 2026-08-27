#!/usr/bin/env bash
# Julian Nix path — MelFrontend + OnnxLoader (Godot 4.6 MS-ORT), then headless smokes.
# Run from vizemes-align root: bash godot-demo/tools/julian_vizemes_smoke.sh
#
# Do NOT use bare `nix develop` GODOT_BIN from godot-onnx-loader for these smokes —
# that shell still pins nixpkgs godot_4 4.5.1. Vizemes requires 4.6+; the proven
# path is tools/godot_46_ms_ort.sh (MS ORT bundle + nixos-26.05 godot_4_6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ONNX="${ONNX_LOADER_ROOT:-$ROOT/../godot-onnx-loader}"
NIXPKGS="${NIXPKGS:-github:nixos/nixpkgs/nixos-26.05}"

if [[ ! -d "$ONNX/.git" ]]; then
	echo "julian_vizemes_smoke: missing godot-onnx-loader at $ONNX" >&2
	exit 1
fi

echo "=== MelFrontend scons (.#train) ==="
cd "$ROOT"
nix develop .#train --command bash -c '
	set -euo pipefail
	cd gdextension
	git submodule update --init --recursive 2>/dev/null || true
	scons platform=linux target=template_debug
	test -f godot/bin/libvizemes_mel.linux.template_debug.x86_64.so
'

echo "=== OnnxLoader Godot 4.6 MS-ORT (csv proof) ==="
cd "$ONNX"
bash tools/godot_46_ms_ort.sh

echo "=== Vizemes headless smokes on Godot 4.6 ==="
root_q=$(printf '%q' "$ROOT")
nix shell "${NIXPKGS}#godot_4_6" --command bash -c "
set -euo pipefail
G=\$(command -v godot4 || command -v godot || true)
test -n \"\$G\" && test -x \"\$G\"
export GODOT_BIN=\"\$G\"
unset ONNX_ORT_BIN
export ONNX_LOADER_SKIP_SESSION_RELEASE=1
cd $root_q
bash godot-demo/tools/godot_mel_smoke.sh
"

echo "JULIAN_VIZEMES_SMOKE_OK"
