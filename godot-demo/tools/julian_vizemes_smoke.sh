#!/usr/bin/env bash
# Julian Nix path — build MelFrontend + OnnxLoader, then headless vizemes smokes.
# Run from vizemes-align root: bash godot-demo/tools/julian_vizemes_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ONNX="${ONNX_LOADER_ROOT:-$ROOT/../godot-onnx-loader}"

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

echo "=== OnnxLoader scons + Godot smokes (nix develop) ==="
nix develop "$ONNX" --command bash -c "
	set -euo pipefail
	cd '$ONNX'
	git submodule update --init --recursive 2>/dev/null || true
	rm -f addons/onnx_loader/bin/libonnxruntime.so*
	scons platform=linux target=template_debug
	cd '$ROOT'
	bash godot-demo/tools/godot_mel_smoke.sh
"

echo "JULIAN_VIZEMES_SMOKE_OK"
