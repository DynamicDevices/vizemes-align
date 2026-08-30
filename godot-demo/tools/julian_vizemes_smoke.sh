#!/usr/bin/env bash
# Julian Nix path — MelFrontend + OnnxLoader (store Godot 4.6 + store ORT).
# Run from vizemes-align root: bash godot-demo/tools/julian_vizemes_smoke.sh
#
# A/B (mid 1067): nixpkgs godot_4_6 + nixpkgs onnxruntime PASS; MS ORT under
# nixpkgs Godot free()s. Do not use tools/godot_46_ms_ort.sh on Nix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ONNX="${ONNX_LOADER_ROOT:-$ROOT/../godot-onnx-loader}"
NIXPKGS="${NIXPKGS:-github:nixos/nixpkgs/nixos-26.05}"

if [[ ! -d "$ONNX/.git" ]]; then
	echo "julian_vizemes_smoke: missing godot-onnx-loader at $ONNX" >&2
	exit 1
fi
if [[ ! -x "$ONNX/tools/godot_46_nix_store_ort.sh" ]]; then
	echo "julian_vizemes_smoke: need godot-onnx-loader tools/godot_46_nix_store_ort.sh" >&2
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

echo "=== OnnxLoader Godot 4.6 + nixpkgs ORT (csv proof) ==="
cd "$ONNX"
bash tools/godot_46_nix_store_ort.sh
ln -sfn "$ONNX/addons/onnx_loader" "$ROOT/godot-demo/addons/onnx_loader"
rm -f "$ROOT/godot-demo/addons/onnx_loader/bin/libonnxruntime.so"*

echo "=== GDScript --check-only (editor parse footguns) ==="
root_q=$(printf '%q' "$ROOT")
nix shell "${NIXPKGS}#godot_4_6" --command bash -c "
set -euo pipefail
G=\$(command -v godot4 || command -v godot || true)
test -n \"\$G\" && test -x \"\$G\"
export GODOT_BIN=\"\$G\"
cd $root_q
bash godot-demo/tools/gdscript_check_only.sh
"

echo "=== Vizemes headless smokes on store Godot 4.6 + store ORT ==="
root_q=$(printf '%q' "$ROOT")
onnx_q=$(printf '%q' "$ONNX")
nix shell "${NIXPKGS}#godot_4_6" "${NIXPKGS}#onnxruntime" --command bash -c "
set -euo pipefail
G=\$(command -v godot4 || command -v godot || true)
test -n \"\$G\" && test -x \"\$G\"
export GODOT_BIN=\"\$G\"
ORTSO=\$(ls -1 /nix/store/*-onnxruntime-*/lib/libonnxruntime.so 2>/dev/null | head -1)
test -n \"\$ORTSO\" && test -f \"\$ORTSO\"
export ONNX_ORT_BIN=\"\$(dirname \"\$ORTSO\")\"
export ORT_BUNDLE=0
export ONNX_LOADER_ROOT=$onnx_q
export ONNX_LOADER_SKIP_SESSION_RELEASE=1
cd $root_q
bash godot-demo/tools/godot_mel_smoke.sh
bash godot-demo/tools/run_tcn_load_probe.sh
"

echo "JULIAN_VIZEMES_SMOKE_OK"
