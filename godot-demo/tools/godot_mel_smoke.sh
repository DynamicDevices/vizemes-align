#!/usr/bin/env bash
# Headless Godot smokes — ensure both GDExtensions are registered (extension_list cache).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT_BIN:-${GODOT:-${HOME}/Downloads/Godot_v4.6.1-stable_linux.x86_64}}"
EXT_LIST="$ROOT/.godot/extension_list.cfg"

if [[ ! -x "$GODOT" ]]; then
	echo "GODOT_BIN must point at Godot 4.x" >&2
	exit 1
fi

mkdir -p "$ROOT/.godot"
cat >"$EXT_LIST" <<EOF
res://addons/onnx_loader/onnx_loader.gdextension
res://addons/vizemes_mel/vizemes_mel.gdextension
EOF

if [[ -z "${ONNX_ORT_BIN:-}" ]]; then
	unset LD_LIBRARY_PATH
fi

run_scene() {
	local scene="$1"
	local marker="$2"
	local out="/tmp/vizemes-${scene//\//-}.txt"
	cd "$ROOT"
	"$GODOT" --headless --path . --quit-after 1 "$scene" 2>&1 | tee "$out"
	grep -q "$marker" "$out"
}

run_scene "res://mel_smoke.tscn" "GODOT_MEL_ONNX_SMOKE_OK"
run_scene "res://lipsync_smoke.tscn" "GODOT_LIPSYNC_SMOKE_OK"
echo "GODOT_VIZEMES_SMOKE_OK"
