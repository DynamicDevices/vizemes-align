#!/usr/bin/env bash
# Headless Godot smokes — ensure both GDExtensions are registered (extension_list cache).
# Nix: run inside godot-onnx-loader `nix develop` (sets GODOT_BIN + ONNX_ORT_BIN).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT_BIN:-${GODOT:-}}"
if [[ -z "$GODOT" || ! -x "$GODOT" ]]; then
	for candidate in "${HOME}/Downloads/Godot_v4.6.1-stable_linux.x86_64" godot4 godot; do
		if [[ "$candidate" == /* ]]; then
			if [[ -x "$candidate" ]]; then
				GODOT="$candidate"
				break
			fi
		elif G="$(command -v "$candidate" 2>/dev/null)" && [[ -x "$G" ]]; then
			GODOT="$G"
			break
		fi
	done
fi
EXT_LIST="$ROOT/.godot/extension_list.cfg"
MEL_SO="$ROOT/../gdextension/godot/bin/libvizemes_mel.linux.template_debug.x86_64.so"

if [[ ! -x "$GODOT" ]]; then
	echo "GODOT_BIN must point at Godot 4.6+ (try: ~/Downloads/Godot_v4.6.1-stable_linux.x86_64 or godot-onnx-loader tools/godot_46_ms_ort.sh)" >&2
	exit 1
fi

GODOT_VER="$("$GODOT" --version 2>/dev/null | head -n1 || true)"
case "$GODOT_VER" in
	*4.6*|*4.7*|*4.8*|*4.9*) ;;
	*)
		echo "Godot 4.6+ required (got: ${GODOT_VER:-unknown}). Set GODOT_BIN to a 4.6 binary." >&2
		exit 1
		;;
esac

if [[ ! -f "$MEL_SO" ]]; then
	echo "MelFrontend .so missing: $MEL_SO" >&2
	echo "Rebuild: nix develop .#train --command bash -c 'cd gdextension && scons platform=linux target=template_debug'" >&2
	echo "Or run: bash godot-demo/tools/julian_vizemes_smoke.sh" >&2
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
run_scene "res://streaming_smoke.tscn" "GODOT_STREAMING_SMOKE_OK"
run_scene "res://seek_probe.tscn" "GODOT_SEEK_PROBE_OK"
echo "GODOT_VIZEMES_SMOKE_OK"
