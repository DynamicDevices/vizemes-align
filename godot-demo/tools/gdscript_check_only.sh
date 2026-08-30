#!/usr/bin/env bash
# Headless Godot --check-only over every project .gd (editor parse footguns).
# Usage (from vizemes-align or godot-demo):
#   bash godot-demo/tools/gdscript_check_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT_BIN:-${GODOT:-}}"
if [[ -z "$GODOT" || ! -x "$GODOT" ]]; then
	for candidate in "${HOME}/Downloads/Godot_v4.6.1-stable_linux.x86_64" godot4 godot; do
		if [[ "$candidate" == /* && -x "$candidate" ]]; then
			GODOT="$candidate"
			break
		elif G="$(command -v "$candidate" 2>/dev/null)" && [[ -x "$G" ]]; then
			GODOT="$G"
			break
		fi
	done
fi
if [[ ! -x "$GODOT" ]]; then
	echo "gdscript_check_only: set GODOT_BIN to Godot 4.6+" >&2
	exit 2
fi

GODOT_VER="$("$GODOT" --version 2>/dev/null | head -n1 || true)"
case "$GODOT_VER" in
	*4.6*|*4.7*|*4.8*|*4.9*) ;;
	*)
		echo "gdscript_check_only: Godot 4.6+ required (got: ${GODOT_VER:-unknown})" >&2
		exit 2
		;;
esac

mkdir -p "$ROOT/.godot"
if [[ ! -f "$ROOT/.godot/extension_list.cfg" ]]; then
	cat >"$ROOT/.godot/extension_list.cfg" <<EOF
res://addons/onnx_loader/onnx_loader.gdextension
res://addons/vizemes_mel/vizemes_mel.gdextension
EOF
fi

mapfile -t SCRIPTS < <(
	cd "$ROOT"
	find . -name '*.gd' \
		! -path './.godot/*' \
		! -path './addons/vizeme-onnxmodels/*' \
		| sed 's|^\./||' | sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
	echo "gdscript_check_only: no .gd files under $ROOT" >&2
	exit 2
fi

OUT_DIR="/tmp/vizemes-gdscript-check-only"
mkdir -p "$OUT_DIR"
fail=0
cd "$ROOT"
for rel in "${SCRIPTS[@]}"; do
	res="res://${rel}"
	safe="${rel//\//_}"
	out="${OUT_DIR}/${safe}.txt"
	echo "=== check-only ${res} ==="
	set +e
	"$GODOT" --headless --path . --check-only --script "$res" >"$out" 2>&1
	rc=$?
	set -e
	if grep -qE 'SCRIPT ERROR:|Parse Error:|Failed to load script' "$out"; then
		echo "FAIL: parse error in ${res}" >&2
		grep -E 'SCRIPT ERROR:|Parse Error:|Failed to load script|at: ' "$out" | head -n 8 >&2
		fail=1
	elif [[ "$rc" -ne 0 ]]; then
		echo "FAIL: godot exit ${rc} for ${res}" >&2
		tail -n 20 "$out" >&2
		fail=1
	fi
done

if [[ "$fail" -ne 0 ]]; then
	echo "GODOT_GDSCRIPT_CHECK_ONLY_FAIL (logs under $OUT_DIR)" >&2
	exit 1
fi
echo "GODOT_GDSCRIPT_CHECK_ONLY_OK scripts=${#SCRIPTS[@]}"
