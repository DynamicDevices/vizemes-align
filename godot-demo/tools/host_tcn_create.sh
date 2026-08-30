#!/usr/bin/env bash
# Host CreateSession for tier-b-tcn (no Godot) — compare with Godot free() (Julian 993).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX_ROOT="${ONNX_LOADER_ROOT:-$ROOT/../godot-onnx-loader}"
TCN="${1:-$ROOT/addons/vizeme-onnxmodels/tier-b-tcn/model_final.onnx}"
A="$ONNX_ROOT/build/libonnx_runtime.a"
ORTDIR="$ONNX_ROOT/addons/onnx_loader/bin"
SRC="$ONNX_ROOT/src"
OUT="$ONNX_ROOT/build/host_tcn_create"

if [[ ! -f "$TCN" ]]; then
	echo "host_tcn_create: missing $TCN" >&2
	exit 2
fi
if [[ ! -f "$A" || ! -f "$ORTDIR/libonnxruntime.so.1" ]]; then
	echo "host_tcn_create: need built onnx-loader at $ONNX_ROOT" >&2
	exit 2
fi

mkdir -p "$ONNX_ROOT/build"
cat >"$ONNX_ROOT/build/host_tcn_create.c" <<'C'
#include "onnx_runtime.h"
#include <stdio.h>
int main(int argc, char **argv) {
	fprintf(stderr, "HOST_TCN_BUILD %s\n", ONNX_LOADER_BUILD);
	OnnxRuntime *rt = onnx_runtime_create(argv[1]);
	if (!rt) {
		fprintf(stderr, "HOST_TCN_CREATE_FAIL\n");
		return 1;
	}
	fprintf(stderr, "HOST_TCN_CREATE_OK in=%s isize=%d osize=%d\n",
		onnx_runtime_input_name(rt), onnx_runtime_input_size(rt),
		onnx_runtime_output_size(rt));
	onnx_runtime_destroy(rt);
	fprintf(stderr, "HOST_TCN_DESTROY_OK\n");
	return 0;
}
C

_gcc() {
	if command -v gcc >/dev/null 2>&1; then
		gcc "$@"
	else
		echo "host_tcn_create: gcc missing" >&2
		exit 2
	fi
}

_gcc -O0 -I "$SRC" "$ONNX_ROOT/build/host_tcn_create.c" "$A" \
	-L "$ORTDIR" -lonnxruntime -Wl,-rpath,"$ORTDIR" -ldl -lpthread -lm -o "$OUT"

bash "$ONNX_ROOT/tools/with_bundled_ort.sh" "$OUT" "$TCN"
