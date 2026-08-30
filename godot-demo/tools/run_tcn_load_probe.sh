#!/usr/bin/env bash
# Headless TCN vs ci-smoke load under the current GODOT_BIN (Julian mid 993).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT_BIN:-${GODOT:-}}"
if [[ -z "$GODOT" || ! -x "$GODOT" ]]; then
	echo "run_tcn_load_probe: set GODOT_BIN" >&2
	exit 2
fi
OUT="${OUT:-/tmp/vizemes-tcn-load-probe.txt}"
cd "$ROOT"
# Prefer onnx_loader-only for the TCN probe (isolate MelFrontend).
PROBE_EXT="${PROBE_EXT_LIST:-onnx_only}"
mkdir -p .godot
case "$PROBE_EXT" in
onnx_only)
	cat >.godot/extension_list.cfg <<EOF
res://addons/onnx_loader/onnx_loader.gdextension
EOF
	;;
*)
	if [[ ! -f .godot/extension_list.cfg ]]; then
		cat >.godot/extension_list.cfg <<EOF
res://addons/onnx_loader/onnx_loader.gdextension
res://addons/vizemes_mel/vizemes_mel.gdextension
EOF
	fi
	;;
esac

# Host path first (same ORT .so) — splits Godot-only vs ORT/Nix.
if [[ -n "${ONNX_LOADER_ROOT:-}" && -f "${ONNX_LOADER_ROOT}/build/libonnx_runtime.a" ]]; then
	echo "=== host CreateSession TCN (no Godot) ==="
	set +e
	bash "$ROOT/tools/host_tcn_create.sh" \
		"$ROOT/addons/vizeme-onnxmodels/tier-b-tcn/model_final.onnx"
	host_rc=$?
	set -e
	echo "host_tcn_create exit=$host_rc"
fi

# Once Godot has mapped libstdc++.so.6, dlopen(ORT) reuses it (RPATH cannot
# replace). Prefer preloading the same gcc runtime we bundled beside ORT.
if [[ -z "${LD_PRELOAD:-}" && -d /nix/store ]]; then
	_preload=()
	_bin="$ROOT/addons/onnx_loader/bin"
	[[ -f "$_bin/libstdc++.so.6" ]] && _preload+=("$_bin/libstdc++.so.6")
	[[ -f "$_bin/libgcc_s.so.1" ]] && _preload+=("$_bin/libgcc_s.so.1")
	if [[ ${#_preload[@]} -gt 0 ]]; then
		export LD_PRELOAD
		LD_PRELOAD="$(IFS=:; echo "${_preload[*]}")${LD_PRELOAD:+:$LD_PRELOAD}"
		echo "run_tcn_load_probe: LD_PRELOAD=$LD_PRELOAD"
	fi
fi

set +e
"$GODOT" --headless --path . --script res://tools/tcn_load_probe.gd >"$OUT" 2>&1
rc=$?
set -e
cat "$OUT"
if grep -F 'free(): invalid size' "$OUT" >/dev/null; then
	echo "run_tcn_load_probe: free(): invalid size in log" >&2
	exit 1
fi
if grep -E '^Aborted' "$OUT" >/dev/null; then
	echo "run_tcn_load_probe: Aborted in log" >&2
	exit 1
fi
if [[ "$rc" -ne 0 ]]; then
	echo "run_tcn_load_probe: godot exit $rc" >&2
	exit 1
fi
grep -q GODOT_TCN_LOAD_PROBE_OK "$OUT"
echo "run_tcn_load_probe: OK"
