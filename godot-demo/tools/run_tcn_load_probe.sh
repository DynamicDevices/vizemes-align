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
mkdir -p .godot
if [[ ! -f .godot/extension_list.cfg ]]; then
	cat >.godot/extension_list.cfg <<EOF
res://addons/onnx_loader/onnx_loader.gdextension
res://addons/vizemes_mel/vizemes_mel.gdextension
EOF
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
