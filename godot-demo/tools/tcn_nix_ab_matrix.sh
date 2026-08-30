#!/usr/bin/env bash
# Julian mid 1032/1034: A/B TCN load under Nix — which lever clears free()?
# Prints a matrix; exit 0 always (informational). Hard gates stay in CI mel smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO_ROOT:-$(cd "$ROOT/.." && pwd)}"
OUT_DIR="${OUT_DIR:-/tmp/vizemes-tcn-ab}"
NIXPKGS="${NIXPKGS:-github:nixos/nixpkgs/nixos-26.05}"
OFFICIAL_URL="${OFFICIAL_GODOT_URL:-https://github.com/godotengine/godot/releases/download/4.6.1-stable/Godot_v4.6.1-stable_linux.x86_64.zip}"
mkdir -p "$OUT_DIR"
export ONNX_LOADER_SKIP_SESSION_RELEASE="${ONNX_LOADER_SKIP_SESSION_RELEASE:-1}"
export ONNX_LOADER_ROOT="${ONNX_LOADER_ROOT:-}"

# 1) nixpkgs Godot + bundled MS ORT (current red path)
if command -v nix >/dev/null 2>&1; then
	nix shell "${NIXPKGS}#godot_4_6" "${NIXPKGS}#gcc" --command bash -c '
		set -euo pipefail
		G=$(command -v godot4 || command -v godot)
		export GODOT_BIN="$G"
		unset ONNX_ORT_BIN
		export ONNX_LOADER_ROOT="'"$ONNX_LOADER_ROOT"'"
		export ONNX_LOADER_SKIP_SESSION_RELEASE=1
		export OUT="'"$OUT_DIR"'/nixpkgs_godot_ms_ort-godot.txt"
		cd "'"$ROOT"'"
		bash tools/run_tcn_load_probe.sh
	' >"$OUT_DIR/nixpkgs_godot_ms_ort.txt" 2>&1 || true
	if grep -Fq 'free(): invalid size' "$OUT_DIR/nixpkgs_godot_ms_ort.txt"; then
		echo "RESULT nixpkgs_godot_ms_ort FAIL free():_invalid_size"
	elif grep -Fq 'GODOT_TCN_LOAD_PROBE_OK' "$OUT_DIR/nixpkgs_godot_ms_ort.txt"; then
		echo "RESULT nixpkgs_godot_ms_ort PASS"
	else
		echo "RESULT nixpkgs_godot_ms_ort FAIL other"
		tail -n 15 "$OUT_DIR/nixpkgs_godot_ms_ort.txt" || true
	fi
else
	echo "RESULT nixpkgs_godot_ms_ort SKIP no-nix"
fi

# 2) official Godot zip + same bundled MS ORT (on this Nix host).
# Official ELF often needs steam-run / nix-ld on NixOS.
OFFICIAL_DIR="$OUT_DIR/official-godot"
OFFICIAL_BIN="$OFFICIAL_DIR/Godot_v4.6.1-stable_linux.x86_64"
if [[ ! -x "$OFFICIAL_BIN" ]]; then
	mkdir -p "$OFFICIAL_DIR"
	curl -fsSL "$OFFICIAL_URL" -o "$OUT_DIR/godot-official.zip"
	unzip -o -q "$OUT_DIR/godot-official.zip" -d "$OFFICIAL_DIR"
	chmod +x "$OFFICIAL_BIN"
fi
OFFICIAL_WRAP=(env)
if command -v nix >/dev/null 2>&1; then
	if nix shell "${NIXPKGS}#steam-run" -c true 2>/dev/null; then
		OFFICIAL_WRAP=(nix shell "${NIXPKGS}#steam-run" -c steam-run)
	fi
fi
(
	export GODOT_BIN="$OFFICIAL_BIN"
	# When using steam-run, wrap the probe's godot invocation via GODOT_BIN that is a small shim.
	if [[ "${OFFICIAL_WRAP[0]}" == nix ]]; then
		SHIM="$OUT_DIR/godot-official-shim.sh"
		cat >"$SHIM" <<EOF
#!/usr/bin/env bash
exec nix shell ${NIXPKGS}#steam-run -c steam-run $(printf '%q' "$OFFICIAL_BIN") "\$@"
EOF
		chmod +x "$SHIM"
		export GODOT_BIN="$SHIM"
	fi
	unset ONNX_ORT_BIN
	export OUT="$OUT_DIR/official_godot_ms_ort-godot.txt"
	cd "$ROOT"
	bash tools/run_tcn_load_probe.sh
) >"$OUT_DIR/official_godot_ms_ort.txt" 2>&1 || true
if grep -Fq 'free(): invalid size' "$OUT_DIR/official_godot_ms_ort.txt"; then
	echo "RESULT official_godot_ms_ort FAIL free():_invalid_size"
elif grep -Fq 'GODOT_TCN_LOAD_PROBE_OK' "$OUT_DIR/official_godot_ms_ort.txt"; then
	echo "RESULT official_godot_ms_ort PASS"
else
	echo "RESULT official_godot_ms_ort FAIL other"
	tail -n 15 "$OUT_DIR/official_godot_ms_ort.txt" || true
fi

# 3) nixpkgs Godot + store ORT — hide bundled MS copy so ONNX_ORT_BIN wins.
if command -v nix >/dev/null 2>&1 && [[ -n "${ONNX_LOADER_ROOT:-}" ]]; then
	BUNDLE_BIN="$ROOT/addons/onnx_loader/bin"
	HIDE_DIR="$OUT_DIR/hide-ms-ort"
	mkdir -p "$HIDE_DIR"
	for f in libonnxruntime.so.1 libonnxruntime.so libstdc++.so.6 libgcc_s.so.1; do
		if [[ -e "$BUNDLE_BIN/$f" ]]; then
			mv "$BUNDLE_BIN/$f" "$HIDE_DIR/$f"
		fi
	done
	nix shell "${NIXPKGS}#godot_4_6" "${NIXPKGS}#onnxruntime" "${NIXPKGS}#gcc" --command bash -c '
		set -euo pipefail
		G=$(command -v godot4 || command -v godot)
		export GODOT_BIN="$G"
		ORTSO=$(ls -1 /nix/store/*-onnxruntime-*/lib/libonnxruntime.so 2>/dev/null | head -1 || true)
		if [[ -z "$ORTSO" || ! -f "$ORTSO" ]]; then
			echo "no store ORT .so" >&2
			exit 3
		fi
		export ONNX_ORT_BIN="$(dirname "$ORTSO")"
		export ONNX_LOADER_ROOT="'"$ONNX_LOADER_ROOT"'"
		export ONNX_LOADER_SKIP_SESSION_RELEASE=1
		export OUT="'"$OUT_DIR"'/nixpkgs_godot_store_ort-godot.txt"
		cd "'"$ROOT"'"
		bash tools/run_tcn_load_probe.sh
	' >"$OUT_DIR/nixpkgs_godot_store_ort.txt" 2>&1 || true
	# restore MS bundle
	for f in libonnxruntime.so.1 libonnxruntime.so libstdc++.so.6 libgcc_s.so.1; do
		if [[ -e "$HIDE_DIR/$f" ]]; then
			mv "$HIDE_DIR/$f" "$BUNDLE_BIN/$f"
		fi
	done
	if grep -Fq 'free(): invalid size' "$OUT_DIR/nixpkgs_godot_store_ort.txt"; then
		echo "RESULT nixpkgs_godot_store_ort FAIL free():_invalid_size"
	elif grep -Fq 'GODOT_TCN_LOAD_PROBE_OK' "$OUT_DIR/nixpkgs_godot_store_ort.txt"; then
		echo "RESULT nixpkgs_godot_store_ort PASS"
	else
		echo "RESULT nixpkgs_godot_store_ort FAIL other"
		tail -n 15 "$OUT_DIR/nixpkgs_godot_store_ort.txt" || true
	fi
else
	echo "RESULT nixpkgs_godot_store_ort SKIP"
fi

echo "TCN_AB_MATRIX_DONE dir=$OUT_DIR"
