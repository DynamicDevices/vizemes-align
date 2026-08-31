#!/usr/bin/env bash
# Copy export/ ONNX packs into the Godot-distributable addon (res://addons/vizeme-onnxmodels).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADDON="$ROOT/godot-demo/addons/vizeme-onnxmodels"
mkdir -p "$ADDON/tier-b" "$ADDON/tier-b-tcn" "$ADDON/one-stem-overfit" "$ADDON/ci-smoke" "$ADDON/fixtures"
copy_pack() {
  local src="$1" dest="$2"
  [[ -d "$src" ]] || return 0
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "$src"/model*.onnx "$src"/viseme_timeline*.json "$src"/seek_probe*.json; do
    cp -f "$f" "$dest/"
  done
}
copy_pack "$ROOT/export/tier-b" "$ADDON/tier-b"
copy_pack "$ROOT/export/tier-b-tcn" "$ADDON/tier-b-tcn"
copy_pack "$ROOT/export/one-stem-overfit" "$ADDON/one-stem-overfit"
copy_pack "$ROOT/export/ci-smoke" "$ADDON/ci-smoke"
# Wav fixtures for offline timeline / seek (stay inside Godot project)
mkdir -p "$ADDON/fixtures"
shopt -s nullglob
for w in "$ROOT/export/ci-smoke/"*.wav; do
  cp -f "$w" "$ADDON/fixtures/$(basename "$w")"
done
echo "Synced into $ADDON"
find "$ADDON" -type f | sort
