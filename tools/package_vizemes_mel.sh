#!/usr/bin/env bash
# Package addons/vizemes_mel for consumers (debug .so; release if present).
# Output: /tmp/vizemes-mel-<ver>-linux-x86_64.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_VER="${PKG_VER:-$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)}"
OUT_DIR="${OUT_DIR:-/tmp/vizemes-mel-assetlib}"
ZIP="${ZIP:-/tmp/vizemes-mel-${PKG_VER}-linux-x86_64.zip}"

cd "$ROOT"
git submodule update --init --recursive
if [[ "${SKIP_SCONS:-0}" != "1" ]]; then
	( cd gdextension && scons -j"$(nproc)" platform=linux target=template_debug )
fi

DBG="addons/vizemes_mel/bin/libvizemes_mel.linux.template_debug.x86_64.so"
test -f "$DBG"
test -f addons/vizemes_mel/vizemes_mel.gdextension

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/addons/vizemes_mel/bin"
cp -a addons/vizemes_mel/vizemes_mel.gdextension "$OUT_DIR/addons/vizemes_mel/"
[[ -f addons/vizemes_mel/README.md ]] && cp -a addons/vizemes_mel/README.md "$OUT_DIR/addons/vizemes_mel/"
cp -a "$DBG" "$OUT_DIR/addons/vizemes_mel/bin/"
shopt -s nullglob
for f in addons/vizemes_mel/bin/libvizemes_mel.linux.template_release*.so; do
	cp -a "$f" "$OUT_DIR/addons/vizemes_mel/bin/"
done

rm -f "$ZIP"
( cd "$OUT_DIR" && zip -qr "$ZIP" addons )
echo "wrote $ZIP"
unzip -l "$ZIP" | head -20
