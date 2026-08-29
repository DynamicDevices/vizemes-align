# gpm consumer spike

Reproducible install of the three DynamicDevices Godot addons via
[gpm](https://github.com/cafecito-games/godot-package-manager) (GitHub Release
zips — not git-source-only like gd-plug).

## Develop vs release

| Stage | Coupling |
|-------|----------|
| Develop | Symlink sibling `addons/` trees (vizemes `godot-demo` pattern) |
| Release | `addons.toml` + `gpm install` from Release zip assets |

## Setup

```bash
# once
curl -sL https://github.com/cafecito-games/godot-package-manager/releases/download/v0.3.0/gpm_0.3.0_linux_amd64.tar.gz \
  | tar -xz -C ~/.local/bin gpm

cd examples/gpm-consumer
gpm install   # uses addons.toml + addons.lock
```

Expect `addons/{onnx_loader,speexdsp,vizemes_mel}/` with prebuilt `.so` files.

## Manifest

See `addons.toml`. Pins:

- `DynamicDevices/godot-onnx-loader` @ `v0.2.1`
- `DynamicDevices/godot-speexdsp` @ `v0.1.0`
- `DynamicDevices/vizemes-align` @ `v0.1.0-mel-addon` (MelFrontend ship unit)

Godot **4.6+** for onnx_loader; speexdsp/vizemes_mel build for 4.5+ ABI.

## Notes

- Commit `addons.toml` + `addons.lock`; ignore installed `addons/` (or vendor if you prefer).
- Cutting new addon zips: `godot-onnx-loader/tools/package_assetlib.sh`,
  `godot-speexdsp/tools/package_assetlib.sh`, `vizemes-align/tools/package_vizemes_mel.sh`.
