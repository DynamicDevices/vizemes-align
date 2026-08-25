# Mel frontend origin

Vendored from `OpenLipSync/audio/mel_spectrogram.{c,h}` as the **first**
waveform→feature implementation for vizemes-align.

Not a compatibility commitment — Julian: frontend must stay swappable (LPC etc.).
When improving or replacing, keep `VizemesFrontendOps` in
`gdextension/include/feature_frontend.h` as the shared boundary.
