# Source/filter viseme experiment

## Question

The Mel frontend keeps a smoothed magnitude spectrum but discards phase and mixes
voice source with vocal-tract filtering. OVRLipSync's performance on normal and
whispered speech suggests that a small, speech-specific representation may be a
better classifier input. This experiment compares Mel with linear-prediction
filter features, with and without explicit source measurements.

## Runtime contract

`gdextension/frontends/source_filter/source_filter.c` is the single production
implementation used by both Godot and training. It accepts exactly 400 mono
floating-point PCM samples at 16 kHz: 25 ms support, advanced by callers in
160-sample (10 ms) hops. It allocates no heap memory and keeps no hidden state.

The stable filter representation is 16 LPC reflection coefficients. The optional
eight source/reliability values are RMS, LPC prediction gain, periodicity,
log-pitch, pitch confidence, pitch validity, harmonic/noise ratio, and residual
spectral tilt. Pitch is found by normalized time-domain autocorrelation; this path
does not use an FFT. Invalid pitch has a separate validity value so a model is not
asked to interpret an arbitrary placeholder as reliable evidence.

The 80-bin LPC envelope is a dense visualization of the 16-coefficient filter. It
is deliberately not part of the training vector.

## Preview

Build the GDExtension, then open `godot-demo/source_filter_preview.tscn`. The
scene uses normal Godot controls; only the variable-sized measurements are
uploaded as textures and drawn by the GPU. It accepts uncompressed mono PCM16
WAV files at 16 kHz.

The purple-to-orange panel shows the LPC filter envelope. Eight aligned traces
show source energy, prediction gain, periodicity, pitch and its reliability,
harmonicity, and tilt. It has been checked with normal LibriSpeech, synthetic
support probes, ordinary recorded vowels, and Oculus-headset whispered speech.

## Matched pilot

Build three tensor sets from the same aligned utterances and split:

1. `--frontend mel`: existing 80-value Mel baseline.
2. `--frontend lpc-filter`: 16 reflection coefficients.
3. `--frontend lpc-source-filter`: the same 16 coefficients plus eight explicit
   source and reliability values.

Use the same TCN, labels, lookahead, blend policy, seed, utterance list, and
speaker-held-out validation for all three. The comparison asks whether the
speech-specific representation improves held-out accuracy, temporal stability,
whisper behavior, inference cost, or perceived animation. It does not establish
that OVRLipSync uses LPC.

The host C benchmark currently measures about 45 microseconds per frame on the
development laptop, or about 0.45% of one CPU core at a 10 ms hop. Re-benchmark
on each target rather than treating this number as portable.

## Gates completed before remote training

- Native C sine/noise/silence regression.
- Godot 4.6 GDExtension analysis smoke.
- Normal, synthetic, vowel, and headset-whisper preview loads.
- One-step 24-feature TCN training and ONNX export.
- Godot ONNX loader inference with dynamic shape `[1, time, 24]`.

Selected-stem fits remain capacity tests, never held-out accuracy. Remote runs
must record the exact split, frontend contract, model metadata, convergence, and
whether they used CPU or an accelerator.
