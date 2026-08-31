# Vizeme Opus side-channel contract

Version 1 carries exactly one byte on every 20 ms Opus packet (50 bytes/s,
400 bit/s). It sends sparse transition boundary events rather than repeating a
blend magnitude. The receiver's audio playback buffer reconstructs the blend at
the reported audio times.

## Timing terms

Keep these independent in manifests and benchmarks:

- Mel hop/output cadence: 10 ms for the current models.
- Acoustic context: 20 overlapping Mel frames, about 215 ms including the
  analysis window; this is not a 20 ms boundary resolution.
- Label lookahead: normally 50 ms, compensated at playback.
- Metadata cadence: one byte per 20 ms Opus packet.
- Receiver buffer: at least 200 ms, so recently inferred boundaries can still
  be applied before their corresponding audio is rendered.

MFA supplies acoustically aligned hard phone intervals. It does not export a
measured gradual blend window. Transition shape must therefore come from phone
posterior mass around the boundary, or from a learned causal mapper evaluated
against that deterministic rule; a fixed 60 ms soft ramp is not ground truth.

## Byte layout

| Byte | Meaning |
|---|---|
| `0ttooo` conceptually, high nibble `0x0..0xE`, low `0x0..0x7` | target viseme `0..14` START, `ooo` packets ago |
| high nibble `0x0..0xE`, low `0x8..0xF` | same target END, `(low & 7)` packets ago |
| `0xF0` | no event |
| `0xF1..0xFF` | absolute resync to dominant viseme `0..14` |

The relative range is 0..7 packets (0..140 ms), within the receiver's 200 ms
buffer. For example, START(target 12, 3 packets ago) followed by END(target 12,
1 packet ago) reconstructs the transition interval on buffered audio. Periodic
absolute resync bounds the effect of a lost START or END event. An orphan END is
ignored rather than inventing a transition.

The executable contract and loss-recovery regression are
`scripts/viseme_sidechannel.py` and `scripts/test_phone_viseme_baseline.py`.
Godot implements the same byte layout and buffered decoder in
`godot-demo/viseme_utils.gd`; `phone_mapper_smoke.tscn` repeats the same
transition and lost-START/resync examples at runtime. The older ID+confidence
byte used by the timeline plot is explicitly named a **preview byte** and is
not a wire format.
