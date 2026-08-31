# Phone recognition A/B plan

This experiment keeps the working direct Mel-to-viseme TCN as model A and adds
an explicit two-stage path:

1. model B1: causal log-Mel-to-phone TCN;
2. deterministic phone-posterior-to-viseme table plus 60 ms causal smoothing;
3. model B2 (challenger only): small causal learned phone-posterior-to-viseme
   ONNX mapper.

MFA provides the training phone intervals, not runtime phone probabilities.
Those probabilities are the softmax output of B1. MFA's legacy Kaldi acoustic
path uses 25 ms / 10 ms MFCC features, CMVN, contextual GMM/HMM phone states and
speaker transforms; MFCCs are derived from a Mel filterbank but are not the same
as this project's 80-bin log-Mel input. See the [MFA paper][mfa-paper],
[current MFA feature/training configuration][mfa-config], and [MFA acoustic
model reference][mfa-model].

End-to-end recognisers do not necessarily expose a phone layer. Hybrid HMM
systems do; CTC/RNN-T/encoder-decoder systems commonly emit characters,
word-pieces or text tokens from learned acoustic representations. A representation
model such as [wav2vec 2.0][wav2vec] can instead be fine-tuned with an explicit
phoneme CTC head. Vizemes uses its own small phone head so the runtime boundary
and probability contract is explicit and causal.

## Reproduce

Build the MFA-derived phone targets once:

```sh
python3 scripts/build_phone_tensors.py --subset test-clean
```

Train B1 with approximately the same default TCN width, depth and ten-minute
wall-clock budget as the direct TCN:

```sh
python3 scripts/train_phone_tcn.py \
  --subset test-clean \
  --channels 128 --layers 5 --wall-seconds 600 \
  --output export/tier-c/phone.onnx
```

The command reports phone frame accuracy, stress-collapsed PER and phone
boundary timing. It also maps B1's posterior stream through the deterministic
table and reports final viseme frame accuracy, boundary error, transition ratio
and excess-transition jitter.

Train and evaluate the learned Stage-B challenger on the same B1 outputs:

```sh
python3 scripts/train_viseme_mapper.py \
  --phone-onnx export/tier-c/phone.onnx \
  --subset test-clean --wall-seconds 300 \
  --output export/tier-c/viseme_map.onnx
```

Both ONNX files embed schema-2 model identity, complete vocabularies, input and
normalisation contracts, architecture, provenance and validation metrics. The
learned mapper also embeds the SHA-256 identity of its upstream phone ONNX.

The phone ONNX additionally embeds the complete deterministic stage-B contract
in `phone_to_viseme`: its 15-viseme vocabulary, one viseme ID for every phone
output column, mapping-file SHA-256/version, hop, causal smoothing constant and
top-k sparsity. Godot consumes that object from `vizemes_meta_json`; it does not
load the training mapping JSON at runtime. `phone_mapper_smoke.tscn` gates
posterior summation, the absence of a false P-to-B visual boundary, causal
smoothing and top-2 normalisation. The normal TCN timeline detects the contract
and maps phone logits before the existing face/timeline presentation path.

## Decision rule

The learned mapper replaces the deterministic table only if held-out evidence
shows a material visual improvement without a latency or stability regression:

- lower mean and p95 viseme boundary error;
- transition ratio closer to 1.0 and no higher excess jitter;
- equal or better viseme frame accuracy;
- zero additional lookahead and acceptable ONNX runtime cost;
- Julian prefers it in the same Godot clips during blind/manual A/B review.

A bounded 30-second engineering run over 128 utterances proved the comparison
path works but rejected the current learned challenger: frame accuracy rose from
0.522 to 0.557, while transition ratio worsened from 1.25 to 2.95 and mean
boundary error from 1.28 s to 4.45 s. This is not the final ten-minute model
result; it is evidence that the deterministic mapper remains the baseline until
the full controlled run and Godot review.

## Controlled ten-minute result

The final common evaluator uses the same 262 held-out utterances (190,492 Mel
frames) and the same MFA-phone-to-viseme reference for both paths. Both acoustic
models used 128 channels, five causal TCN layers and a 600-second training
budget. The direct model has 505,103 parameters; the phone model has 266,949.

| Metric | Direct viseme TCN | Phone TCN + deterministic table |
|---|---:|---:|
| Viseme frame accuracy | 0.4859 | 0.5237 |
| Boundary mean absolute error | 2564 ms | 424 ms |
| Boundary p95 absolute error | 7510 ms | 1380 ms |
| Transition ratio (ideal 1.0) | 2.179 | 0.994 |
| Excess transition jitter | 11.27 Hz | 0.00 Hz |
| Algorithmic lookahead | 50 ms | 0 ms |
| CPU real-time factor, sessions preloaded | 0.00113 | 0.00453 |

The phone model's raw stress-collapsed PER is still 2.10, so this is evidence
for the final visual mapping, not a claim of production-quality speech
recognition. Reproduce the common comparison with:

```sh
python3 scripts/eval_direct_phone_ab.py \
  --direct-onnx godot-demo/addons/vizeme-onnxmodels/tier-b-tcn/model_final.onnx \
  --phone-onnx godot-demo/addons/vizeme-onnxmodels/tier-c/model_final.onnx \
  --output docs/benchmarks/direct_phone_ab.json
```

The checked-in JSON is the exact report. Godot's headless timeline has loaded
both models over the same clip and printed `GODOT_VISEME_TIMELINE_OK`; Julian's
visual/manual preference is the remaining acceptance gate.

[mfa-paper]: https://montreal-forced-aligner.readthedocs.io/en/v3.3.0/_downloads/998b0c31eadaf048e8e3de805b9ef8e6/MFA_paper_Interspeech2017.pdf
[mfa-config]: https://montreal-forced-aligner.readthedocs.io/en/v3.2.3/user_guide/configuration/acoustic_modeling.html
[mfa-model]: https://montreal-forced-aligner.readthedocs.io/en/stable/reference/acoustic_modeling/index.html
[wav2vec]: https://arxiv.org/abs/2006.11477
