extends Node
## Chunked PCM → push_pcm / get_next_context (live mic shape) → ONNX.

const VisemeUtils := preload("res://viseme_utils.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")
const CHUNK_SAMPLES := 1600


func _ready() -> void:
	var paths := ClipProbeIo.resolve_ci_smoke_paths()
	var onnx_path := str(paths.get("onnx", ""))
	var wav_path := ClipProbeIo.models_abs().path_join("fixtures/ci-fixture.wav")

	var mel = ClassDB.instantiate("MelFrontend")
	var loader = ClassDB.instantiate("OnnxLoader")
	if mel == null or loader == null:
		push_error("MelFrontend or OnnxLoader missing")
		get_tree().quit(1)
		return
	if not loader.load_model(onnx_path) \
			or not VisemeUtils.configure_mel_from_onnx(mel, loader):
		get_tree().quit(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	mel.begin_stream()
	var stream_ctx: Array = []
	var pos := 0
	while pos < pcm.size():
		var end := mini(pos + CHUNK_SAMPLES, pcm.size())
		var chunk := pcm.slice(pos, end)
		mel.push_pcm(chunk)
		while mel.count_available_contexts() > 0:
			stream_ctx.append(mel.get_next_context())
		pos = end

	if stream_ctx.is_empty():
		push_error("stream queue produced no contexts")
		get_tree().quit(1)
		return

	var logits: PackedFloat32Array = loader.predict(stream_ctx[-1])
	if logits.is_empty():
		push_error("predict failed on streaming context")
		get_tree().quit(1)
		return

	var batch_ctx: Array = mel.build_utterance_contexts(pcm)
	# Batch uses the centred, full-utterance-normalised training transform while
	# streaming uses causal framing and running statistics. Their edge frame
	# counts may differ by at most one analysis window.
	var window_samples := int(str(loader.get_metadata_value("vizemes_window_length_samples")))
	var hop_samples := int(str(loader.get_metadata_value("vizemes_hop_length_samples")))
	var edge_frames: int = ceili(float(window_samples) / float(hop_samples))
	if absi(stream_ctx.size() - batch_ctx.size()) > edge_frames:
		push_error("streaming/batch context count drift %d vs %d (limit %d)" % [
			stream_ctx.size(), batch_ctx.size(), edge_frames
		])
		get_tree().quit(1)
		return

	# Live mix_rate change: MelFrontend should speex_resampler_set_rate, not tear down.
	var stereo := PackedVector2Array()
	stereo.resize(480)
	for i in 480:
		var a := 0.1 * sin(TAU * 440.0 * float(i) / 48000.0)
		stereo[i] = Vector2(a, a)
	mel.begin_stream()
	mel.push_pcm_stereo(stereo, 48000)
	mel.push_pcm_stereo(stereo, 44100)  # rate change → set_rate path
	var after_rate: int = mel.count_available_contexts()
	# VisemeUtils SpeexResampler.set_rate reuse
	var r1: PackedFloat32Array = VisemeUtils.resample_pcm(pcm.slice(0, 480), 16000, 8000)
	var r2: PackedFloat32Array = VisemeUtils.resample_pcm(pcm.slice(0, 480), 16000, 12000)
	if r1.is_empty() or r2.is_empty() or r2.size() <= r1.size():
		push_error("VisemeUtils.set_rate resample check failed")
		get_tree().quit(1)
		return

	# AEC: far-end tone + delayed echo in near; should not crash / should emit
	assert(mel.configure_aec(100, 10))
	mel.begin_stream()
	var far := PackedFloat32Array()
	far.resize(160)
	var near := PackedFloat32Array()
	near.resize(160)
	for n in 20:
		for i in 160:
			var t: float = 0.2 * sin(TAU * 440.0 * float(n * 160 + i) / 16000.0)
			far[i] = t
			near[i] = 0.4 * t
		mel.push_far_end_pcm(far)
		mel.push_pcm(near)
	mel.disable_aec()

	# Latency API (Julian realign): after stereo push, Speex resampler exists.
	var dsp_n: int = mel.get_dsp_latency_samples()
	var dsp_s: float = mel.get_dsp_latency_seconds()
	if mel.is_preprocess_enabled() and mel.get_preprocess_frame_size() <= 0:
		push_error("preprocess frame_size missing")
		get_tree().quit(1)
		return
	if dsp_s < 0.0:
		push_error("dsp latency negative")
		get_tree().quit(1)
		return

	print(
		"streaming_contexts=%d batch_contexts=%d logits=%d rate_change_ctx=%d dsp_lat=%d (%.4fs)" % [
			stream_ctx.size(), batch_ctx.size(), logits.size(), after_rate, dsp_n, dsp_s
		]
	)
	print("GODOT_STREAMING_SMOKE_OK")
	get_tree().quit(0)
