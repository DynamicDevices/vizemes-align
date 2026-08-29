extends Node
## Chunked PCM → push_pcm / get_next_context (live mic shape) → ONNX.

const VisemeUtils := preload("res://viseme_utils.gd")
const CHUNK_SAMPLES := 1600


func _ready() -> void:
	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	var json_path := root.path_join("export/ci-smoke/model.json")
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var wav_path := root.path_join("export/ci-smoke/ci-fixture.wav")

	var mel = ClassDB.instantiate("MelFrontend")
	var loader = ClassDB.instantiate("OnnxLoader")
	if mel == null or loader == null:
		push_error("MelFrontend or OnnxLoader missing")
		get_tree().quit(1)
		return
	if not VisemeUtils.configure_mel_from_json(mel, json_path) or not loader.load_model(onnx_path):
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
	if stream_ctx.size() != batch_ctx.size():
		push_error("streaming/batch context count mismatch %d vs %d" % [stream_ctx.size(), batch_ctx.size()])
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

	print(
		"streaming_contexts=%d batch_contexts=%d logits=%d rate_change_ctx=%d" % [
			stream_ctx.size(), batch_ctx.size(), logits.size(), after_rate
		]
	)
	print("GODOT_STREAMING_SMOKE_OK")
	get_tree().quit(0)
