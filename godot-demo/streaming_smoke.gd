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
	print(
		"streaming_contexts=%d batch_contexts=%d logits=%d" % [
			stream_ctx.size(), batch_ctx.size(), logits.size()
		]
	)
	print("GODOT_STREAMING_SMOKE_OK")
	get_tree().quit(0)
