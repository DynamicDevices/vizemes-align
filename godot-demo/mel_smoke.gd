extends Node
## MelFrontend + OnnxLoader end-to-end: ci-fixture.wav → mel contexts → viseme logits.

const CHUNK := 1600  # reserved for future streaming/mic path


func _softmax(logits: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(logits.size())
	if logits.is_empty():
		return out
	var m := logits[0]
	for i in range(1, logits.size()):
		if logits[i] > m:
			m = logits[i]
	var sum := 0.0
	for i in logits.size():
		out[i] = exp(logits[i] - m)
		sum += out[i]
	if sum <= 0.0:
		sum = 1.0
	for i in logits.size():
		out[i] /= sum
	return out


func _load_wav_pcm(path: String) -> PackedFloat32Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("open wav failed: %s" % path)
		return PackedFloat32Array()
	if f.get_buffer(4).get_string_from_ascii() != "RIFF":
		push_error("not RIFF wav: %s" % path)
		return PackedFloat32Array()
	f.seek(8)
	var fmt_found := false
	var channels := 1
	var rate := 16000
	var bits := 16
	while f.get_position() < f.get_length() - 8:
		var tag := f.get_buffer(4).get_string_from_ascii()
		var chunk_sz := f.get_32()
		if tag == "fmt ":
			fmt_found = true
			var _audio_fmt := f.get_16()
			channels = f.get_16()
			rate = f.get_32()
			f.get_32() # byte rate
			f.get_16() # block align
			bits = f.get_16()
			if chunk_sz > 16:
				f.seek(f.get_position() + chunk_sz - 16)
		elif tag == "data":
			if not fmt_found:
				push_error("wav missing fmt chunk: %s" % path)
				return PackedFloat32Array()
			if channels != 1 or bits != 16:
				push_error("expected mono 16-bit wav sr=%d ch=%d bits=%d" % [rate, channels, bits])
				return PackedFloat32Array()
			var raw := f.get_buffer(chunk_sz)
			var n := raw.size() / 2
			var pcm := PackedFloat32Array()
			pcm.resize(n)
			for i in n:
				var s := raw[i * 2] | (raw[i * 2 + 1] << 8)
				if s >= 32768:
					s -= 65536
				pcm[i] = float(s) / 32768.0
			return pcm
		else:
			f.seek(f.get_position() + chunk_sz)
	push_error("wav missing data chunk: %s" % path)
	return PackedFloat32Array()


func _ready() -> void:
	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	var json_path := root.path_join("export/ci-smoke/model.json")
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var wav_path := root.path_join("export/ci-smoke/ci-fixture.wav")

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		push_error("MelFrontend missing — build gdextension .so")
		get_tree().quit(1)
		return
	if not mel.configure_from_json(json_path):
		push_error("MelFrontend configure failed: %s" % json_path)
		get_tree().quit(1)
		return

	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null:
		push_error("OnnxLoader missing — build onnx_loader addon")
		get_tree().quit(1)
		return
	if not loader.load_model(onnx_path):
		push_error("OnnxLoader load_model failed: %s" % onnx_path)
		get_tree().quit(1)
		return

	if mel.get_input_features() != loader.get_input_size():
		push_error("feature dim mismatch mel=%d onnx=%d" % [
			mel.get_input_features(), loader.get_input_size()])
		get_tree().quit(1)
		return

	var pcm := _load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	var batch: Array = mel.build_utterance_contexts(pcm)
	if batch.is_empty():
		push_error("no mel contexts from ci-fixture.wav")
		get_tree().quit(1)
		return

	var contexts := 0
	for ctx_variant in batch:
		var ctx: PackedFloat32Array = ctx_variant
		if ctx.size() != mel.get_input_features():
			push_error("bad context size %d" % ctx.size())
			get_tree().quit(1)
			return
		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			push_error("predict failed at context %d" % contexts)
			get_tree().quit(1)
			return
		var _w := _softmax(logits)
		contexts += 1

	if contexts <= 0:
		push_error("no mel contexts from ci-fixture.wav")
		get_tree().quit(1)
		return

	print("mel_contexts=%d logits=%d" % [contexts, loader.get_output_size()])
	print("GODOT_MEL_ONNX_SMOKE_OK")
	get_tree().quit(0)
