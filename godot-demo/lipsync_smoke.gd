extends Node
## End-to-end lipsync shape: wav → mel → ONNX → OVR visemes → VisemeSystemStub.set_visemes.

const VisemeUtils := preload("res://viseme_utils.gd")


func _ready() -> void:
	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	var json_path := root.path_join("export/ci-smoke/model.json")
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var wav_path := root.path_join("export/ci-smoke/ci-fixture.wav")

	var id_to_name: Array = VisemeUtils.load_id_to_name(json_path)
	if id_to_name.is_empty():
		push_error("model.json visemes missing")
		get_tree().quit(1)
		return

	var mel = ClassDB.instantiate("MelFrontend")
	var loader = ClassDB.instantiate("OnnxLoader")
	if mel == null or loader == null:
		push_error("MelFrontend or OnnxLoader missing")
		get_tree().quit(1)
		return
	if not mel.configure_from_json(json_path) or not loader.load_model(onnx_path):
		get_tree().quit(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	var contexts: Array = mel.build_utterance_contexts(pcm)
	if contexts.is_empty():
		push_error("no mel contexts")
		get_tree().quit(1)
		return

	var stub = $VisemeSystemStub
	var frames := 0
	for ctx_variant in contexts:
		var ctx: PackedFloat32Array = ctx_variant
		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			push_error("predict failed")
			get_tree().quit(1)
			return
		var w := VisemeUtils.softmax(logits)
		var ovr := VisemeUtils.mlp_to_ovr(w, id_to_name)
		stub.set_visemes(ovr)
		frames += 1

	if frames <= 0 or stub.call_count != frames:
		push_error("viseme feed mismatch")
		get_tree().quit(1)
		return

	var top := VisemeUtils.argmax(stub.last_ovr)
	print(
		"lipsync_frames=%d ovr_top=%s weight=%.4f" % [
			frames, VisemeUtils.OVR_NAMES[top], stub.last_ovr[top]
		]
	)
	print("GODOT_LIPSYNC_SMOKE_OK")
	get_tree().quit(0)
