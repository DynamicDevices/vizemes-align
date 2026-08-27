extends Node
## End-to-end lipsync shape: wav → mel → ONNX → OVR visemes → VisemeSystemStub.set_visemes.

const VisemePipelineScript := preload("res://viseme_pipeline.gd")
const VisemeUtils := preload("res://viseme_utils.gd")


func _ready() -> void:
	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	var json_path := root.path_join("export/ci-smoke/model.json")
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var wav_path := root.path_join("export/ci-smoke/ci-fixture.wav")

	var pipe = VisemePipelineScript.new()
	if not pipe.setup(json_path, onnx_path):
		get_tree().quit(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	var stub = $VisemeSystemStub
	var frames := 0
	for ctx_variant in pipe.mel.build_utterance_contexts(pcm):
		var ctx: PackedFloat32Array = ctx_variant
		var ovr := pipe.predict_ovr(ctx)
		if ovr.is_empty():
			push_error("predict failed")
			get_tree().quit(1)
			return
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
