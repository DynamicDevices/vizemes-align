extends Node
## End-to-end lipsync shape: wav → mel → ONNX → OVR → VisemeSystem (or Stub).

const VisemePipelineScript := preload("res://viseme_pipeline.gd")
const VisemeUtils := preload("res://viseme_utils.gd")
const VisemeTarget := preload("res://viseme_target.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")


func _ready() -> void:
	var paths := ClipProbeIo.resolve_ci_smoke_paths()
	var onnx_path := str(paths.get("onnx", ""))
	var wav_path := ClipProbeIo.models_abs().path_join("fixtures/ci-fixture.wav")

	var pipe = VisemePipelineScript.new()
	if not pipe.setup(onnx_path):
		get_tree().quit(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	var target := VisemeTarget.resolve(self)
	if target == null:
		get_tree().quit(1)
		return

	var frames := 0
	var last_ovr := PackedFloat32Array()
	for ctx_variant in pipe.mel.build_utterance_contexts(pcm):
		var ctx: PackedFloat32Array = ctx_variant
		var ovr := pipe.predict_ovr(ctx)
		if ovr.is_empty():
			push_error("predict failed")
			get_tree().quit(1)
			return
		VisemeTarget.feed(target, ovr)
		last_ovr = ovr
		frames += 1

	var call_count: int = frames
	if "call_count" in target:
		call_count = int(target.get("call_count"))
	if frames <= 0 or call_count < frames:
		# Real VisemeSystem has no call_count — trust frames > 0
		if target.name != "VisemeSystem" or frames <= 0:
			push_error("viseme feed mismatch")
			get_tree().quit(1)
			return

	var top := VisemeUtils.argmax(last_ovr)
	print(
		"lipsync_frames=%d ovr_top=%s weight=%.4f target=%s" % [
			frames, VisemeUtils.OVR_NAMES[top], last_ovr[top], target.name
		]
	)
	print("GODOT_LIPSYNC_SMOKE_OK")
	get_tree().quit(0)
