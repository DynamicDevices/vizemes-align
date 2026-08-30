extends Node
## Minimal mic → MelFrontend contexts (Julian mid=829).
## Shout into the mic; printed RMS / peak of each context should move.

const VisemeUtils := preload("res://viseme_utils.gd")

var _mel: Object
var _frames := 0
var _ctx_n := 0


func _ready() -> void:
	if not ProjectSettings.get_setting("audio/driver/enable_input", false):
		ProjectSettings.set_setting("audio/driver/enable_input", true)
	var err := AudioServer.set_input_device_active(true)
	if err != OK:
		push_error("mic failed: %s" % error_string(err))
		return

	var paths := ClipProbeIo.resolve_model_paths(false)
	var onnx_path := str(paths.get("onnx", ""))
	var json_path := str(paths.get("json", ""))

	_mel = ClassDB.instantiate("MelFrontend")
	if _mel == null:
		push_error("MelFrontend missing")
		return
	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null or onnx_path.is_empty() or not loader.load_model(onnx_path):
		push_error("OnnxLoader / model missing under res://addons/vizeme-onnxmodels")
		return
	if not VisemeUtils.configure_mel_from_onnx(_mel, loader, json_path):
		push_error("configure failed: %s" % onnx_path)
		return
	if _mel.has_method("configure_preprocess"):
		_mel.configure_preprocess(true, true, true, 8000.0, false, 10)
	_mel.begin_stream()
	print("mic_context_print ready — shout; contexts print as produced (%s)" % onnx_path)


func _process(_dt: float) -> void:
	if _mel == null:
		return
	var avail := AudioServer.get_input_frames_available()
	if avail <= 0:
		return
	var n := mini(avail, 4096)
	var frames: PackedVector2Array = AudioServer.get_input_frames(n)
	var mix := int(round(AudioServer.get_input_mix_rate()))
	_mel.push_pcm_stereo(frames, mix)
	_frames += frames.size()
	while _mel.count_available_contexts() > 0:
		var ctx: PackedFloat32Array = _mel.get_next_context()
		_ctx_n += 1
		var peak := 0.0
		var sum := 0.0
		for v in ctx:
			var a := absf(v)
			sum += a
			if a > peak:
				peak = a
		var mean := sum / float(maxi(1, ctx.size()))
		var vad_s := "?"
		if _mel.has_method("get_last_vad"):
			vad_s = str(_mel.get_last_vad())
		print(
			"ctx=%d frames=%d peak=%.3f mean=%.3f lag=%.3fs queued=%d vad=%s" % [
				_ctx_n, _frames, peak, mean, _mel.last_context_time_offset(),
				_mel.count_available_contexts(), vad_s
			]
		)


func _exit_tree() -> void:
	AudioServer.set_input_device_active(false)
