extends Node
## Poll MelAudioEffect on the Master bus (Julian AudioEffect path).
## Configure the effect's MelFrontend, then print contexts as the bus plays.

const VisemeUtils := preload("res://viseme_utils.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")

var _fx: Object ## MelAudioEffect


func _ready() -> void:
	if not ClassDB.class_exists("MelAudioEffect"):
		push_error("MelAudioEffect missing — rebuild vizemes_mel")
		return
	var paths := ClipProbeIo.resolve_model_paths(false)
	var onnx_path := str(paths.get("onnx", ""))
	_fx = ClassDB.instantiate("MelAudioEffect")
	var mel = _fx.get_mel_frontend()
	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null or onnx_path.is_empty() or not loader.load_model(onnx_path):
		push_error("OnnxLoader / model missing under res://addons/vizeme-onnxmodels")
		return
	if not VisemeUtils.configure_mel_from_onnx(mel, loader):
		push_error("configure failed")
		return
	mel.begin_stream()
	var idx := AudioServer.get_bus_index("Master")
	AudioServer.add_bus_effect(idx, _fx)
	print("mel_audio_effect_print: MelAudioEffect on Master — play audio / use mic into bus (%s)" % onnx_path)


func _process(_dt: float) -> void:
	if _fx == null:
		return
	while _fx.count_available_contexts() > 0:
		var ctx: PackedFloat32Array = _fx.get_next_context()
		var peak := 0.0
		for v in ctx:
			peak = maxf(peak, absf(v))
		print("effect_ctx peak=%.3f lag=%.3fs" % [peak, _fx.last_context_time_offset()])
