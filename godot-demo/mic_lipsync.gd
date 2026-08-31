extends Control
## Live mic → MelFrontend.push_pcm_stereo → ONNX → VisemeSystem or Stub.
## Godot 4.6+: AudioServer.get_input_frames + get_input_mix_rate (no capture bus).
## Speex AGC/VAD/denoise on by default; CheckButton / G gates silence.

const VisemePipelineScript := preload("res://viseme_pipeline.gd")
const VisemeUtils := preload("res://viseme_utils.gd")
const VisemeTarget := preload("res://viseme_target.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")
const PULL_MAX := 4096

@onready var _label: Label = %Label
@onready var _vad_gate: CheckButton = %VadGate
@onready var _aec: CheckButton = %AecCheck

var _pipe
var _target: Node
var _frames: int = 0
var _last_ovr: PackedFloat32Array = PackedFloat32Array()
var _input_ok := false
var _hard_bytes := PackedByteArray()
var _gate_on := false
var _aec_on := false
var _far_capture: AudioEffectCapture


func _ready() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("mic_lipsync requires a display and audio device (not headless)")
		get_tree().quit(1)
		return

	_pipe = VisemePipelineScript.new()
	var paths: Dictionary = ClipProbeIo.resolve_model_paths(false)
	var onnx_path := str(paths.get("onnx", ""))
	if onnx_path.is_empty():
		push_error("mic_lipsync: no ONNX under res://addons/vizeme-onnxmodels — run scripts/sync_vizeme_onnxmodels.sh")
		get_tree().quit(1)
		return
	if not _pipe.setup(onnx_path):
		push_error("mic_lipsync: setup failed for %s" % onnx_path)
		get_tree().quit(1)
		return
	_label.text = "Listening… model=%s via %s" % [paths.get("id", "?"), onnx_path.get_file()]
	_target = VisemeTarget.resolve(self)
	if _target == null:
		get_tree().quit(1)
		return
	_pipe.begin_stream()

	_vad_gate.button_pressed = false
	_vad_gate.toggled.connect(_on_vad_gate_toggled)
	_aec.button_pressed = false
	_aec.toggled.connect(_on_aec_toggled)

	if not ProjectSettings.get_setting("audio/driver/enable_input", false):
		ProjectSettings.set_setting("audio/driver/enable_input", true)
	AudioServer.input_device = "Default"
	var err := AudioServer.set_input_device_active(true)
	_input_ok = err == OK
	if not _input_ok:
		_label.text = "Mic input failed (check OS permission / audio/driver/enable_input)"
		push_error("AudioServer.set_input_device_active(true) → %s" % error_string(err))
		return
	_label.text = "Listening… (mic → visemes via %s @ %.0f Hz in)" % [
		_target.name, AudioServer.get_input_mix_rate()
	]


func _ensure_far_capture() -> void:
	## Capture Master output as AEC far-end (whatever Godot is playing).
	if _far_capture != null:
		return
	var master := AudioServer.get_bus_index(&"Master")
	if master < 0:
		return
	_far_capture = AudioEffectCapture.new()
	AudioServer.add_bus_effect(master, _far_capture)


func _on_vad_gate_toggled(pressed: bool) -> void:
	_gate_on = pressed
	if _pipe != null and _pipe.has_method("set_vad_gate"):
		_pipe.set_vad_gate(pressed)
		_pipe.begin_stream()


func _on_aec_toggled(pressed: bool) -> void:
	_aec_on = pressed
	if _pipe == null or not _pipe.has_method("set_aec"):
		return
	if pressed:
		_ensure_far_capture()
	_pipe.set_aec(pressed)
	_pipe.begin_stream()


func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_G:
			_vad_gate.button_pressed = not _vad_gate.button_pressed
		elif event.keycode == KEY_A:
			_aec.button_pressed = not _aec.button_pressed


func _exit_tree() -> void:
	if _input_ok:
		AudioServer.set_input_device_active(false)


func _process(_delta: float) -> void:
	if not _input_ok or _pipe == null or _target == null:
		return
	if _aec_on and _far_capture != null:
		var far_n: int = _far_capture.get_frames_available()
		if far_n > 0:
			var far_buf: PackedVector2Array = _far_capture.get_buffer(mini(far_n, PULL_MAX))
			if not far_buf.is_empty():
				_pipe.feed_far_end_stereo(far_buf, int(round(AudioServer.get_mix_rate())))
	var avail := AudioServer.get_input_frames_available()
	if avail <= 0:
		return
	var n := mini(avail, PULL_MAX)
	var buffer: PackedVector2Array = AudioServer.get_input_frames(n)
	if buffer.is_empty():
		return
	var in_rate := int(round(AudioServer.get_input_mix_rate()))
	# Explicit type: _pipe is untyped, so := cannot infer feed_pcm_stereo's int.
	var produced: int = _pipe.feed_pcm_stereo(buffer, in_rate, _target)
	_frames += produced
	for _i in produced:
		if _pipe.last_ovr.size() > 0:
			_hard_bytes.append(VisemeUtils.soft_to_preview_byte(_pipe.last_ovr))
	if _pipe.last_ovr.size() > 0:
		_last_ovr = _pipe.last_ovr
	var vad_s: String = "?"
	if _pipe.has_method("last_vad"):
		vad_s = str(_pipe.last_vad())
	var gate_s := "on" if _gate_on else "off"
	var aec_s := "on" if _aec_on else "off"
	if _last_ovr.size() > 0:
		var top := VisemeUtils.argmax(_last_ovr)
		var hb := 0
		if not _hard_bytes.is_empty():
			hb = _hard_bytes[_hard_bytes.size() - 1]
		_label.text = "viseme=%s  frames=%d  hard=%dB last=0x%02X  via=%s  in=%.0fHz  vad=%s  gate=%s  aec=%s" % [
			VisemeUtils.OVR_NAMES[top], _frames, _hard_bytes.size(), hb, _target.name,
			AudioServer.get_input_mix_rate(), vad_s, gate_s, aec_s
		]
	else:
		_label.text = "Listening… frames=%d  vad=%s  gate=%s  aec=%s  (silence dropped when gate on)" % [
			_frames, vad_s, gate_s, aec_s
		]
