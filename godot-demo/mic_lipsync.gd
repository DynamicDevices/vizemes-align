extends Node
## Live mic → push_pcm_contexts → ONNX → VisemeSystem or Stub (editor / GUI run).
## Godot 4.6+: AudioServer.get_input_frames + get_input_mix_rate (no capture bus).

const VisemePipelineScript := preload("res://viseme_pipeline.gd")
const VisemeUtils := preload("res://viseme_utils.gd")
const VisemeTarget := preload("res://viseme_target.gd")
const TARGET_RATE := 16000
## Max frames to pull per tick (keeps _process bounded).
const PULL_MAX := 4096

@onready var _label: Label = $Label

var _pipe
var _target: Node
var _frames: int = 0
var _last_ovr: PackedFloat32Array = PackedFloat32Array()
var _input_ok := false
var _hard_bytes := PackedByteArray()
var _hard_acc_pcm := PackedFloat32Array()
var _hard_need := 320 ## 20 ms @ 16 kHz


func _ready() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("mic_lipsync requires a display and audio device (not headless)")
		get_tree().quit(1)
		return

	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	_pipe = VisemePipelineScript.new()
	if not _pipe.setup(
		root.path_join("export/ci-smoke/model.json"),
		root.path_join("export/ci-smoke/model.onnx")
	):
		get_tree().quit(1)
		return
	_target = VisemeTarget.resolve(self)
	if _target == null:
		get_tree().quit(1)
		return
	_pipe.begin_stream()

	# Direct mic buffer (Godot 4.6). Needs ProjectSettings audio/driver/enable_input.
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


func _exit_tree() -> void:
	if _input_ok:
		AudioServer.set_input_device_active(false)


func _process(_delta: float) -> void:
	if not _input_ok or _pipe == null or _target == null:
		return
	var avail := AudioServer.get_input_frames_available()
	if avail <= 0:
		return
	var n := mini(avail, PULL_MAX)
	var buffer: PackedVector2Array = AudioServer.get_input_frames(n)
	if buffer.is_empty():
		return
	var pcm := VisemeUtils.stereo_to_mono(buffer)
	var in_rate := int(round(AudioServer.get_input_mix_rate()))
	pcm = VisemeUtils.resample_pcm(pcm, in_rate, TARGET_RATE)
	_frames += _pipe.feed_pcm_mono_16k(pcm, _target)
	# Accumulate 20 ms chunks → hard byte stream (VoIP sideband shape).
	for s in pcm:
		_hard_acc_pcm.append(s)
	while _hard_acc_pcm.size() >= _hard_need:
		if _pipe.last_ovr.size() > 0:
			_hard_bytes.append(VisemeUtils.soft_to_hard_byte(_pipe.last_ovr))
		var keep := PackedFloat32Array()
		for i in range(_hard_need, _hard_acc_pcm.size()):
			keep.append(_hard_acc_pcm[i])
		_hard_acc_pcm = keep
	if _pipe.last_ovr.size() > 0:
		_last_ovr = _pipe.last_ovr
	if _last_ovr.size() > 0:
		var top := VisemeUtils.argmax(_last_ovr)
		var hb := 0
		if not _hard_bytes.is_empty():
			hb = _hard_bytes[_hard_bytes.size() - 1]
		_label.text = "viseme=%s  frames=%d  hard=%dB last=0x%02X  via=%s  in=%.0fHz" % [
			VisemeUtils.OVR_NAMES[top], _frames, _hard_bytes.size(), hb, _target.name,
			AudioServer.get_input_mix_rate()
		]
