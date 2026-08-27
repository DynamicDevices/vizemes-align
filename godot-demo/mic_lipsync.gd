extends Node
## Live mic → push_pcm_contexts → ONNX → VisemeSystem or Stub (editor / GUI run).

const VisemePipelineScript := preload("res://viseme_pipeline.gd")
const VisemeUtils := preload("res://viseme_utils.gd")
const VisemeTarget := preload("res://viseme_target.gd")
const RECORD_BUS := "VizemesRecord"
const TARGET_RATE := 16000

@onready var _player: AudioStreamPlayer = $AudioStreamPlayer
@onready var _label: Label = $Label

var _capture: AudioEffectCapture
var _pipe
var _target: Node
var _frames: int = 0
var _last_ovr: PackedFloat32Array = PackedFloat32Array()


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
	_setup_record_bus()
	_player.stream = AudioStreamMicrophone.new()
	_player.bus = RECORD_BUS
	_player.play()
	_label.text = "Listening… (mic → visemes via %s)" % _target.name


func _setup_record_bus() -> void:
	if AudioServer.get_bus_index(RECORD_BUS) >= 0:
		var idx := AudioServer.get_bus_index(RECORD_BUS)
		_capture = AudioServer.get_bus_effect(idx, 0) as AudioEffectCapture
		return
	AudioServer.add_bus()
	var idx := AudioServer.bus_count - 1
	AudioServer.set_bus_name(idx, RECORD_BUS)
	_capture = AudioEffectCapture.new()
	AudioServer.add_bus_effect(idx, _capture)


func _process(_delta: float) -> void:
	if _capture == null or _pipe == null or _target == null:
		return
	var avail := _capture.get_frames_available()
	if avail <= 0:
		return
	var buffer: PackedVector2Array = _capture.get_buffer(avail)
	var pcm := VisemeUtils.stereo_to_mono(buffer)
	pcm = VisemeUtils.resample_pcm(pcm, int(AudioServer.get_mix_rate()), TARGET_RATE)
	_frames += _pipe.feed_pcm_mono_16k(pcm, _target)
	if _pipe.last_ovr.size() > 0:
		_last_ovr = _pipe.last_ovr
	if _last_ovr.size() > 0:
		var top := VisemeUtils.argmax(_last_ovr)
		_label.text = "viseme=%s  frames=%d  via=%s" % [
			VisemeUtils.OVR_NAMES[top], _frames, _target.name
		]
