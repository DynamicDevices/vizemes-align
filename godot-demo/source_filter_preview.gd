extends Control

const DEFAULT_FIXTURE := "res://addons/vizeme-onnxmodels/fixtures/timeline_1320-122617-0010.wav"
const SCALAR_COUNT := 8
const EXTENSION_PATH := "res://addons/vizemes_mel/vizemes_mel.gdextension"

@onready var path_edit: LineEdit = %PathEdit
@onready var status_label: Label = %StatusLabel
@onready var envelope_view: TextureRect = %EnvelopeView
@onready var scalar_view: TextureRect = %ScalarView
@onready var file_dialog: FileDialog = %FileDialog

var frontend: Object


func _ready() -> void:
	path_edit.text = _command_line_fixture()
	if not ClassDB.class_exists("SourceFilterFrontend"):
		var error := GDExtensionManager.load_extension(EXTENSION_PATH)
		if error != OK and error != ERR_ALREADY_IN_USE:
			status_label.text = "Could not load Vizemes extension: %s" % error_string(error)
			return
	frontend = ClassDB.instantiate("SourceFilterFrontend")
	if frontend == null:
		status_label.text = "SourceFilterFrontend is unavailable"
		return
	_load_path(path_edit.text)


func _command_line_fixture() -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--source-filter-wav="):
			return argument.trim_prefix("--source-filter-wav=")
	return DEFAULT_FIXTURE


func _on_load_pressed() -> void:
	_load_path(path_edit.text)


func _on_browse_pressed() -> void:
	file_dialog.popup_centered_ratio(0.75)


func _on_file_selected(path: String) -> void:
	path_edit.text = path
	_load_path(path)


func _load_path(path: String) -> void:
	if frontend == null:
		return
	var load_path := path
	if not path.is_absolute_path() and not path.begins_with("res://") and not path.begins_with("user://"):
		load_path = ProjectSettings.globalize_path("res://" + path.trim_prefix("godot-demo/"))
	var stream := AudioStreamWAV.load_from_file(load_path)
	if stream == null:
		status_label.text = "Could not read WAV: %s" % path
		return
	if stream.format != AudioStreamWAV.FORMAT_16_BITS or stream.stereo or stream.mix_rate != 16000:
		status_label.text = "Expected uncompressed mono 16-bit PCM WAV at 16 kHz"
		return
	var bytes := stream.data
	var pcm := PackedFloat32Array()
	pcm.resize(bytes.size() / 2)
	for index in pcm.size():
		pcm[index] = float(bytes.decode_s16(index * 2)) / 32768.0
	var analysis: Dictionary = frontend.call("analyze_utterance", pcm)
	if analysis.is_empty():
		status_label.text = "Audio is shorter than one 25 ms analysis window"
		return
	_upload_envelope(analysis)
	_upload_scalars(analysis)
	var seconds := float(pcm.size()) / 16000.0
	status_label.text = "%d frames, %.2f s — 25 ms support, 10 ms hop, LPC order 16" % [analysis["frames"], seconds]
	print("SOURCE_FILTER_PREVIEW_OK ", path, " ", status_label.text)


func _upload_envelope(analysis: Dictionary) -> void:
	var frames: int = analysis["frames"]
	var source: PackedFloat32Array = analysis["envelope_db"]
	var transposed := PackedFloat32Array()
	transposed.resize(frames * 80)
	for frame in frames:
		for band in 80:
			transposed[band * frames + frame] = source[frame * 80 + band]
	var image := Image.create_from_data(frames, 80, false, Image.FORMAT_RF, transposed.to_byte_array())
	var texture := ImageTexture.create_from_image(image)
	var material := ShaderMaterial.new()
	material.shader = preload("res://source_filter_envelope.gdshader")
	material.set_shader_parameter("measurements", texture)
	envelope_view.material = material


func _upload_scalars(analysis: Dictionary) -> void:
	var frames: int = analysis["frames"]
	var source: PackedFloat32Array = analysis["scalars"]
	var normalized := PackedFloat32Array()
	normalized.resize(frames * SCALAR_COUNT)
	for frame in frames:
		var offset := frame * SCALAR_COUNT
		normalized[0 * frames + frame] = clamp((source[offset] + 60.0) / 60.0, 0.0, 1.0)
		normalized[1 * frames + frame] = clamp(source[offset + 1] / 30.0, 0.0, 1.0)
		normalized[2 * frames + frame] = clamp(source[offset + 2], 0.0, 1.0)
		var pitch := source[offset + 3]
		normalized[3 * frames + frame] = clamp(log(maxf(pitch, 70.0) / 70.0) / log(400.0 / 70.0), 0.0, 1.0)
		normalized[4 * frames + frame] = clamp(source[offset + 4], 0.0, 1.0)
		normalized[5 * frames + frame] = clamp(source[offset + 5], 0.0, 1.0)
		normalized[6 * frames + frame] = clamp((source[offset + 6] + 5.0) / 35.0, 0.0, 1.0)
		normalized[7 * frames + frame] = clamp((source[offset + 7] + 18.0) / 36.0, 0.0, 1.0)
	var image := Image.create_from_data(frames, SCALAR_COUNT, false, Image.FORMAT_RF, normalized.to_byte_array())
	var texture := ImageTexture.create_from_image(image)
	var material := ShaderMaterial.new()
	material.shader = preload("res://source_filter_lines.gdshader")
	material.set_shader_parameter("measurements", texture)
	scalar_view.material = material
