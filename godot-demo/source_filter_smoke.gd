extends SceneTree


func _init() -> void:
	if not ClassDB.class_exists("SourceFilterFrontend"):
		assert(GDExtensionManager.load_extension(
				"res://addons/vizemes_mel/vizemes_mel.gdextension") == OK)
	var frontend: Object = ClassDB.instantiate("SourceFilterFrontend")
	assert(frontend != null)
	assert(frontend.call("get_sample_rate") == 16000)
	assert(frontend.call("get_window_samples") == 400)
	assert(frontend.call("get_hop_samples") == 160)
	var pcm := PackedFloat32Array()
	pcm.resize(frontend.call("get_window_samples"))
	for index in pcm.size():
		pcm[index] = 0.4 * sin(TAU * 200.0 * index / frontend.call("get_sample_rate"))
	var result: Dictionary = frontend.call("analyze_frame", pcm)
	assert(not result.is_empty())
	assert(result.pitch_valid)
	assert(absf(result.pitch_hz - 200.0) < 4.0)
	assert(result.reflection.size() == 16)
	assert(result.envelope_db.size() == 80)
	print("SOURCE_FILTER_GODOT_OK pitch=", result.pitch_hz,
			" periodicity=", result.periodicity)
	quit()
