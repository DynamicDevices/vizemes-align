extends SceneTree


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var arguments := OS.get_cmdline_user_args()
	assert(not arguments.is_empty())
	if not ClassDB.class_exists("OnnxLoader"):
		assert(GDExtensionManager.load_extension(
				"res://addons/onnx_loader/onnx_loader.gdextension") == OK)
	var loader: Object = ClassDB.instantiate("OnnxLoader")
	assert(loader != null)
	assert(loader.call("load_model", arguments[0]))
	var features := PackedFloat32Array()
	features.resize(32 * 24)
	var logits: PackedFloat32Array = loader.call(
			"predict_shaped", features, PackedInt32Array([1, 32, 24]))
	assert(logits.size() == 32 * 15)
	print("SOURCE_FILTER_ONNX_GODOT_OK logits=", logits.size())
	quit()
