extends SceneTree
## Headless: load TCN then ci-smoke; print diagnostics / catch abort.

func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	if not ClassDB.class_exists("OnnxLoader"):
		push_error("OnnxLoader missing")
		quit(2)
		return
	var tcn := ProjectSettings.globalize_path(
		"res://addons/vizeme-onnxmodels/tier-b-tcn/model_final.onnx"
	)
	var smoke := ProjectSettings.globalize_path(
		"res://addons/vizeme-onnxmodels/ci-smoke/model.onnx"
	)
	print("TCN_PATH ", tcn)
	print("SMOKE_PATH ", smoke)
	_try_load("tcn", tcn)
	_try_load("ci-smoke", smoke)
	print("GODOT_TCN_LOAD_PROBE_OK")
	quit(0)


func _try_load(tag: String, path: String) -> void:
	var loader = ClassDB.instantiate("OnnxLoader")
	print("DIAG_BEFORE_%s " % tag, loader.get_diagnostics())
	var ok: bool = loader.load_model(path)
	print("LOAD_%s ok=%s diag=%s" % [tag, ok, loader.get_diagnostics()])
	loader.unload_model()
	print("UNLOAD_%s done" % tag)
