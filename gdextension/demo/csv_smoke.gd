extends Node
## CSV → VizemesOnnx.predict smoke (mirrors scripts/sanity_check_onnx.py).
## Expects built extension + repo export/ci-smoke next to gdextension/.

func _ready() -> void:
	var root := ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()
	var json_path := root.path_join("export/ci-smoke/model.json")
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var csv_path := root.path_join("export/ci-smoke/demo_inputs.csv")

	var m = ClassDB.instantiate("VizemesOnnx")
	if m == null:
		push_error("VizemesOnnx class missing — is vizemes.gdextension loaded?")
		get_tree().quit(1)
		return
	if not m.load_model(json_path, onnx_path):
		push_error("load_model failed: %s" % json_path)
		get_tree().quit(1)
		return

	var nfeat: int = m.get_input_features()
	var file := FileAccess.open(csv_path, FileAccess.READ)
	if file == null:
		push_error("open csv failed: %s" % csv_path)
		get_tree().quit(1)
		return
	var header := file.get_csv_line()
	var hits := 0
	var n := 0
	print("%5s  %-8s  %-8s  hit" % ["probe", "expect", "argmax"])
	while not file.eof_reached():
		var row := file.get_csv_line()
		if row.size() < 3 + nfeat:
			continue
		var probe := int(row[0])
		var expect_id := int(row[1])
		var expect_name := str(row[2])
		var ctx := PackedFloat32Array()
		ctx.resize(nfeat)
		for i in nfeat:
			ctx[i] = float(row[3 + i])
		var w: PackedFloat32Array = m.predict(ctx)
		if w.is_empty():
			push_error("predict failed probe=%d" % probe)
			get_tree().quit(1)
			return
		var argmax := 0
		for i in range(1, w.size()):
			if w[i] > w[argmax]:
				argmax = i
		var hit := argmax == expect_id
		hits += 1 if hit else 0
		n += 1
		print("%5d  %-8s  %8d  %s" % [probe, expect_name, argmax, "Y" if hit else "."])
	print("hit_rate=%d/%d" % [hits, n])
	print("GODOT_ONNX_CSV_SMOKE_OK rows=%d" % n)
	get_tree().quit(0 if n > 0 else 1)
