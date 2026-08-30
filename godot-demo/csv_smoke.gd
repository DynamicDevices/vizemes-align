extends Node
## OnnxLoader CSV smoke against vizemes export/ci-smoke (shared loader addon).

func _softmax(logits: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(logits.size())
	if logits.is_empty():
		return out
	var m := logits[0]
	for i in range(1, logits.size()):
		if logits[i] > m:
			m = logits[i]
	var sum := 0.0
	for i in logits.size():
		out[i] = exp(logits[i] - m)
		sum += out[i]
	if sum <= 0.0:
		sum = 1.0
	for i in logits.size():
		out[i] /= sum
	return out


func _ready() -> void:
	var paths := ClipProbeIo.resolve_model_paths(false)
	var onnx_path := str(paths.get("onnx", ""))
	var pack := str(paths.get("dir", ClipProbeIo.models_abs().path_join("ci-smoke")))
	var csv_path := pack.path_join("demo_inputs.csv")
	if not FileAccess.file_exists(csv_path):
		csv_path = ClipProbeIo.models_abs().path_join("ci-smoke/demo_inputs.csv")

	var m = ClassDB.instantiate("OnnxLoader")
	if m == null:
		push_error("OnnxLoader missing — build addon .so and check addons/onnx_loader symlink")
		get_tree().quit(1)
		return
	if not m.load_model(onnx_path):
		push_error("load_model failed: %s" % onnx_path)
		get_tree().quit(1)
		return
	print("ONNX_LOADER_DIAG ", m.get_diagnostics())

	var nfeat: int = m.get_input_size()
	var file := FileAccess.open(csv_path, FileAccess.READ)
	if file == null:
		push_error("open csv failed: %s" % csv_path)
		get_tree().quit(1)
		return
	var _header := file.get_csv_line()
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
		var logits: PackedFloat32Array = m.predict(ctx)
		if logits.is_empty():
			push_error("predict failed probe=%d" % probe)
			get_tree().quit(1)
			return
		var w := _softmax(logits)
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
