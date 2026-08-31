extends Node
## MelFrontend + OnnxLoader: ci-fixture.wav → mel → logits; hit-rate vs demo_inputs.csv.

const VisemeUtils := preload("res://viseme_utils.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")
const MATCH_L2 := 0.01


func _run_hit_rate(loader: Object, contexts: Array, csv_path: String, nfeat: int) -> Dictionary:
	var file := FileAccess.open(csv_path, FileAccess.READ)
	if file == null:
		return {"ok": false, "hits": 0, "n": 0}
	file.get_csv_line()
	var hits := 0
	var n := 0
	print("%5s  %-8s  %-8s  hit  match" % ["probe", "expect", "argmax"])
	while not file.eof_reached():
		var row := file.get_csv_line()
		if row.size() < 3 + nfeat:
			continue
		var probe := int(row[0])
		var expect_id := int(row[1])
		var expect_name := str(row[2])
		var ref := PackedFloat32Array()
		ref.resize(nfeat)
		for i in nfeat:
			ref[i] = float(row[3 + i])
		var ctx_idx := VisemeUtils.find_nearest_context(ref, contexts, MATCH_L2)
		if ctx_idx < 0:
			print("%5d  %-8s  %8s  .  no_ctx" % [probe, expect_name, "-"])
			n += 1
			continue
		var ctx: PackedFloat32Array = contexts[ctx_idx]
		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			return {"ok": false, "hits": hits, "n": n}
		var w := VisemeUtils.softmax(logits)
		var argmax := VisemeUtils.argmax(w)
		var hit := argmax == expect_id
		hits += 1 if hit else 0
		n += 1
		print("%5d  %-8s  %8d  %s  ctx=%d" % [probe, expect_name, argmax, "Y" if hit else ".", ctx_idx])
	return {"ok": true, "hits": hits, "n": n}


func _ready() -> void:
	var paths := ClipProbeIo.resolve_ci_smoke_paths()
	var onnx_path := str(paths.get("onnx", ""))
	var pack := str(paths.get("dir", ClipProbeIo.models_abs().path_join("ci-smoke")))
	var wav_path := ClipProbeIo.models_abs().path_join("fixtures/ci-fixture.wav")
	var csv_path := pack.path_join("demo_inputs.csv")
	if not FileAccess.file_exists(csv_path):
		csv_path = ClipProbeIo.models_abs().path_join("ci-smoke/demo_inputs.csv")

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		push_error("MelFrontend missing — build gdextension .so")
		get_tree().quit(1)
		return
	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null or not loader.load_model(onnx_path):
		push_error("OnnxLoader load failed: %s" % onnx_path)
		get_tree().quit(1)
		return
	if not VisemeUtils.configure_mel_from_onnx(mel, loader):
		push_error("MelFrontend configure failed: %s" % onnx_path)
		get_tree().quit(1)
		return

	var nfeat: int = mel.get_input_features()
	if nfeat != loader.get_input_size():
		push_error("feature dim mismatch mel=%d onnx=%d" % [nfeat, loader.get_input_size()])
		get_tree().quit(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		get_tree().quit(1)
		return

	var contexts: Array = mel.build_utterance_contexts(pcm)
	if contexts.is_empty():
		push_error("no mel contexts from ci-fixture.wav")
		get_tree().quit(1)
		return

	var hr := _run_hit_rate(loader, contexts, csv_path, nfeat)
	if not hr.get("ok", false) or int(hr.get("n", 0)) <= 0:
		get_tree().quit(1)
		return

	print("mel_contexts=%d hit_rate=%d/%d" % [contexts.size(), hr.hits, hr.n])
	# Soft floor for ci-fixture smoke model; richer test-clean should beat this.
	if float(hr.hits) / float(hr.n) < 0.5:
		push_error("hit_rate below 50% — check mel/ONNX parity or retrain")
		get_tree().quit(1)
		return
	print("GODOT_MEL_ONNX_SMOKE_OK")
	get_tree().quit(0)
