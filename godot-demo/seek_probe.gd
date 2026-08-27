extends Node
## Seek-point probe: wav + alignment expects vs MelFrontend + ONNX (editor or headless).
## Open this scene in Godot and run — prints side-by-side table to the Output panel.

const VisemeUtils := preload("res://viseme_utils.gd")

var _quit_on_done := true


func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()


func _is_headless() -> bool:
	return DisplayServer.get_name() == "headless" or OS.has_feature("dedicated_server")


func _mel_l2(a: PackedFloat32Array, b: Array) -> float:
	if a.size() != b.size():
		return 1.0e9
	var d := 0.0
	for i in a.size():
		var diff := a[i] - float(b[i])
		d += diff * diff
	return sqrt(d)


func _ready() -> void:
	_quit_on_done = _is_headless()
	var root := _repo_root()
	var probe_path := root.path_join("export/ci-smoke/seek_probe.json")
	if not FileAccess.file_exists(probe_path):
		push_error("missing %s — run: python3 scripts/export_seek_probe.py" % probe_path)
		_finish(1)
		return

	var f := FileAccess.open(probe_path, FileAccess.READ)
	var probe: Variant = JSON.parse_string(f.get_as_text())
	if typeof(probe) != TYPE_DICTIONARY:
		push_error("bad seek_probe.json")
		_finish(1)
		return

	var json_path := root.path_join(str(probe.get("model_json", "export/ci-smoke/model.json")))
	var onnx_path := root.path_join(str(probe.get("onnx", "export/ci-smoke/model.onnx")))
	var wav_path := root.path_join(str(probe["wav"]))
	var mel_l2_max := float(probe.get("mel_l2_max", 0.05))
	var seeks: Array = probe.get("seeks", [])

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		push_error("MelFrontend missing — build gdextension .so")
		_finish(1)
		return
	if not mel.configure_from_json(json_path):
		push_error("MelFrontend configure failed")
		_finish(1)
		return

	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null:
		push_error("OnnxLoader missing — build onnx_loader addon")
		_finish(1)
		return
	if not loader.load_model(onnx_path):
		push_error("OnnxLoader load_model failed")
		_finish(1)
		return

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		_finish(1)
		return

	var contexts: Array = mel.build_utterance_contexts(pcm)
	if contexts.is_empty():
		push_error("no mel contexts from wav")
		_finish(1)
		return

	var id_to_name := VisemeUtils.load_id_to_name(json_path)
	print("seek_probe stem=%s contexts=%d seeks=%d" % [probe.get("stem", "?"), contexts.size(), seeks.size()])
	print(
		"%7s  %6s  %-8s  %-8s  %8s  %8s  %s"
		% ["t_sec", "frame", "expect", "got", "mel_l2", "hit", "phone"]
	)

	var hits := 0
	var mel_ok := 0
	var n := 0
	for s in seeks:
		if typeof(s) != TYPE_DICTIONARY:
			continue
		var t_sec := float(s.get("t_sec", 0.0))
		var frame := int(s.get("frame", 0))
		var expect_id := int(s.get("expect_id", 0))
		var expect_name := str(s.get("expect_name", "?"))
		var phone := str(s.get("phone", ""))
		var mel_ref: Array = s.get("mel_context", [])
		var win_i := int(s.get("window", frame))
		if win_i < 0 or win_i >= contexts.size():
			# Fall back: clamp to available contexts
			win_i = clampi(win_i, 0, contexts.size() - 1)

		var ctx: PackedFloat32Array = contexts[win_i]
		var mel_l2 := _mel_l2(ctx, mel_ref)
		var mel_match := mel_l2 <= mel_l2_max
		if mel_match:
			mel_ok += 1

		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			push_error("predict failed at t=%.3f" % t_sec)
			_finish(1)
			return
		var w := VisemeUtils.softmax(logits)
		var got := VisemeUtils.argmax(w)
		var got_name := str(id_to_name[got]) if got < id_to_name.size() else str(got)
		var hit := got == expect_id
		if hit:
			hits += 1
		n += 1
		print(
			"%7.3f  %6d  %-8s  %-8s  %8.4f  %8s  %s"
			% [t_sec, frame, expect_name, got_name, mel_l2, "Y" if hit else ".", phone]
		)
		# Compact mel fingerprint so editor experiments stay comparable
		var head := ""
		var n_show := mini(4, ctx.size())
		for i in n_show:
			if i > 0:
				head += ","
			head += "%.4f" % ctx[i]
		print("         mel_head=[%s] … n=%d l2_vs_train=%.4f %s" % [
			head, ctx.size(), mel_l2, "OK" if mel_match else "DRIFT"
		])

	if n <= 0:
		push_error("no seeks in probe")
		_finish(1)
		return

	print("seek_probe hit_rate=%d/%d mel_match=%d/%d" % [hits, n, mel_ok, n])
	if mel_ok < n:
		push_error("mel drift vs training path — check MelFrontend vs mel_features_c")
		_finish(1)
		return

	print("GODOT_SEEK_PROBE_OK")
	_finish(0)


func _finish(code: int) -> void:
	if _quit_on_done:
		get_tree().quit(code)
	elif code != 0:
		push_error("seek_probe failed (exit would be %d)" % code)
