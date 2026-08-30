extends Control
## Quality companion: expect (MFA) vs got (ONNX) table + full MEL dumps.
## Primary day-to-day plot is viseme_timeline.tscn — use this for MEL parity.
## Editor: type stem + Load (no auto-quit). Headless: prints + GODOT_SEEK_PROBE_OK.

const VisemeUtils := preload("res://viseme_utils.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")

var _quit_on_done := true
var _stem_edit: LineEdit
var _status: Label
var _table: RichTextLabel
var _load_btn: Button
var _dump_btn: Button
var _last_probe: Dictionary = {}
var _last_rows: Array = []


func _ready() -> void:
	_quit_on_done = ClipProbeIo.is_headless()
	_build_ui()
	var code := _run_probe()
	if _quit_on_done:
		await get_tree().process_frame
		get_tree().quit(code)


func _build_ui() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	grow_horizontal = Control.GROW_DIRECTION_BOTH
	grow_vertical = Control.GROW_DIRECTION_BOTH

	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 12)
	margin.add_theme_constant_override("margin_right", 12)
	margin.add_theme_constant_override("margin_top", 10)
	margin.add_theme_constant_override("margin_bottom", 10)
	add_child(margin)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	margin.add_child(vbox)

	var title := Label.new()
	title.text = "Seek probe — expect (MFA) vs got (ONNX) + MEL dumps"
	title.add_theme_font_size_override("font_size", 18)
	vbox.add_child(title)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	vbox.add_child(row)

	var stem_lbl := Label.new()
	stem_lbl.text = "stem"
	row.add_child(stem_lbl)

	_stem_edit = LineEdit.new()
	_stem_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_stem_edit.placeholder_text = ClipProbeIo.DEFAULT_STEM
	_stem_edit.text = ClipProbeIo.DEFAULT_STEM
	_stem_edit.text_submitted.connect(func(_t): _on_load_pressed())
	row.add_child(_stem_edit)

	_load_btn = Button.new()
	_load_btn.text = "Load"
	_load_btn.pressed.connect(_on_load_pressed)
	row.add_child(_load_btn)

	_dump_btn = Button.new()
	_dump_btn.text = "Dump MEL"
	_dump_btn.tooltip_text = "Write full expect+got MEL tensors under export/debug/"
	_dump_btn.pressed.connect(_on_dump_pressed)
	row.add_child(_dump_btn)

	var mic_btn := Button.new()
	mic_btn.text = "Mic…"
	mic_btn.tooltip_text = "Open live mic lipsync scene"
	mic_btn.pressed.connect(func(): get_tree().change_scene_to_file("res://mic_lipsync.tscn"))
	row.add_child(mic_btn)

	var timeline_btn := Button.new()
	timeline_btn.text = "Timeline…"
	timeline_btn.tooltip_text = "Primary quality plot (viseme_timeline)"
	timeline_btn.pressed.connect(func(): get_tree().change_scene_to_file("res://viseme_timeline.tscn"))
	row.add_child(timeline_btn)

	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(_status)

	_table = RichTextLabel.new()
	_table.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_table.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_table.fit_content = false
	_table.scroll_active = true
	_table.bbcode_enabled = true
	vbox.add_child(_table)

	if _quit_on_done:
		# Headless: keep node tree minimal; still run probe.
		_load_btn.disabled = true
		_dump_btn.disabled = true


func _set_status(msg: String) -> void:
	_status.text = msg
	print("seek_probe %s" % msg)


func _on_load_pressed() -> void:
	var stem := _stem_edit.text.strip_edges()
	if stem.is_empty():
		stem = ClipProbeIo.DEFAULT_STEM
		_stem_edit.text = stem
	_set_status("exporting seek_probe for %s…" % stem)
	_load_btn.disabled = true
	var result: Dictionary = ClipProbeIo.export_seek_probe(stem, 8, ClipProbeIo.DEFAULT_SUBSET)
	_load_btn.disabled = false
	if not result.get("ok", false):
		_set_status("export failed (code %s): %s" % [str(result.get("code")), str(result.get("output"))])
		push_error(_status.text)
		return
	print(result.get("output", ""))
	_run_probe()


func _on_dump_pressed() -> void:
	if _last_rows.is_empty():
		_set_status("nothing to dump — run probe first")
		return
	var stem := str(_last_probe.get("stem", "unknown"))
	var n := 0
	for row in _last_rows:
		ClipProbeIo.dump_seek_mel(
			stem,
			float(row.get("t_sec", 0.0)),
			row.get("mel_ref", []),
			row.get("got_mel", PackedFloat32Array()),
			{
				"expect_name": row.get("expect_name", ""),
				"got_name": row.get("got_name", ""),
				"phone": row.get("phone", ""),
				"mel_l2": row.get("mel_l2", 0.0),
				"hit": row.get("hit", false),
			}
		)
		n += 1
	_set_status("dumped %d MEL tensors under export/debug/" % n)


func _mel_l2(a: PackedFloat32Array, b: Array) -> float:
	if a.size() != b.size():
		return 1.0e9
	var d := 0.0
	for i in a.size():
		var diff := a[i] - float(b[i])
		d += diff * diff
	return sqrt(d)


func _run_probe() -> int:
	## Prefer res:// paths so FileAccess works without abs-path footguns.
	var probe_res := OS.get_environment("SEEK_PROBE_JSON").strip_edges()
	if probe_res.is_empty():
		for cand in [
			"res://addons/vizeme-onnxmodels/ci-smoke/seek_probe.json",
			"res://addons/vizeme-onnxmodels/ci-smoke/seek_probe_ci_fixture.json",
		]:
			if FileAccess.file_exists(cand):
				probe_res = cand
				break
	elif not probe_res.begins_with("res://") and not probe_res.begins_with("user://"):
		## Absolute override must still sit under the project.
		var abs_p := probe_res if probe_res.is_absolute_path() else ClipProbeIo.project_abs().path_join(probe_res)
		if abs_p.begins_with(ClipProbeIo.project_abs()) and FileAccess.file_exists(abs_p):
			probe_res = abs_p
		else:
			probe_res = ""
	if probe_res.is_empty() or not FileAccess.file_exists(probe_res):
		_set_status("missing seek_probe.json under res://addons/vizeme-onnxmodels — Load a stem or sync")
		push_error(_status.text)
		return 1

	var f := FileAccess.open(probe_res, FileAccess.READ)
	var probe: Variant = JSON.parse_string(f.get_as_text())
	if typeof(probe) != TYPE_DICTIONARY:
		_set_status("bad seek_probe.json")
		push_error(_status.text)
		return 1

	_last_probe = probe
	_stem_edit.text = str(probe.get("stem", ClipProbeIo.DEFAULT_STEM))

	var model_paths := ClipProbeIo.resolve_ci_smoke_paths()
	var json_path := str(model_paths.get("json", ""))
	var onnx_path := str(model_paths.get("onnx", ""))
	var wav_path := ClipProbeIo.resolve_wav_for_probe(probe)
	var mel_l2_max := float(probe.get("mel_l2_max", 0.05))
	var seeks: Array = probe.get("seeks", [])

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		_set_status("MelFrontend missing — build gdextension .so")
		push_error(_status.text)
		return 1
	if not VisemeUtils.configure_mel_from_json(mel, json_path):
		_set_status("MelFrontend configure failed")
		push_error(_status.text)
		return 1

	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null:
		_set_status("OnnxLoader missing — build onnx_loader addon")
		push_error(_status.text)
		return 1
	if not loader.load_model(onnx_path):
		_set_status("OnnxLoader load_model failed")
		push_error(_status.text)
		return 1

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		_set_status("empty wav: %s" % wav_path)
		return 1

	var contexts: Array = mel.build_utterance_contexts(pcm)
	if contexts.is_empty():
		_set_status("no mel contexts from wav")
		push_error(_status.text)
		return 1

	var id_to_name := VisemeUtils.load_id_to_name(json_path)
	_last_rows.clear()

	print("seek_probe stem=%s contexts=%d seeks=%d" % [probe.get("stem", "?"), contexts.size(), seeks.size()])
	print(
		"%7s  %6s  %-8s  %-8s  %8s  %8s  %s"
		% ["t_sec", "frame", "expect", "got", "mel_l2", "hit", "phone"]
	)

	var hits := 0
	var mel_ok := 0
	var n := 0
	var bb := "[table=7][cell][b]t_sec[/b][/cell][cell][b]frame[/b][/cell][cell][b]expect[/b][/cell][cell][b]got[/b][/cell][cell][b]mel_l2[/b][/cell][cell][b]hit[/b][/cell][cell][b]phone[/b][/cell]"

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
			win_i = clampi(win_i, 0, contexts.size() - 1)

		var ctx: PackedFloat32Array = contexts[win_i]
		var mel_l2 := _mel_l2(ctx, mel_ref)
		var mel_match := mel_l2 <= mel_l2_max
		if mel_match:
			mel_ok += 1

		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			_set_status("predict failed at t=%.3f" % t_sec)
			push_error(_status.text)
			return 1
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
		print("         %s l2_vs_train=%.4f %s" % [
			ClipProbeIo.mel_fingerprint(ctx), mel_l2, "OK" if mel_match else "DRIFT"
		])

		# Always dump full tensors for quality/MEL iteration (Julian mid 680).
		ClipProbeIo.dump_seek_mel(
			str(probe.get("stem", "?")),
			t_sec,
			mel_ref,
			ctx,
			{"expect_name": expect_name, "got_name": got_name, "phone": phone, "mel_l2": mel_l2, "hit": hit}
		)

		var hit_col := "green" if hit else "red"
		bb += "[cell]%.3f[/cell][cell]%d[/cell][cell]%s[/cell][cell]%s[/cell][cell]%.4f[/cell][cell][color=%s]%s[/color][/cell][cell]%s[/cell]" % [
			t_sec, frame, expect_name, got_name, mel_l2, hit_col, "Y" if hit else ".", phone
		]

		_last_rows.append({
			"t_sec": t_sec,
			"expect_name": expect_name,
			"got_name": got_name,
			"phone": phone,
			"mel_l2": mel_l2,
			"hit": hit,
			"mel_ref": mel_ref,
			"got_mel": ctx,
		})

	bb += "[/table]"
	_table.clear()
	_table.append_text(bb)

	if n <= 0:
		_set_status("no seeks in probe")
		push_error(_status.text)
		return 1

	var summary := "stem=%s hit_rate=%d/%d mel_match=%d/%d (dumps in export/debug/)" % [
		probe.get("stem", "?"), hits, n, mel_ok, n
	]
	_set_status(summary)
	print("seek_probe %s" % summary)
	if mel_ok < n:
		var subset := str(probe.get("subset", ""))
		var msg := "mel drift vs training path — check MelFrontend vs mel_features_c"
		# CI fixture must stay strict. Real test-clean stems are for quality UI;
		# export/MEL path can differ slightly across hosts — warn, don't fail smoke.
		if subset == "ci-fixture" or OS.get_environment("SEEK_PROBE_STRICT_MEL") == "1":
			push_error(msg)
			return 1
		push_warning(msg)

	print("GODOT_SEEK_PROBE_OK")
	return 0
