extends Control
## Overlay plot: 15 ONNX viseme weights vs time + MFA/trained label boxes.
## Editor: wheel zoom, middle-drag pan, left-drag select, Space/P play selection
## via AudioStreamGenerator. Headless: load + print GODOT_VISEME_TIMELINE_OK.

const VisemeUtils := preload("res://viseme_utils.gd")
const ClipProbeIo := preload("res://clip_probe_io.gd")
const VisemeTarget := preload("res://viseme_target.gd")
const TimelineSelectOverlay := preload("res://timeline_select_overlay.gd")
const SAMPLE_RATE := 16000
## Single-caret Space play: ± this many seconds around the caret.
const CARET_PLAY_PAD_S := 0.10

## Distinct colours for the 15 model classes (silence … ou).
const LINE_COLORS: Array[Color] = [
	Color(0.55, 0.55, 0.55),
	Color(0.90, 0.25, 0.25),
	Color(0.95, 0.55, 0.15),
	Color(0.95, 0.85, 0.20),
	Color(0.45, 0.85, 0.25),
	Color(0.20, 0.75, 0.45),
	Color(0.15, 0.80, 0.80),
	Color(0.25, 0.55, 0.95),
	Color(0.45, 0.35, 0.95),
	Color(0.75, 0.30, 0.90),
	Color(0.95, 0.35, 0.70),
	Color(0.85, 0.45, 0.35),
	Color(0.60, 0.70, 0.30),
	Color(0.30, 0.60, 0.70),
	Color(0.70, 0.50, 0.30),
]

var _quit_on_done := true
var _duration_s := 1.0
var _hop_s := 0.01
var _context_frames := 20
var _names: Array = []
var _boxes: Array = []
var _words: Array = [] ## MFA word intervals {start,end,word}
## Each entry: PackedFloat32Array of length n_visemes (softmax at that hop).
var _series: Array = []
## Optional second ONNX softmax series for A/B compare (same mel contexts).
var _series_b: Array = []
var _label_a := "A"
var _label_b := "B"
var _show_a := true
var _show_b := true
var _show_disagree := true
var _show_hard := true
var _hard_bytes := PackedByteArray()
var _hard_frame_s := 0.02
var _status := ""
var _help := "wheel=zoom  mid-drag=pan  left-drag=select  Space=play  A/B=models  D=disagree  H=hard  Esc=clear"

var _pcm := PackedFloat32Array()
var _view_t0 := 0.0
var _view_t1 := 1.0
var _sel_t0 := -1.0
var _sel_t1 := -1.0
var _plot := Rect2()
var _overlay: Control

var _player: AudioStreamPlayer
var _gen: AudioStreamGenerator
var _playback: AudioStreamGeneratorPlayback
var _play_i := 0
var _play_end := 0
var _playing := false

var _stem_edit: LineEdit
var _load_btn: Button
var _rec_btn: Button
var _ui_top := 40.0
var _face: Node ## FacePanel / VisemeSystem
var _recording := false
var _rec_pcm := PackedFloat32Array()
var _rec_frames_left := 0
var _rec_seconds := 3.0


func _repo_root() -> String:
	return ClipProbeIo.repo_root()


func _is_headless() -> bool:
	return ClipProbeIo.is_headless()


func _ready() -> void:
	_quit_on_done = _is_headless()
	grow_horizontal = Control.GROW_DIRECTION_BOTH
	grow_vertical = Control.GROW_DIRECTION_BOTH
	# Stem bar + keys on this node; plot pointing lives on the overlay child.
	mouse_filter = Control.MOUSE_FILTER_STOP if not _quit_on_done else Control.MOUSE_FILTER_IGNORE
	focus_mode = Control.FOCUS_ALL
	resized.connect(_on_resized)
	if not _quit_on_done:
		_build_stem_bar()
		_resolve_face()
		_ensure_select_overlay()
	var code := _load_and_run()
	if not _quit_on_done:
		_setup_audio()
		grab_focus()
	_refresh_plot_rect()
	_layout_select_overlay()
	queue_redraw()
	if _quit_on_done:
		await get_tree().process_frame
		get_tree().quit(code)


func _ensure_select_overlay() -> void:
	if _overlay != null:
		return
	_overlay = TimelineSelectOverlay.new()
	_overlay.name = "SelectOverlay"
	_overlay.host = self
	_overlay.selection_changed.connect(_on_overlay_selection)
	_overlay.view_changed.connect(func(): queue_redraw())
	add_child(_overlay)


func _layout_select_overlay() -> void:
	if _overlay == null:
		return
	_refresh_plot_rect()
	_overlay.position = _plot.position
	_overlay.size = _plot.size
	_overlay.queue_redraw()


func _on_overlay_selection(t0: float, t1: float) -> void:
	_sel_t0 = t0
	_sel_t1 = t1
	queue_redraw()


func _resolve_face() -> void:
	var parent := get_parent()
	if parent != null:
		var panel := parent.get_node_or_null("FacePanel")
		if panel != null:
			_face = panel
			return
	_face = get_node_or_null("VisemeSystem")
	if _face == null:
		_face = get_node_or_null("VisemeSystemStub")


func _feed_face_at_playhead() -> void:
	_feed_face_at_time(float(_play_i) / float(SAMPLE_RATE))


func _feed_face_at_time(t: float) -> void:
	if _face == null or _series.is_empty():
		return
	var t0 := float(_context_frames - 1) * _hop_s
	var idx := int(round((t - t0) / _hop_s))
	idx = clampi(idx, 0, _series.size() - 1)
	var soft: PackedFloat32Array = _series[idx]
	var ovr := VisemeUtils.mlp_to_ovr(soft, _names)
	if _face.has_method("set_visemes"):
		_face.set_visemes(ovr)
	else:
		VisemeTarget.feed(_face, ovr)


func _build_stem_bar() -> void:
	var bar := HBoxContainer.new()
	bar.set_anchors_preset(Control.PRESET_TOP_WIDE)
	bar.offset_left = 8
	bar.offset_right = -8
	bar.offset_top = 4
	bar.offset_bottom = 36
	bar.add_theme_constant_override("separation", 8)
	bar.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(bar)

	var lbl := Label.new()
	lbl.text = "stem"
	bar.add_child(lbl)

	_stem_edit = LineEdit.new()
	_stem_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_stem_edit.placeholder_text = ClipProbeIo.DEFAULT_STEM
	_stem_edit.text = ClipProbeIo.DEFAULT_STEM
	_stem_edit.text_submitted.connect(func(_t): _on_stem_load())
	bar.add_child(_stem_edit)

	_load_btn = Button.new()
	_load_btn.text = "Load"
	_load_btn.pressed.connect(_on_stem_load)
	bar.add_child(_load_btn)

	var seek_btn := Button.new()
	seek_btn.text = "Seek table…"
	seek_btn.tooltip_text = "Side-by-side expect/got + MEL dumps"
	seek_btn.pressed.connect(func(): get_tree().change_scene_to_file("res://seek_probe.tscn"))
	bar.add_child(seek_btn)

	var mic_btn := Button.new()
	mic_btn.text = "Mic live…"
	mic_btn.tooltip_text = "Full-screen live mic scene"
	mic_btn.pressed.connect(func(): get_tree().change_scene_to_file("res://mic_lipsync.tscn"))
	bar.add_child(mic_btn)

	_rec_btn = Button.new()
	_rec_btn.text = "Record 3s"
	_rec_btn.tooltip_text = "Capture mic → replace graph PCM/curves with this clip (no MFA). Load restores the stem sample."
	_rec_btn.pressed.connect(_on_record_pressed)
	bar.add_child(_rec_btn)

	var play_btn := Button.new()
	play_btn.text = "Play"
	play_btn.tooltip_text = "Play caret ±100ms, or selection range, or visible window"
	play_btn.pressed.connect(_play_selection)
	bar.add_child(play_btn)

	_ui_top = 44.0
	_help = "Play/Space  wheel=zoom  mid=pan  left=caret/sel  Esc  Record replaces graph"


func _on_stem_load() -> void:
	var stem := _stem_edit.text.strip_edges()
	if stem.is_empty():
		stem = ClipProbeIo.DEFAULT_STEM
		_stem_edit.text = stem
	_status = "exporting timeline for %s…" % stem
	queue_redraw()
	_load_btn.disabled = true
	var result: Dictionary = ClipProbeIo.export_viseme_timeline(stem, ClipProbeIo.DEFAULT_SUBSET)
	_load_btn.disabled = false
	if not result.get("ok", false):
		_status = "export failed: %s" % str(result.get("output"))
		push_error(_status)
		queue_redraw()
		return
	print(result.get("output", ""))
	_stop_playback()
	_series.clear()
	_series_b.clear()
	_boxes.clear()
	_words.clear()
	var code := _load_and_run()
	if code != 0:
		push_error("reload failed after export")
	queue_redraw()
	grab_focus()


func _on_resized() -> void:
	_refresh_plot_rect()
	_layout_select_overlay()
	queue_redraw()


## Plot area from current control size (shared by input + draw so coords stay in sync).
func _refresh_plot_rect() -> void:
	var r := _canvas_size()
	var left := 72.0
	var top := _ui_top + 8.0
	var right := 24.0
	var bottom := 56.0
	_plot = Rect2(left, top, maxf(32.0, r.x - left - right), maxf(32.0, r.y - top - bottom))


func _on_record_pressed() -> void:
	if _recording:
		return
	if not ProjectSettings.get_setting("audio/driver/enable_input", false):
		ProjectSettings.set_setting("audio/driver/enable_input", true)
	var err := AudioServer.set_input_device_active(true)
	if err != OK:
		_status = "mic failed: %s" % error_string(err)
		queue_redraw()
		return
	_stop_playback()
	_recording = true
	_rec_pcm = PackedFloat32Array()
	_rec_frames_left = int(_rec_seconds * maxf(1.0, AudioServer.get_input_mix_rate()))
	_rec_btn.disabled = true
	_rec_btn.text = "Recording…"
	_status = "recording %.1fs…" % _rec_seconds
	queue_redraw()


func _finish_recording() -> void:
	_recording = false
	AudioServer.set_input_device_active(false)
	_rec_btn.disabled = false
	_rec_btn.text = "Record 3s"
	var in_rate := int(round(AudioServer.get_input_mix_rate()))
	var mono := VisemeUtils.resample_pcm(_rec_pcm, in_rate, SAMPLE_RATE)
	if mono.size() < SAMPLE_RATE / 2:
		_status = "recording too short"
		queue_redraw()
		return
	_stop_playback()
	_status = "inferring recording into graph…"
	queue_redraw()
	# Infer first; only keep new PCM if curves recompute (else playable clip + stale graph).
	var prev_pcm := _pcm
	_pcm = mono
	var code := _infer_from_pcm()
	if code != 0:
		_pcm = prev_pcm
		_status = "record infer failed (graph unchanged) — check MelFrontend/OnnxLoader + model.onnx"
	else:
		_view_t0 = 0.0
		_view_t1 = _duration_s
		_sel_t0 = -1.0
		_sel_t1 = -1.0
		if _overlay != null and _overlay.has_method("clear_selection"):
			_overlay.clear_selection()
		_play_i = 0
		_play_end = 0
		_feed_face_at_time(0.0)
		_status = "graph = your %.2fs recording (no MFA) — Space plays it; Load restores stem" % _duration_s
	queue_redraw()
	grab_focus()


func _infer_from_pcm() -> int:
	## Live/mic preview only: MelFrontend + ONNX → soft series + hard bytes.
	## Does NOT run MFA / export_viseme_timeline (train-set alignment stays on Load).
	if not ClassDB.class_exists("MelFrontend") or not ClassDB.class_exists("OnnxLoader"):
		push_error("need MelFrontend + OnnxLoader")
		return 1
	var root := _repo_root()
	var onnx_path := root.path_join("export/ci-smoke/model.onnx")
	var json_path := root.path_join("export/ci-smoke/model.json")
	if not FileAccess.file_exists(onnx_path) or not FileAccess.file_exists(json_path):
		return 1
	_names = VisemeUtils.load_id_to_name(json_path)
	var mel = ClassDB.instantiate("MelFrontend")
	var loader = ClassDB.instantiate("OnnxLoader")
	if mel == null or loader == null:
		return 1
	if not VisemeUtils.configure_mel_from_json(mel, json_path):
		push_error("MelFrontend configure failed for record infer")
		return 1
	if not loader.load_model(onnx_path):
		return 1
	var contexts: Array = mel.build_utterance_contexts(_pcm)
	if contexts.is_empty():
		push_error("record infer: no mel contexts")
		return 1
	_series = _predict_series(loader, contexts)
	if _series.is_empty():
		return 1
	_series_b.clear()
	# Live clips have no MFA boxes — clear any leftover train-set align overlay.
	_boxes.clear()
	_words.clear()
	_context_frames = 20
	_hop_s = 0.01
	_rebuild_hard_bytes()
	var t0 := float(_context_frames - 1) * _hop_s
	_duration_s = t0 + float(maxi(0, _series.size() - 1)) * _hop_s
	return 0


func _setup_audio() -> void:
	_gen = AudioStreamGenerator.new()
	_gen.mix_rate = float(SAMPLE_RATE)
	_gen.buffer_length = 0.1
	_player = AudioStreamPlayer.new()
	_player.stream = _gen
	add_child(_player)


func _load_and_run() -> int:
	var root := _repo_root()
	var path := root.path_join("export/ci-smoke/viseme_timeline.json")
	if not FileAccess.file_exists(path):
		_status = "missing viseme_timeline.json — run: python3 scripts/export_viseme_timeline.py"
		push_error(_status)
		return 1

	var f := FileAccess.open(path, FileAccess.READ)
	var probe: Variant = JSON.parse_string(f.get_as_text())
	if typeof(probe) != TYPE_DICTIONARY:
		_status = "bad viseme_timeline.json"
		push_error(_status)
		return 1

	var json_path := root.path_join(str(probe.get("model_json", "export/ci-smoke/model.json")))
	var onnx_path := root.path_join(str(probe.get("onnx", "export/ci-smoke/model.onnx")))
	var onnx_b_rel := str(probe.get("onnx_b", ""))
	var wav_path := root.path_join(str(probe["wav"]))
	_hop_s = float(probe.get("hop_s", 0.01))
	_duration_s = float(probe.get("duration_s", 1.0))
	_context_frames = int(probe.get("context_frames", 20))
	_names = probe.get("viseme_names", [])
	_boxes = probe.get("boxes", [])
	_words = probe.get("words", [])
	_label_a = str(probe.get("label_a", "A:hidden64"))
	_label_b = str(probe.get("label_b", "B"))
	if _stem_edit != null:
		_stem_edit.text = str(probe.get("stem", ClipProbeIo.DEFAULT_STEM))

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		_status = "MelFrontend missing"
		push_error(_status)
		return 1
	if not VisemeUtils.configure_mel_from_json(mel, json_path):
		_status = "MelFrontend configure failed"
		push_error(_status)
		return 1

	var loader = ClassDB.instantiate("OnnxLoader")
	if loader == null:
		_status = "OnnxLoader missing"
		push_error(_status)
		return 1
	if not loader.load_model(onnx_path):
		_status = "OnnxLoader load_model failed"
		push_error(_status)
		return 1

	_pcm = VisemeUtils.load_wav_pcm(wav_path)
	if _pcm.is_empty():
		_status = "empty wav"
		return 1

	var contexts: Array = mel.build_utterance_contexts(_pcm)
	if contexts.is_empty():
		_status = "no mel contexts"
		push_error(_status)
		return 1

	_series = _predict_series(loader, contexts)
	if _series.is_empty():
		return 1

	_series_b.clear()
	if not onnx_b_rel.is_empty():
		var onnx_b_path := root.path_join(onnx_b_rel)
		if FileAccess.file_exists(onnx_b_path):
			var loader_b = ClassDB.instantiate("OnnxLoader")
			if loader_b != null and loader_b.load_model(onnx_b_path):
				_series_b = _predict_series(loader_b, contexts)
				if _label_b == "B":
					_label_b = "B:%s" % onnx_b_rel.get_file()
			else:
				push_warning("onnx_b failed to load: %s" % onnx_b_path)
		else:
			push_warning("onnx_b missing: %s" % onnx_b_path)

	_rebuild_hard_bytes()

	var t0 := float(_context_frames - 1) * _hop_s
	var t_end := t0 + float(maxi(0, _series.size() - 1)) * _hop_s
	_duration_s = maxf(_duration_s, t_end)
	_view_t0 = 0.0
	_view_t1 = _duration_s

	_status = "stem=%s win=%d hard=%dB@%.0fms dur=%.2fs A=%s B=%s" % [
		str(probe.get("stem", "?")),
		_series.size(),
		_hard_bytes.size(),
		_hard_frame_s * 1000.0,
		_duration_s,
		_label_a,
		_label_b if not _series_b.is_empty() else "off",
	]
	print("viseme_timeline %s" % _status)
	print("GODOT_VISEME_TIMELINE_OK")
	return 0


func _rebuild_hard_bytes() -> void:
	var t0 := float(_context_frames - 1) * _hop_s
	_hard_bytes = VisemeUtils.series_to_hard_bytes(_series, _hop_s, t0, _hard_frame_s)


func _predict_series(loader, contexts: Array) -> Array:
	var out: Array = []
	for i in contexts.size():
		var ctx: PackedFloat32Array = contexts[i]
		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			_status = "predict failed at window %d" % i
			push_error(_status)
			return []
		out.append(VisemeUtils.softmax(logits))
	return out


func _view_span() -> float:
	return maxf(1e-4, _view_t1 - _view_t0)


func _x_to_t(x: float) -> float:
	if _plot.size.x <= 0.0:
		return _view_t0
	var u := clampf((x - _plot.position.x) / _plot.size.x, 0.0, 1.0)
	return _view_t0 + u * _view_span()


func _t_to_x(t: float) -> float:
	var u := clampf((t - _view_t0) / _view_span(), 0.0, 1.0)
	return _plot.position.x + u * _plot.size.x


func _clamp_view() -> void:
	var span := _view_span()
	span = clampf(span, 0.05, maxf(0.05, _duration_s))
	_view_t0 = clampf(_view_t0, 0.0, maxf(0.0, _duration_s - span))
	_view_t1 = _view_t0 + span
	if _view_t1 > _duration_s:
		_view_t1 = _duration_s
		_view_t0 = maxf(0.0, _view_t1 - span)


func _gui_input(event: InputEvent) -> void:
	## Prefer unhandled path so Space works even when overlay/LineEdit had focus.
	if _quit_on_done:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		_handle_key(event as InputEventKey)


func _unhandled_input(event: InputEvent) -> void:
	if _quit_on_done:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		var k := event as InputEventKey
		if k.keycode in [KEY_SPACE, KEY_P, KEY_ESCAPE, KEY_R, KEY_A, KEY_B, KEY_D, KEY_H]:
			_handle_key(k)
			get_viewport().set_input_as_handled()


func _handle_key(key: InputEventKey) -> void:
	match key.keycode:
		KEY_SPACE, KEY_P:
			_play_selection()
			accept_event()
		KEY_ESCAPE:
			_stop_playback()
			_sel_t0 = -1.0
			_sel_t1 = -1.0
			if _overlay != null and _overlay.has_method("clear_selection"):
				_overlay.clear_selection()
			queue_redraw()
			accept_event()
		KEY_R:
			_view_t0 = 0.0
			_view_t1 = _duration_s
			queue_redraw()
			if _overlay != null:
				_overlay.queue_redraw()
			accept_event()
		KEY_A:
			_show_a = not _show_a
			queue_redraw()
			accept_event()
		KEY_B:
			_show_b = not _show_b
			queue_redraw()
			accept_event()
		KEY_D:
			_show_disagree = not _show_disagree
			queue_redraw()
			accept_event()
		KEY_H:
			_show_hard = not _show_hard
			queue_redraw()
			accept_event()


func _zoom_at(x: float, factor: float) -> void:
	## Kept for compatibility; overlay owns zoom in the editor.
	var t_focus := _x_to_t(x)
	var span := _view_span() * factor
	span = clampf(span, 0.05, maxf(0.05, _duration_s))
	var left_frac := (t_focus - _view_t0) / _view_span()
	_view_t0 = t_focus - left_frac * span
	_view_t1 = _view_t0 + span
	_clamp_view()
	queue_redraw()


func _play_selection() -> void:
	if _pcm.is_empty():
		_status = "nothing to play (empty PCM)"
		queue_redraw()
		return
	if _player == null:
		_setup_audio()
	if _player == null:
		_status = "audio player missing"
		queue_redraw()
		return
	var a := _sel_t0
	var b := _sel_t1
	if a < 0.0 or b < 0.0:
		a = _view_t0
		b = _view_t1
	elif absf(b - a) < 0.02:
		var t := a
		a = t - CARET_PLAY_PAD_S
		b = t + CARET_PLAY_PAD_S
	if a > b:
		var tmp := a
		a = b
		b = tmp
	a = clampf(a, 0.0, _duration_s)
	b = clampf(b, 0.0, _duration_s)
	_play_i = clampi(int(floor(a * float(SAMPLE_RATE))), 0, maxi(0, _pcm.size() - 1))
	_play_end = clampi(int(ceil(b * float(SAMPLE_RATE))), 0, _pcm.size())
	if _play_end <= _play_i:
		_status = "play range empty"
		queue_redraw()
		return
	_stop_playback()
	_player.play()
	_playback = _player.get_stream_playback() as AudioStreamGeneratorPlayback
	_playing = _playback != null
	if _playing:
		_status = "playing %.3f–%.3fs (%d samples)" % [a, b, _play_end - _play_i]
		queue_redraw()
	else:
		_status = "play failed (no AudioStreamGeneratorPlayback)"
		queue_redraw()


func _stop_playback() -> void:
	_playing = false
	_playback = null
	if _player != null and _player.playing:
		_player.stop()


func _process(_delta: float) -> void:
	if _recording:
		var avail := AudioServer.get_input_frames_available()
		if avail > 0:
			var n := mini(avail, _rec_frames_left)
			var buf: PackedVector2Array = AudioServer.get_input_frames(n)
			var mono := VisemeUtils.stereo_to_mono(buf)
			for s in mono:
				_rec_pcm.append(s)
			_rec_frames_left -= mono.size()
		if _rec_frames_left <= 0:
			_finish_recording()
		return
	if not _playing or _playback == null:
		return
	_feed_face_at_playhead()
	var frames := _playback.get_frames_available()
	if frames <= 0:
		return
	var buf := PackedVector2Array()
	buf.resize(frames)
	var filled := 0
	for i in frames:
		if _play_i >= _play_end:
			break
		var s := _pcm[_play_i]
		buf[i] = Vector2(s, s)
		_play_i += 1
		filled += 1
	if filled <= 0:
		_playing = false
		_status = "play done"
		queue_redraw()
		return
	if filled < frames:
		buf.resize(filled)
	_playback.push_buffer(buf)
	if _play_i >= _play_end:
		_playing = false
		_status = "play done"
		queue_redraw()


func _draw() -> void:
	_refresh_plot_rect()
	_layout_select_overlay()
	var r := _canvas_size()
	_draw_chrome(r)
	if _duration_s <= 0.0:
		return
	# Selection caret/band is drawn by SelectOverlay (translucent child).
	_draw_mfa_boxes()
	_draw_disagree_ribbon()
	_draw_hard_ribbon()

	var curve_top := _plot.position.y + _plot.size.y * 0.28
	var curve_h := _plot.size.y * 0.68
	var n_v := _n_visemes()
	var t0 := float(_context_frames - 1) * _hop_s
	if _show_a:
		_draw_series_curves(_series, t0, n_v, curve_top, curve_h, 2.0, 1.0)
	if _show_b and not _series_b.is_empty():
		_draw_series_curves(_series_b, t0, n_v, curve_top, curve_h, 1.2, 0.55)

	_draw_time_axis(curve_top, curve_h)
	_draw_legend(n_v)


func _canvas_size() -> Vector2:
	# Never fall back to full viewport — that desyncs drawn curves from click x
	# when this control is only a SplitContainer pane.
	var r := size
	if r.x <= 1.0 or r.y <= 1.0:
		r = get_rect().size
	return r


func _n_visemes() -> int:
	if _series.is_empty():
		return 0
	var first: PackedFloat32Array = _series[0]
	var n := first.size()
	n = mini(n, LINE_COLORS.size())
	if not _names.is_empty():
		n = mini(n, _names.size())
	return n


func _draw_chrome(r: Vector2) -> void:
	draw_rect(Rect2(Vector2.ZERO, r), Color(0.08, 0.09, 0.11))
	draw_rect(_plot, Color(0.12, 0.13, 0.16))
	if _quit_on_done:
		draw_string(
			ThemeDB.fallback_font, Vector2(12, 22),
			"Viseme timeline — ONNX weights + MFA boxes (quality UI)",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 16, Color(0.9, 0.9, 0.9)
		)
	draw_string(
		ThemeDB.fallback_font, Vector2(12, r.y - 28), _status,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.7, 0.75, 0.8)
	)
	if not _quit_on_done:
		draw_string(
			ThemeDB.fallback_font, Vector2(12, r.y - 12), _help,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.55, 0.6, 0.65)
		)


func _draw_mfa_boxes() -> void:
	## Top: MFA English words; below: collapsed viseme expect labels.
	var band := Rect2(_plot.position.x, _plot.position.y, _plot.size.x, _plot.size.y * 0.22)
	draw_rect(band, Color(0.16, 0.17, 0.20))
	var word_h := band.size.y * 0.42
	var vis_y := band.position.y + word_h
	var vis_h := band.size.y - word_h
	for w in _words:
		if typeof(w) != TYPE_DICTIONARY:
			continue
		var wt0 := float(w.get("start", 0.0))
		var wt1 := float(w.get("end", 0.0))
		if wt1 < _view_t0 or wt0 > _view_t1:
			continue
		var text := str(w.get("word", "")).strip_edges()
		if text.is_empty():
			continue
		var low := text.to_lower()
		if low == "sil" or low == "sp" or low == "spn":
			continue
		var x0 := _t_to_x(wt0)
		var x1 := _t_to_x(wt1)
		var ww := maxf(2.0, x1 - x0)
		draw_rect(Rect2(x0, band.position.y + 2.0, ww, word_h - 4.0), Color(0.22, 0.28, 0.36, 0.55))
		if ww > 18.0:
			draw_string(
				ThemeDB.fallback_font,
				Vector2(x0 + 2.0, band.position.y + word_h * 0.72),
				text,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.92, 0.94, 0.98, 0.95)
			)
	for b in _boxes:
		if typeof(b) != TYPE_DICTIONARY:
			continue
		var bt0 := float(b.get("start", 0.0))
		var bt1 := float(b.get("end", 0.0))
		if bt1 < _view_t0 or bt0 > _view_t1:
			continue
		var vid := int(b.get("expect_id", 0))
		var x0 := _t_to_x(bt0)
		var x1 := _t_to_x(bt1)
		var col := LINE_COLORS[clampi(vid, 0, LINE_COLORS.size() - 1)]
		draw_rect(
			Rect2(x0, vis_y + 2.0, maxf(2.0, x1 - x0), vis_h - 4.0),
			Color(col.r, col.g, col.b, 0.35)
		)
		if x1 - x0 > 28.0:
			draw_string(
				ThemeDB.fallback_font,
				Vector2(x0 + 3.0, vis_y + vis_h * 0.70),
				str(b.get("expect_name", "?")),
				HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(1, 1, 1, 0.9)
			)


func _draw_disagree_ribbon() -> void:
	if not _show_disagree or _series_b.is_empty():
		return
	var t0 := float(_context_frames - 1) * _hop_s
	var n := mini(_series.size(), _series_b.size())
	var seg := -1.0
	var y := _plot.position.y + _plot.size.y * 0.22
	var h := _plot.size.y * 0.06
	var col := Color(1.0, 0.25, 0.25, 0.55)
	for i in n:
		var t := t0 + float(i) * _hop_s
		var wa: PackedFloat32Array = _series[i]
		var wb: PackedFloat32Array = _series_b[i]
		var bad := VisemeUtils.argmax(wa) != VisemeUtils.argmax(wb) and t >= _view_t0 and t <= _view_t1
		if bad:
			if seg < 0.0:
				seg = t
		elif seg >= 0.0:
			draw_rect(Rect2(_t_to_x(seg), y, maxf(1.0, _t_to_x(t) - _t_to_x(seg)), h), col)
			seg = -1.0
	if seg >= 0.0:
		var t_end := minf(_view_t1, t0 + float(n - 1) * _hop_s)
		draw_rect(Rect2(_t_to_x(seg), y, maxf(1.0, _t_to_x(t_end) - _t_to_x(seg)), h), col)


func _draw_hard_ribbon() -> void:
	## Stepped MPEG-4-style hard id track (1 byte / 20 ms) under MFA boxes.
	if not _show_hard or _hard_bytes.is_empty():
		return
	var t0 := float(_context_frames - 1) * _hop_s
	var y := _plot.position.y + _plot.size.y * 0.22
	var h := _plot.size.y * 0.06
	for i in _hard_bytes.size():
		var t := t0 + float(i) * _hard_frame_s
		var t1 := t + _hard_frame_s
		if t1 < _view_t0 or t > _view_t1:
			continue
		var b: int = _hard_bytes[i]
		var vid := VisemeUtils.hard_byte_id(b)
		var blend := VisemeUtils.hard_byte_blend01(b)
		var col := LINE_COLORS[clampi(vid, 0, LINE_COLORS.size() - 1)]
		draw_rect(
			Rect2(_t_to_x(t), y, maxf(1.0, _t_to_x(t1) - _t_to_x(t)), h),
			Color(col.r, col.g, col.b, 0.25 + 0.55 * blend)
		)


func _draw_time_axis(curve_top: float, curve_h: float) -> void:
	draw_line(
		Vector2(_plot.position.x, curve_top + curve_h),
		Vector2(_plot.position.x + _plot.size.x, curve_top + curve_h),
		Color(0.5, 0.5, 0.55), 1.0
	)
	var ty := _plot.position.y + _plot.size.y + 22.0
	draw_string(
		ThemeDB.fallback_font, Vector2(_plot.position.x, ty), "view %.3fs" % _view_t0,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.75, 0.75, 0.8)
	)
	draw_string(
		ThemeDB.fallback_font, Vector2(_plot.position.x + _plot.size.x - 72.0, ty), "%.3fs" % _view_t1,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.75, 0.75, 0.8)
	)
	if _sel_t0 >= 0.0 and _sel_t1 >= 0.0:
		var sel_label := "caret %.3fs (Space/Play = ±100ms)" % _sel_t0
		if absf(_sel_t1 - _sel_t0) >= 0.02:
			sel_label = "sel %.3f–%.3fs" % [minf(_sel_t0, _sel_t1), maxf(_sel_t0, _sel_t1)]
		draw_string(
			ThemeDB.fallback_font,
			Vector2(_plot.position.x + _plot.size.x * 0.22, ty),
			sel_label,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.95, 0.85, 0.35)
		)


func _draw_legend(n_v: int) -> void:
	var lx := _plot.position.x + _plot.size.x + 12.0
	var ly := _plot.position.y
	for vi in n_v:
		draw_rect(Rect2(lx, ly + float(vi) * 18.0, 12, 12), LINE_COLORS[vi])
		draw_string(
			ThemeDB.fallback_font,
			Vector2(lx + 18.0, ly + float(vi) * 18.0 + 11.0),
			str(_names[vi]) if vi < _names.size() else str(vi),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.85, 0.85, 0.9)
		)
	var foot := ly + float(n_v) * 18.0 + 10.0
	draw_string(
		ThemeDB.fallback_font, Vector2(lx, foot),
		("%s%s" % ["● " if _show_a else "○ ", _label_a]),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.9, 0.9, 0.95)
	)
	if _series_b.is_empty():
		return
	draw_string(
		ThemeDB.fallback_font, Vector2(lx, foot + 16.0),
		("%s%s" % ["● " if _show_b else "○ ", _label_b]),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.75, 0.8, 0.9)
	)
	draw_string(
		ThemeDB.fallback_font, Vector2(lx, foot + 32.0),
		("%sdisagree" % ["● " if _show_disagree else "○ "]),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1.0, 0.45, 0.45)
	)


func _draw_series_curves(
	series: Array,
	t_series0: float,
	n_v: int,
	curve_top: float,
	curve_h: float,
	width: float,
	alpha: float
) -> void:
	for vi in n_v:
		var base: Color = LINE_COLORS[vi]
		var col := Color(base.r, base.g, base.b, alpha)
		var pts := PackedVector2Array()
		for i in series.size():
			var t := t_series0 + float(i) * _hop_s
			if t < _view_t0 - _hop_s or t > _view_t1 + _hop_s:
				continue
			var w: PackedFloat32Array = series[i]
			var x := _t_to_x(t)
			var y := curve_top + curve_h * (1.0 - clampf(w[vi], 0.0, 1.0))
			pts.append(Vector2(x, y))
		if pts.size() >= 2:
			draw_polyline(pts, col, width, true)
