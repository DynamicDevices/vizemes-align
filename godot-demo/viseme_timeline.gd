extends Control
## Overlay plot: 15 ONNX viseme weights vs time + MFA/trained label boxes.
## Editor: wheel zoom, middle-drag pan, left-drag select, Space/P play selection
## via AudioStreamGenerator. Headless: load + print GODOT_VISEME_TIMELINE_OK.

const VisemeUtils := preload("res://viseme_utils.gd")
const SAMPLE_RATE := 16000

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
## Each entry: PackedFloat32Array of length n_visemes (softmax at that hop).
var _series: Array = []
## Optional second ONNX softmax series for A/B compare (same mel contexts).
var _series_b: Array = []
var _label_a := "A"
var _label_b := "B"
var _show_a := true
var _show_b := true
var _show_disagree := true
var _status := ""
var _help := "wheel=zoom  mid-drag=pan  left-drag=select  Space=play  A/B=toggle models  D=disagree  Esc=clear"

var _pcm := PackedFloat32Array()
var _view_t0 := 0.0
var _view_t1 := 1.0
var _sel_t0 := -1.0
var _sel_t1 := -1.0
var _plot := Rect2()

var _drag_mode := ""  # "", "pan", "select"
var _drag_anchor_t := 0.0
var _drag_anchor_x := 0.0
var _pan_view_t0 := 0.0
var _pan_view_t1 := 0.0

var _player: AudioStreamPlayer
var _gen: AudioStreamGenerator
var _playback: AudioStreamGeneratorPlayback
var _play_i := 0
var _play_end := 0
var _playing := false


func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()


func _is_headless() -> bool:
	return DisplayServer.get_name() == "headless" or OS.has_feature("dedicated_server")


func _ready() -> void:
	_quit_on_done = _is_headless()
	set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	set_anchors_preset(PRESET_FULL_RECT)
	grow_horizontal = Control.GROW_DIRECTION_BOTH
	grow_vertical = Control.GROW_DIRECTION_BOTH
	mouse_filter = Control.MOUSE_FILTER_STOP if not _quit_on_done else Control.MOUSE_FILTER_IGNORE
	focus_mode = Control.FOCUS_ALL
	resized.connect(_on_resized)
	var code := _load_and_run()
	if not _quit_on_done:
		_setup_audio()
		grab_focus()
	_refresh_plot_rect()
	queue_redraw()
	if _quit_on_done:
		await get_tree().process_frame
		get_tree().quit(code)


func _on_resized() -> void:
	_refresh_plot_rect()
	queue_redraw()


## Plot area from current control size (shared by input + draw so coords stay in sync).
func _refresh_plot_rect() -> void:
	var r := size
	if r.x <= 1.0 or r.y <= 1.0:
		r = get_viewport_rect().size
	var left := 72.0
	var top := 36.0
	var right := 160.0
	var bottom := 56.0
	_plot = Rect2(left, top, maxf(32.0, r.x - left - right), maxf(32.0, r.y - top - bottom))


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
	_label_a = str(probe.get("label_a", "A:hidden64"))
	_label_b = str(probe.get("label_b", "B"))

	var mel = ClassDB.instantiate("MelFrontend")
	if mel == null:
		_status = "MelFrontend missing"
		push_error(_status)
		return 1
	if not mel.configure_from_json(json_path):
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

	var t0 := float(_context_frames - 1) * _hop_s
	var t_end := t0 + float(maxi(0, _series.size() - 1)) * _hop_s
	_duration_s = maxf(_duration_s, t_end)
	_view_t0 = 0.0
	_view_t1 = _duration_s

	_status = "stem=%s win=%d dur=%.2fs A=%s B=%s" % [
		str(probe.get("stem", "?")),
		_series.size(),
		_duration_s,
		_label_a,
		_label_b if not _series_b.is_empty() else "off",
	]
	print("viseme_timeline %s" % _status)
	print("GODOT_VISEME_TIMELINE_OK")
	return 0


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
	if _quit_on_done:
		return
	_refresh_plot_rect()
	# Prefer local mouse pos so press/drag share one coordinate space (fixes
	# selection start drifting left of the click while release looked right).
	var mouse_x := get_local_mouse_position().x
	var mouse_pos := get_local_mouse_position()
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			_zoom_at(mouse_x, 0.85)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			_zoom_at(mouse_x, 1.15)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_MIDDLE:
			if mb.pressed and _plot.has_point(mouse_pos):
				_drag_mode = "pan"
				_drag_anchor_x = mouse_x
				_pan_view_t0 = _view_t0
				_pan_view_t1 = _view_t1
				accept_event()
			elif not mb.pressed and _drag_mode == "pan":
				_drag_mode = ""
				accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed and _plot.has_point(mouse_pos):
				_drag_mode = "select"
				_drag_anchor_x = mouse_x
				_drag_anchor_t = _x_to_t(_drag_anchor_x)
				_sel_t0 = _drag_anchor_t
				_sel_t1 = _drag_anchor_t
				queue_redraw()
				accept_event()
			elif not mb.pressed and _drag_mode == "select":
				_sel_t0 = _x_to_t(_drag_anchor_x)
				_sel_t1 = _x_to_t(mouse_x)
				_drag_mode = ""
				if absf(_sel_t1 - _sel_t0) < 0.02:
					_sel_t0 = -1.0
					_sel_t1 = -1.0
				else:
					if _sel_t0 > _sel_t1:
						var tmp := _sel_t0
						_sel_t0 = _sel_t1
						_sel_t1 = tmp
				queue_redraw()
				accept_event()
	elif event is InputEventMouseMotion:
		if _drag_mode == "pan":
			var dt := (_drag_anchor_x - mouse_x) / maxf(1.0, _plot.size.x) * (_pan_view_t1 - _pan_view_t0)
			_view_t0 = _pan_view_t0 + dt
			_view_t1 = _pan_view_t1 + dt
			_clamp_view()
			queue_redraw()
			accept_event()
		elif _drag_mode == "select":
			# Recompute both ends from pixel anchors each frame (survives resize mid-drag).
			_sel_t0 = _x_to_t(_drag_anchor_x)
			_sel_t1 = _x_to_t(mouse_x)
			queue_redraw()
			accept_event()
	elif event is InputEventKey and event.pressed and not event.echo:
		var key := event as InputEventKey
		if key.keycode == KEY_SPACE or key.keycode == KEY_P:
			_play_selection()
			accept_event()
		elif key.keycode == KEY_ESCAPE:
			_stop_playback()
			_sel_t0 = -1.0
			_sel_t1 = -1.0
			queue_redraw()
			accept_event()
		elif key.keycode == KEY_R:
			_view_t0 = 0.0
			_view_t1 = _duration_s
			queue_redraw()
			accept_event()
		elif key.keycode == KEY_A:
			_show_a = not _show_a
			queue_redraw()
			accept_event()
		elif key.keycode == KEY_B:
			_show_b = not _show_b
			queue_redraw()
			accept_event()
		elif key.keycode == KEY_D:
			_show_disagree = not _show_disagree
			queue_redraw()
			accept_event()


func _zoom_at(x: float, factor: float) -> void:
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
		return
	var a := _sel_t0
	var b := _sel_t1
	if a < 0.0 or b < 0.0 or absf(b - a) < 0.02:
		# No selection: play the visible window.
		a = _view_t0
		b = _view_t1
	if a > b:
		var tmp := a
		a = b
		b = tmp
	a = clampf(a, 0.0, _duration_s)
	b = clampf(b, 0.0, _duration_s)
	_play_i = clampi(int(a * float(SAMPLE_RATE)), 0, _pcm.size())
	_play_end = clampi(int(b * float(SAMPLE_RATE)), 0, _pcm.size())
	if _play_end <= _play_i:
		return
	_stop_playback()
	_player.play()
	_playback = _player.get_stream_playback() as AudioStreamGeneratorPlayback
	_playing = _playback != null
	if _playing:
		_status = "playing %.2f–%.2fs" % [a, b]
		queue_redraw()


func _stop_playback() -> void:
	_playing = false
	_playback = null
	if _player != null and _player.playing:
		_player.stop()


func _process(_delta: float) -> void:
	if not _playing or _playback == null:
		return
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
	var r := size
	if r.x <= 1.0 or r.y <= 1.0:
		r = get_viewport_rect().size

	draw_rect(Rect2(Vector2.ZERO, r), Color(0.08, 0.09, 0.11))
	draw_rect(_plot, Color(0.12, 0.13, 0.16))
	draw_string(
		ThemeDB.fallback_font,
		Vector2(12, 22),
		"Viseme timeline — ONNX weights + MFA boxes",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		16,
		Color(0.9, 0.9, 0.9)
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(12, r.y - 28),
		_status,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		13,
		Color(0.7, 0.75, 0.8)
	)
	if not _quit_on_done:
		draw_string(
			ThemeDB.fallback_font,
			Vector2(12, r.y - 12),
			_help,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			12,
			Color(0.55, 0.6, 0.65)
		)

	if _duration_s <= 0.0:
		return

	# Selection overlay
	if _sel_t0 >= 0.0 and _sel_t1 >= 0.0:
		var sa := mini(_sel_t0, _sel_t1)
		var sb := maxf(_sel_t0, _sel_t1)
		var sx0 := _t_to_x(sa)
		var sx1 := _t_to_x(sb)
		draw_rect(
			Rect2(sx0, _plot.position.y, maxf(1.0, sx1 - sx0), _plot.size.y),
			Color(0.95, 0.85, 0.2, 0.18)
		)

	# MFA / trained boxes (background band) — only those intersecting view.
	var box_band := Rect2(_plot.position.x, _plot.position.y, _plot.size.x, _plot.size.y * 0.22)
	draw_rect(box_band, Color(0.16, 0.17, 0.20))
	for b in _boxes:
		if typeof(b) != TYPE_DICTIONARY:
			continue
		var bt0 := float(b.get("start", 0.0))
		var bt1 := float(b.get("end", 0.0))
		if bt1 < _view_t0 or bt0 > _view_t1:
			continue
		var vid := int(b.get("expect_id", 0))
		var name := str(b.get("expect_name", "?"))
		var x0 := _t_to_x(bt0)
		var x1 := _t_to_x(bt1)
		var col := LINE_COLORS[clampi(vid, 0, LINE_COLORS.size() - 1)]
		var fill := Color(col.r, col.g, col.b, 0.35)
		draw_rect(Rect2(x0, box_band.position.y + 4.0, maxf(2.0, x1 - x0), box_band.size.y - 8.0), fill)
		if x1 - x0 > 28.0:
			draw_string(
				ThemeDB.fallback_font,
				Vector2(x0 + 3.0, box_band.position.y + box_band.size.y * 0.62),
				name,
				HORIZONTAL_ALIGNMENT_LEFT,
				-1,
				12,
				Color(1, 1, 1, 0.9)
			)

	# A/B disagreement ribbon (where argmax differs).
	if _show_disagree and not _series_b.is_empty():
		var t_series0_d := float(_context_frames - 1) * _hop_s
		var n_cmp := mini(_series.size(), _series_b.size())
		var seg_start := -1.0
		for i in n_cmp:
			var t := t_series0_d + float(i) * _hop_s
			var wa: PackedFloat32Array = _series[i]
			var wb: PackedFloat32Array = _series_b[i]
			var disagree := VisemeUtils.argmax(wa) != VisemeUtils.argmax(wb)
			if disagree and t >= _view_t0 and t <= _view_t1:
				if seg_start < 0.0:
					seg_start = t
			elif seg_start >= 0.0:
				var x0 := _t_to_x(seg_start)
				var x1 := _t_to_x(t)
				draw_rect(
					Rect2(x0, _plot.position.y + _plot.size.y * 0.22, maxf(1.0, x1 - x0), _plot.size.y * 0.06),
					Color(1.0, 0.25, 0.25, 0.55)
				)
				seg_start = -1.0
		if seg_start >= 0.0:
			var x0b := _t_to_x(seg_start)
			var x1b := _t_to_x(mini(_view_t1, t_series0_d + float(n_cmp - 1) * _hop_s))
			draw_rect(
				Rect2(x0b, _plot.position.y + _plot.size.y * 0.22, maxf(1.0, x1b - x0b), _plot.size.y * 0.06),
				Color(1.0, 0.25, 0.25, 0.55)
			)

	# Weight curves (clip to view).
	var curve_top := _plot.position.y + _plot.size.y * 0.28
	var curve_h := _plot.size.y * 0.68
	var n_v := 0
	if not _series.is_empty():
		var first: PackedFloat32Array = _series[0]
		n_v = first.size()
	n_v = mini(n_v, LINE_COLORS.size())
	n_v = mini(n_v, _names.size() if not _names.is_empty() else n_v)

	var t_series0 := float(_context_frames - 1) * _hop_s
	if _show_a:
		_draw_series_curves(_series, t_series0, n_v, curve_top, curve_h, 2.0, 1.0)
	if _show_b and not _series_b.is_empty():
		_draw_series_curves(_series_b, t_series0, n_v, curve_top, curve_h, 1.2, 0.55)

	draw_line(
		Vector2(_plot.position.x, curve_top + curve_h),
		Vector2(_plot.position.x + _plot.size.x, curve_top + curve_h),
		Color(0.5, 0.5, 0.55),
		1.0
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(_plot.position.x, _plot.position.y + _plot.size.y + 22.0),
		"%.2fs" % _view_t0,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		12,
		Color(0.75, 0.75, 0.8)
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(_plot.position.x + _plot.size.x - 48.0, _plot.position.y + _plot.size.y + 22.0),
		"%.2fs" % _view_t1,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		12,
		Color(0.75, 0.75, 0.8)
	)
	if _sel_t0 >= 0.0 and _sel_t1 >= 0.0 and absf(_sel_t1 - _sel_t0) >= 0.02:
		draw_string(
			ThemeDB.fallback_font,
			Vector2(_plot.position.x + _plot.size.x * 0.35, _plot.position.y + _plot.size.y + 22.0),
			"sel %.2f–%.2fs" % [mini(_sel_t0, _sel_t1), maxf(_sel_t0, _sel_t1)],
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			12,
			Color(0.95, 0.85, 0.35)
		)

	var legend_x := _plot.position.x + _plot.size.x + 12.0
	var legend_y := _plot.position.y
	for vi in n_v:
		var lname := str(_names[vi]) if vi < _names.size() else str(vi)
		var col2: Color = LINE_COLORS[vi]
		draw_rect(Rect2(legend_x, legend_y + float(vi) * 18.0, 12, 12), col2)
		draw_string(
			ThemeDB.fallback_font,
			Vector2(legend_x + 18.0, legend_y + float(vi) * 18.0 + 11.0),
			lname,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			12,
			Color(0.85, 0.85, 0.9)
		)
	# A/B legend footer
	var foot_y := legend_y + float(n_v) * 18.0 + 10.0
	draw_string(
		ThemeDB.fallback_font,
		Vector2(legend_x, foot_y),
		("%s%s" % ["● " if _show_a else "○ ", _label_a]),
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		11,
		Color(0.9, 0.9, 0.95)
	)
	if not _series_b.is_empty():
		draw_string(
			ThemeDB.fallback_font,
			Vector2(legend_x, foot_y + 16.0),
			("%s%s" % ["● " if _show_b else "○ ", _label_b]),
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			11,
			Color(0.75, 0.8, 0.9)
		)
		draw_string(
			ThemeDB.fallback_font,
			Vector2(legend_x, foot_y + 32.0),
			("%sdisagree" % ["● " if _show_disagree else "○ "]),
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			11,
			Color(1.0, 0.45, 0.45)
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
