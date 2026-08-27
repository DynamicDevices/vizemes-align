extends Control
## Overlay plot: 15 ONNX viseme weights vs time + MFA/trained label boxes.
## Open viseme_timeline.tscn in Godot 4.6 and run (F6).

const VisemeUtils := preload("res://viseme_utils.gd")

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
var _status := ""


func _repo_root() -> String:
	return ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()


func _is_headless() -> bool:
	return DisplayServer.get_name() == "headless" or OS.has_feature("dedicated_server")


func _ready() -> void:
	_quit_on_done = _is_headless()
	set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	mouse_filter = MOUSE_FILTER_IGNORE
	var code := _load_and_run()
	queue_redraw()
	if _quit_on_done:
		# Give one frame so headless can still dump status prints.
		await get_tree().process_frame
		get_tree().quit(code)


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
	var wav_path := root.path_join(str(probe["wav"]))
	_hop_s = float(probe.get("hop_s", 0.01))
	_duration_s = float(probe.get("duration_s", 1.0))
	_context_frames = int(probe.get("context_frames", 20))
	_names = probe.get("viseme_names", [])
	_boxes = probe.get("boxes", [])

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

	var pcm := VisemeUtils.load_wav_pcm(wav_path)
	if pcm.is_empty():
		_status = "empty wav"
		return 1

	var contexts: Array = mel.build_utterance_contexts(pcm)
	if contexts.is_empty():
		_status = "no mel contexts"
		push_error(_status)
		return 1

	_series.clear()
	for i in contexts.size():
		var ctx: PackedFloat32Array = contexts[i]
		var logits: PackedFloat32Array = loader.predict(ctx)
		if logits.is_empty():
			_status = "predict failed at window %d" % i
			push_error(_status)
			return 1
		_series.append(VisemeUtils.softmax(logits))

	# Causal windows: first context ends at (context_frames-1)*hop.
	var t0 := float(_context_frames - 1) * _hop_s
	var t_end := t0 + float(maxi(0, _series.size() - 1)) * _hop_s
	_duration_s = maxf(_duration_s, t_end)

	_status = "stem=%s windows=%d duration=%.2fs boxes=%d" % [
		str(probe.get("stem", "?")), _series.size(), _duration_s, _boxes.size()
	]
	print("viseme_timeline %s" % _status)
	print("GODOT_VISEME_TIMELINE_OK")
	return 0


func _draw() -> void:
	var r := get_rect().size
	var left := 72.0
	var top := 36.0
	var right := 160.0
	var bottom := 48.0
	var plot := Rect2(left, top, maxf(32.0, r.x - left - right), maxf(32.0, r.y - top - bottom))

	draw_rect(Rect2(Vector2.ZERO, r), Color(0.08, 0.09, 0.11))
	draw_rect(plot, Color(0.12, 0.13, 0.16))
	draw_string(ThemeDB.fallback_font, Vector2(12, 22), "Viseme timeline — ONNX weights + MFA boxes", HORIZONTAL_ALIGNMENT_LEFT, -1, 16, Color(0.9, 0.9, 0.9))
	draw_string(ThemeDB.fallback_font, Vector2(12, r.y - 16), _status, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.7, 0.75, 0.8))

	if _duration_s <= 0.0:
		return

	# MFA / trained boxes (background band).
	var box_band := Rect2(plot.position.x, plot.position.y, plot.size.x, plot.size.y * 0.22)
	draw_rect(box_band, Color(0.16, 0.17, 0.20))
	for b in _boxes:
		if typeof(b) != TYPE_DICTIONARY:
			continue
		var t0 := float(b.get("start", 0.0))
		var t1 := float(b.get("end", 0.0))
		var vid := int(b.get("expect_id", 0))
		var name := str(b.get("expect_name", "?"))
		var x0 := plot.position.x + plot.size.x * clampf(t0 / _duration_s, 0.0, 1.0)
		var x1 := plot.position.x + plot.size.x * clampf(t1 / _duration_s, 0.0, 1.0)
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

	# Weight curves.
	var curve_top := plot.position.y + plot.size.y * 0.28
	var curve_h := plot.size.y * 0.68
	var n_v := 0
	if not _series.is_empty():
		var first: PackedFloat32Array = _series[0]
		n_v = first.size()
	n_v = mini(n_v, LINE_COLORS.size())
	n_v = mini(n_v, _names.size() if not _names.is_empty() else n_v)

	var t_start := float(_context_frames - 1) * _hop_s

	for vi in n_v:
		var col: Color = LINE_COLORS[vi]
		var pts := PackedVector2Array()
		pts.resize(_series.size())
		for i in _series.size():
			var w: PackedFloat32Array = _series[i]
			var t := t_start + float(i) * _hop_s
			var x := plot.position.x + plot.size.x * clampf(t / _duration_s, 0.0, 1.0)
			var y := curve_top + curve_h * (1.0 - clampf(w[vi], 0.0, 1.0))
			pts[i] = Vector2(x, y)
		if pts.size() >= 2:
			draw_polyline(pts, col, 2.0, true)

	# Axes / legend.
	draw_line(
		Vector2(plot.position.x, curve_top + curve_h),
		Vector2(plot.position.x + plot.size.x, curve_top + curve_h),
		Color(0.5, 0.5, 0.55),
		1.0
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(plot.position.x, plot.position.y + plot.size.y + 22.0),
		"0s",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		12,
		Color(0.75, 0.75, 0.8)
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(plot.position.x + plot.size.x - 40.0, plot.position.y + plot.size.y + 22.0),
		"%.1fs" % _duration_s,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		12,
		Color(0.75, 0.75, 0.8)
	)

	var legend_x := plot.position.x + plot.size.x + 12.0
	var legend_y := plot.position.y
	for vi in n_v:
		var name := str(_names[vi]) if vi < _names.size() else str(vi)
		var col2: Color = LINE_COLORS[vi]
		draw_rect(Rect2(legend_x, legend_y + float(vi) * 18.0, 12, 12), col2)
		draw_string(
			ThemeDB.fallback_font,
			Vector2(legend_x + 18.0, legend_y + float(vi) * 18.0 + 11.0),
			name,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			12,
			Color(0.85, 0.85, 0.9)
		)
