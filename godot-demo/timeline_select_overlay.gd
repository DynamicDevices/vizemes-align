extends Control
## Translucent hit-layer over the plot.
## During drag we keep *pixels*; convert to seconds only when emitting
## (avoids x→time with size.x≈0 mapping the low edge to view_t0 / a "round second").

signal selection_changed(t0: float, t1: float)
signal view_changed

const SELECT_DRAG_PX := 8.0
const MIN_WIDTH_PX := 32.0
const CARET_EPS_S := 0.002

enum Drag { NONE, PAN, SELECT }

var host: Node ## VisemeTimeline
var _drag := Drag.NONE
var _anchor_x := 0.0
var _cur_x := 0.0
var _select_moved := false
var _pan_t0 := 0.0
var _pan_t1 := 0.0
## Committed selection in seconds (−1 = none).
var sel_t0 := -1.0
var sel_t1 := -1.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	focus_mode = Control.FOCUS_ALL
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)


func clear_selection() -> void:
	sel_t0 = -1.0
	sel_t1 = -1.0
	_cur_x = -1.0
	_anchor_x = -1.0
	queue_redraw()
	selection_changed.emit(-1.0, -1.0)


func _ready_size() -> bool:
	return size.x >= MIN_WIDTH_PX and size.y >= 8.0


func _view_t0() -> float:
	return float(host.get("_view_t0"))


func _view_t1() -> float:
	return float(host.get("_view_t1"))


func _view_span() -> float:
	return maxf(1e-6, _view_t1() - _view_t0())


func _x_to_t(x: float) -> float:
	## Requires a real width — callers must check _ready_size().
	var u := clampf(x / size.x, 0.0, 1.0)
	return _view_t0() + u * _view_span()


func _t_to_x(t: float) -> float:
	if not _ready_size():
		return 0.0
	var u := clampf((t - _view_t0()) / _view_span(), 0.0, 1.0)
	return u * size.x


func _px_pair_to_times(x0: float, x1: float) -> Vector2:
	var lo := mini(x0, x1)
	var hi := maxf(x0, x1)
	if hi - lo < SELECT_DRAG_PX:
		var t := _x_to_t(lo)
		return Vector2(t, t)
	return Vector2(_x_to_t(lo), _x_to_t(hi))


func _gui_input(event: InputEvent) -> void:
	if host == null or bool(host.get("_quit_on_done")):
		return
	if event is InputEventKey and event.pressed and not event.echo:
		host.call("_handle_key", event)
		accept_event()
		return
	if not _ready_size():
		# Force host to lay us out; ignore pointing until then.
		if host.has_method("_layout_select_overlay"):
			host.call("_layout_select_overlay")
		if not _ready_size():
			return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		var x := clampf(mb.position.x, 0.0, size.x)
		if mb.pressed:
			grab_focus()
			host.grab_focus()
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			_zoom_at(x, 0.85)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			_zoom_at(x, 1.15)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_MIDDLE:
			if mb.pressed:
				_drag = Drag.PAN
				_anchor_x = x
				_pan_t0 = _view_t0()
				_pan_t1 = _view_t1()
				accept_event()
			elif _drag == Drag.PAN:
				_drag = Drag.NONE
				accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_drag = Drag.SELECT
				_anchor_x = x
				_cur_x = x
				_select_moved = false
				# Preview caret in pixels; commit times on release.
				sel_t0 = -1.0
				sel_t1 = -1.0
				queue_redraw()
				accept_event()
			elif _drag == Drag.SELECT:
				_finish_select(x)
				accept_event()
	elif event is InputEventMouseMotion:
		var x := clampf((event as InputEventMouseMotion).position.x, 0.0, size.x)
		if _drag == Drag.PAN:
			var dt := (_anchor_x - x) / size.x * (_pan_t1 - _pan_t0)
			host.set("_view_t0", _pan_t0 + dt)
			host.set("_view_t1", _pan_t1 + dt)
			host.call("_clamp_view")
			view_changed.emit()
			host.queue_redraw()
			queue_redraw()
			accept_event()
		elif _drag == Drag.SELECT:
			if not _select_moved and absf(x - _anchor_x) >= SELECT_DRAG_PX:
				_select_moved = true
			_cur_x = x
			queue_redraw()
			accept_event()


func _finish_select(x: float) -> void:
	_drag = Drag.NONE
	_cur_x = clampf(x, 0.0, size.x)
	if not _ready_size():
		return
	var tv := _px_pair_to_times(_anchor_x, _cur_x)
	sel_t0 = tv.x
	sel_t1 = tv.y
	_select_moved = false
	selection_changed.emit(sel_t0, sel_t1)
	if sel_t0 >= 0.0:
		host.call("_feed_face_at_time", sel_t0)
	queue_redraw()


func _zoom_at(x: float, factor: float) -> void:
	if not _ready_size():
		return
	var t_focus := _x_to_t(x)
	var span := _view_span() * factor
	var dur := float(host.get("_duration_s"))
	span = clampf(span, 0.05, maxf(0.05, dur))
	var left_frac := (t_focus - _view_t0()) / _view_span()
	host.set("_view_t0", t_focus - left_frac * span)
	host.set("_view_t1", float(host.get("_view_t0")) + span)
	host.call("_clamp_view")
	view_changed.emit()
	host.queue_redraw()
	queue_redraw()


func _draw() -> void:
	# Live drag preview from pixels (not stale times).
	if _drag == Drag.SELECT and _anchor_x >= 0.0:
		var x0 := mini(_anchor_x, _cur_x)
		var x1 := maxf(_anchor_x, _cur_x)
		if x1 - x0 < SELECT_DRAG_PX:
			draw_line(Vector2(x0, 0.0), Vector2(x0, size.y), Color(0.95, 0.85, 0.2, 0.95), 2.0)
		else:
			draw_rect(Rect2(x0, 0.0, x1 - x0, size.y), Color(0.95, 0.85, 0.2, 0.22))
		return
	if sel_t0 < 0.0 or not _ready_size():
		return
	var x0 := _t_to_x(mini(sel_t0, sel_t1))
	var x1 := _t_to_x(maxf(sel_t0, sel_t1))
	if absf(sel_t1 - sel_t0) < CARET_EPS_S or x1 - x0 < 2.0:
		draw_line(Vector2(x0, 0.0), Vector2(x0, size.y), Color(0.95, 0.85, 0.2, 0.95), 2.0)
	else:
		draw_rect(Rect2(x0, 0.0, maxf(2.0, x1 - x0), size.y), Color(0.95, 0.85, 0.2, 0.22))
