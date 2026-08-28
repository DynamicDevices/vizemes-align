extends Control
## Translucent hit-layer over the plot only.
## Authoritative selection is in *seconds* (t0/t1); pixels are derived for drawing.
## Local x 0‥size.x maps linearly to host view window.

signal selection_changed(t0: float, t1: float)
signal view_changed

const SELECT_DRAG_PX := 8.0
const CARET_EPS_S := 0.002

enum Drag { NONE, PAN, SELECT }

var host: Node ## VisemeTimeline
var _drag := Drag.NONE
var _anchor_x := 0.0
var _anchor_t := 0.0
var _select_moved := false
var _pan_t0 := 0.0
var _pan_t1 := 0.0
## Authoritative selection (seconds). Equal → caret. Negative → none.
var sel_t0 := -1.0
var sel_t1 := -1.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	focus_mode = Control.FOCUS_ALL
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)


func clear_selection() -> void:
	sel_t0 = -1.0
	sel_t1 = -1.0
	queue_redraw()
	selection_changed.emit(-1.0, -1.0)


func has_caret() -> bool:
	return sel_t0 >= 0.0 and absf(sel_t1 - sel_t0) < CARET_EPS_S


func selection_times() -> Vector2:
	return Vector2(sel_t0, sel_t1)


func _view_t0() -> float:
	return float(host.get("_view_t0"))


func _view_t1() -> float:
	return float(host.get("_view_t1"))


func _view_span() -> float:
	return maxf(1e-6, _view_t1() - _view_t0())


func _x_to_t(x: float) -> float:
	var u := clampf(x / maxf(1.0, size.x), 0.0, 1.0)
	return _view_t0() + u * _view_span()


func _t_to_x(t: float) -> float:
	var u := clampf((t - _view_t0()) / _view_span(), 0.0, 1.0)
	return u * size.x


func _gui_input(event: InputEvent) -> void:
	if host == null or bool(host.get("_quit_on_done")):
		return
	# Keep Space/P working after clicks (LineEdit otherwise steals focus).
	if event is InputEventKey and event.pressed and not event.echo:
		host.call("_handle_key", event)
		accept_event()
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		var x := mb.position.x
		if mb.pressed:
			grab_focus()
			if host.has_method("grab_focus"):
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
				_anchor_t = _x_to_t(x)
				_select_moved = false
				sel_t0 = _anchor_t
				sel_t1 = _anchor_t
				_emit_sel()
				queue_redraw()
				accept_event()
			elif _drag == Drag.SELECT:
				_finish_select(x)
				accept_event()
	elif event is InputEventMouseMotion:
		var mm := event as InputEventMouseMotion
		var x := mm.position.x
		if _drag == Drag.PAN:
			var dt := (_anchor_x - x) / maxf(1.0, size.x) * (_pan_t1 - _pan_t0)
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
			if _select_moved:
				var t := _x_to_t(x)
				sel_t0 = mini(_anchor_t, t)
				sel_t1 = maxf(_anchor_t, t)
				_emit_sel()
				queue_redraw()
			accept_event()


func _finish_select(x: float) -> void:
	_drag = Drag.NONE
	var t := _x_to_t(x)
	if (not _select_moved) or absf(x - _anchor_x) < SELECT_DRAG_PX:
		sel_t0 = _anchor_t
		sel_t1 = _anchor_t
	else:
		sel_t0 = mini(_anchor_t, t)
		sel_t1 = maxf(_anchor_t, t)
	_select_moved = false
	_emit_sel()
	queue_redraw()


func _emit_sel() -> void:
	selection_changed.emit(sel_t0, sel_t1)
	if host != null and sel_t0 >= 0.0:
		host.call("_feed_face_at_time", sel_t0)


func _zoom_at(x: float, factor: float) -> void:
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
	if sel_t0 < 0.0:
		return
	var x0 := _t_to_x(mini(sel_t0, sel_t1))
	var x1 := _t_to_x(maxf(sel_t0, sel_t1))
	var w := x1 - x0
	if absf(sel_t1 - sel_t0) < CARET_EPS_S or w < 2.0:
		draw_line(Vector2(x0, 0.0), Vector2(x0, size.y), Color(0.95, 0.85, 0.2, 0.95), 2.0)
	else:
		draw_rect(Rect2(x0, 0.0, maxf(2.0, w), size.y), Color(0.95, 0.85, 0.2, 0.22))
