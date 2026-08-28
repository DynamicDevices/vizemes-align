extends Control
## Translucent hit-layer over the plot: pan / zoom / select.
## Own local x (0‥size.x) maps linearly to host view time — no left-margin math.

signal selection_changed(t0: float, t1: float)
signal view_changed

const SELECT_DRAG_PX := 10.0
const CARET_EPS_S := 0.001

enum Drag { NONE, PAN, SELECT }

var host: Node ## VisemeTimeline
var _drag := Drag.NONE
var _anchor_x := 0.0
var _select_moved := false
var _pan_t0 := 0.0
var _pan_t1 := 0.0
## Selection in overlay-local pixels (authoritative for drawing).
var sel_px0 := -1.0
var sel_px1 := -1.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)


func clear_selection() -> void:
	sel_px0 = -1.0
	sel_px1 = -1.0
	queue_redraw()
	selection_changed.emit(-1.0, -1.0)


func has_caret() -> bool:
	return sel_px0 >= 0.0 and absf(sel_px1 - sel_px0) < SELECT_DRAG_PX


func selection_times() -> Vector2:
	## Returns (t0, t1); (-1,-1) if none. Caret → equal times.
	if sel_px0 < 0.0 or host == null:
		return Vector2(-1.0, -1.0)
	var a := _x_to_t(mini(sel_px0, sel_px1))
	var b := _x_to_t(maxf(sel_px0, sel_px1))
	if absf(sel_px1 - sel_px0) < SELECT_DRAG_PX:
		b = a
	return Vector2(a, b)


func _view_span() -> float:
	return maxf(1e-6, float(host.get("_view_t1")) - float(host.get("_view_t0")))


func _x_to_t(x: float) -> float:
	var t0 := float(host.get("_view_t0"))
	var u := clampf(x / maxf(1.0, size.x), 0.0, 1.0)
	return t0 + u * _view_span()


func _t_to_x(t: float) -> float:
	var t0 := float(host.get("_view_t0"))
	var u := clampf((t - t0) / _view_span(), 0.0, 1.0)
	return u * size.x


func _gui_input(event: InputEvent) -> void:
	if host == null or bool(host.get("_quit_on_done")):
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		var x := mb.position.x
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
				_pan_t0 = float(host.get("_view_t0"))
				_pan_t1 = float(host.get("_view_t1"))
				accept_event()
			elif _drag == Drag.PAN:
				_drag = Drag.NONE
				accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_drag = Drag.SELECT
				_anchor_x = x
				_select_moved = false
				sel_px0 = x
				sel_px1 = x
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
				sel_px0 = mini(_anchor_x, x)
				sel_px1 = maxf(_anchor_x, x)
				_emit_sel()
				queue_redraw()
			accept_event()


func _finish_select(x: float) -> void:
	_drag = Drag.NONE
	if (not _select_moved) or absf(x - _anchor_x) < SELECT_DRAG_PX:
		sel_px0 = _anchor_x
		sel_px1 = _anchor_x
	else:
		sel_px0 = mini(_anchor_x, x)
		sel_px1 = maxf(_anchor_x, x)
	_select_moved = false
	_emit_sel()
	queue_redraw()


func _emit_sel() -> void:
	var tv := selection_times()
	selection_changed.emit(tv.x, tv.y)
	if host != null and tv.x >= 0.0:
		host.call("_feed_face_at_time", tv.x)


func _zoom_at(x: float, factor: float) -> void:
	var t_focus := _x_to_t(x)
	var span := _view_span() * factor
	var dur := float(host.get("_duration_s"))
	span = clampf(span, 0.05, maxf(0.05, dur))
	var left_frac := (t_focus - float(host.get("_view_t0"))) / _view_span()
	host.set("_view_t0", t_focus - left_frac * span)
	host.set("_view_t1", float(host.get("_view_t0")) + span)
	host.call("_clamp_view")
	# Re-pin caret pixels to stored times after view change.
	var ht0 := float(host.get("_sel_t0"))
	var ht1 := float(host.get("_sel_t1"))
	if ht0 >= 0.0:
		sel_px0 = _t_to_x(ht0)
		sel_px1 = _t_to_x(ht1) if ht1 >= 0.0 else sel_px0
		if absf(ht1 - ht0) < CARET_EPS_S:
			sel_px1 = sel_px0
	view_changed.emit()
	host.queue_redraw()
	queue_redraw()


func _draw() -> void:
	if sel_px0 < 0.0:
		return
	var x0 := mini(sel_px0, sel_px1)
	var x1 := maxf(sel_px0, sel_px1)
	var w := x1 - x0
	if w < SELECT_DRAG_PX:
		draw_line(Vector2(x0, 0.0), Vector2(x0, size.y), Color(0.95, 0.85, 0.2, 0.95), 2.0)
	else:
		draw_rect(Rect2(x0, 0.0, w, size.y), Color(0.95, 0.85, 0.2, 0.22))
