extends Control
## Quality UI face: RPM mouth close-up + orbit + live blendshape bars (left).

const VisemeFaceScript := preload("res://viseme_face.gd")
const OVR := [
	"sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U", "LA"
]

@onready var _split: HSplitContainer = $HSplit
@onready var _viewport: SubViewport = $HSplit/ViewWrap/SubViewportContainer/SubViewport
@onready var _face_host: Node3D = $HSplit/ViewWrap/SubViewportContainer/SubViewport/World
@onready var _bars: Control = $HSplit/BlendBars
@onready var _view_wrap: Control = $HSplit/ViewWrap
@onready var _svc: SubViewportContainer = $HSplit/ViewWrap/SubViewportContainer

var viseme_face: Node3D
var _cam: Camera3D
var _yaw := 0.0
## Mild look-down; framing tuned midway between prior neck-low and nose-high.
var _pitch := -17.0
var _orbiting := false
var _orbit_last := Vector2.ZERO
var _pivot := Vector3(0.0, 1.74, 0.09)
var _cam_dist := 0.18
var _manual_ovr := PackedFloat32Array()
var _bar_drag := -1


func _ready() -> void:
	custom_minimum_size = Vector2(280, 140)
	_manual_ovr.resize(OVR.size())
	_cam = Camera3D.new()
	_cam.fov = 24.0
	_face_host.add_child(_cam)
	_apply_camera()
	# After avatar loads, snap pivot to head mesh mouth band if possible.
	call_deferred("_retarget_mouth_pivot")

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-40.0, -20.0, 0.0)
	light.light_energy = 1.15
	_face_host.add_child(light)

	var ambient := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.12, 0.13, 0.16)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.45, 0.48, 0.55)
	env.ambient_light_energy = 0.55
	ambient.environment = env
	_face_host.add_child(ambient)

	viseme_face = VisemeFaceScript.new()
	viseme_face.name = "VisemeSystem"
	_face_host.add_child(viseme_face)

	var glb := "res://third_party/lipsync/readyplayerme_avatar.glb"
	if ResourceLoader.exists(glb):
		var packed: PackedScene = load(glb)
		if packed != null:
			var avatar: Node = packed.instantiate()
			avatar.name = "Avatar"
			viseme_face.add_child(avatar)
			viseme_face.call_deferred("_bind_meshes")
		else:
			_add_fallback_label("avatar load failed")
	else:
		_add_fallback_label("Open once in Godot 4.6 to import RPM GLB")

	_svc.mouse_filter = Control.MOUSE_FILTER_STOP
	_svc.gui_input.connect(_on_view_gui_input)
	_bars.mouse_filter = Control.MOUSE_FILTER_STOP
	_bars.draw.connect(_draw_bars)
	_bars.gui_input.connect(_on_bars_gui_input)
	_bars.resized.connect(func(): _bars.queue_redraw())
	resized.connect(_on_face_resized)
	_on_face_resized()


func _on_face_resized() -> void:
	## Keep blend bars ~20% of panel width.
	if _split == null:
		return
	var w := size.x
	if w > 1.0:
		_split.split_offset = int(clampf(w * 0.20, 100.0, 220.0))


func _add_fallback_label(msg: String) -> void:
	var lbl := Label3D.new()
	lbl.text = msg
	lbl.font_size = 28
	lbl.position = _pivot
	_face_host.add_child(lbl)


func _retarget_mouth_pivot() -> void:
	## Aim at mouth band of Wolf3D_Head (Julian: onto the mouth, not nose/neck).
	if viseme_face == null:
		return
	var head := viseme_face.find_child("Wolf3D_Head", true, false)
	if head is MeshInstance3D:
		var mi := head as MeshInstance3D
		var aabb: AABB = mi.get_aabb()
		var local_mouth := aabb.position + Vector3(
			aabb.size.x * 0.5,
			aabb.size.y * 0.36,
			aabb.size.z * 0.88
		)
		_pivot = mi.to_global(local_mouth)
		_cam_dist = clampf(aabb.size.length() * 0.40, 0.10, 0.24)
		_pitch = -12.0
		_apply_camera()


func _apply_camera() -> void:
	if _cam == null:
		return
	var yaw_r := deg_to_rad(_yaw)
	var pitch_r := deg_to_rad(clampf(_pitch, -35.0, 15.0))
	var offset := Vector3(
		sin(yaw_r) * cos(pitch_r),
		sin(pitch_r),
		cos(yaw_r) * cos(pitch_r)
	) * _cam_dist
	_cam.position = _pivot + offset
	_cam.look_at(_pivot, Vector3.UP)


func _on_view_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_orbiting = true
				_orbit_last = mb.position
				_svc.accept_event()
			else:
				_orbiting = false
				_svc.accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			_cam_dist = clampf(_cam_dist * 0.9, 0.14, 0.55)
			_apply_camera()
			_svc.accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			_cam_dist = clampf(_cam_dist * 1.1, 0.14, 0.55)
			_apply_camera()
			_svc.accept_event()
	elif event is InputEventMouseMotion and _orbiting:
		var mm := event as InputEventMouseMotion
		var d := mm.position - _orbit_last
		_orbit_last = mm.position
		_yaw = clampf(_yaw - d.x * 0.35, -55.0, 55.0)
		_pitch = clampf(_pitch + d.y * 0.25, -25.0, 20.0)
		_apply_camera()
		_svc.accept_event()


func set_visemes(vv: Variant) -> void:
	_manual_ovr.resize(OVR.size())
	for i in OVR.size():
		var w := 0.0
		if vv is PackedFloat32Array and i < vv.size():
			w = float(vv[i])
		elif vv is Array and i < vv.size():
			w = float(vv[i])
		_manual_ovr[i] = clampf(w, 0.0, 1.0)
	if viseme_face != null and viseme_face.has_method("set_visemes"):
		viseme_face.set_visemes(_manual_ovr)
	if _bars != null:
		_bars.queue_redraw()


func _bar_row(i: int) -> Rect2:
	## Horizontal bar rows for a tall left column.
	var r := _bars.size
	var n := float(OVR.size())
	var top := 22.0
	var gap := 2.0
	var h := maxf(8.0, (r.y - top - 8.0 - gap * (n - 1.0)) / n)
	var y := top + float(i) * (h + gap)
	var label_w := 36.0
	return Rect2(label_w, y, maxf(8.0, r.x - label_w - 8.0), h)


func _draw_bars() -> void:
	var r := _bars.size
	_bars.draw_rect(Rect2(Vector2.ZERO, r), Color(0.10, 0.11, 0.13))
	_bars.draw_string(
		ThemeDB.fallback_font, Vector2(6, 14),
		"blendshapes (drag)",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.75, 0.78, 0.85)
	)
	for i in OVR.size():
		var br := _bar_row(i)
		_bars.draw_string(
			ThemeDB.fallback_font,
			Vector2(4.0, br.position.y + br.size.y * 0.75),
			OVR[i],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.7, 0.72, 0.78)
		)
		_bars.draw_rect(br, Color(0.18, 0.19, 0.22))
		var w := br.size.x * (_manual_ovr[i] if i < _manual_ovr.size() else 0.0)
		var fill := Rect2(br.position.x, br.position.y, w, br.size.y)
		var col := Color(0.35, 0.75, 0.95) if i != 0 else Color(0.5, 0.5, 0.55)
		_bars.draw_rect(fill, col)


func _on_bars_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_bar_drag = _hit_bar(mb.position)
				if _bar_drag >= 0:
					_set_bar_from_x(_bar_drag, mb.position.x)
					_bars.accept_event()
			else:
				_bar_drag = -1
	elif event is InputEventMouseMotion and _bar_drag >= 0:
		_set_bar_from_x(_bar_drag, (event as InputEventMouseMotion).position.x)
		_bars.accept_event()


func _hit_bar(p: Vector2) -> int:
	for i in OVR.size():
		var br := _bar_row(i)
		var hit := Rect2(0.0, br.position.y, _bars.size.x, br.size.y)
		if hit.has_point(p):
			return i
	return -1


func _set_bar_from_x(i: int, x: float) -> void:
	var br := _bar_row(i)
	var u := clampf((x - br.position.x) / maxf(1.0, br.size.x), 0.0, 1.0)
	_manual_ovr.resize(OVR.size())
	_manual_ovr[i] = u
	if viseme_face != null and viseme_face.has_method("set_visemes"):
		viseme_face.set_visemes(_manual_ovr)
	_bars.queue_redraw()
