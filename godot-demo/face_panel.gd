extends Control
## Quality UI face: RPM mouth close-up + orbit + live blendshape bars.

const VisemeFaceScript := preload("res://viseme_face.gd")
const OVR := [
	"sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U", "LA"
]

@onready var _split: VSplitContainer = $VSplit
@onready var _viewport: SubViewport = $VSplit/ViewWrap/SubViewportContainer/SubViewport
@onready var _face_host: Node3D = $VSplit/ViewWrap/SubViewportContainer/SubViewport/World
@onready var _bars: Control = $VSplit/BlendBars
@onready var _view_wrap: Control = $VSplit/ViewWrap
@onready var _svc: SubViewportContainer = $VSplit/ViewWrap/SubViewportContainer

var viseme_face: Node3D
var _cam: Camera3D
var _yaw := 0.0
var _pitch := 5.0
var _orbiting := false
var _orbit_last := Vector2.ZERO
## Mouth focus (RPM head metres).
var _pivot := Vector3(0.0, 1.48, 0.02)
var _cam_dist := 0.32
var _manual_ovr := PackedFloat32Array()
var _bar_drag := -1


func _ready() -> void:
	custom_minimum_size = Vector2(280, 200)
	_manual_ovr.resize(OVR.size())
	_cam = Camera3D.new()
	_cam.fov = 28.0
	_face_host.add_child(_cam)
	_apply_camera()

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
	if _split != null:
		_split.split_offset = 220


func _add_fallback_label(msg: String) -> void:
	var lbl := Label3D.new()
	lbl.text = msg
	lbl.font_size = 28
	lbl.position = _pivot
	_face_host.add_child(lbl)


func _apply_camera() -> void:
	if _cam == null:
		return
	var yaw_r := deg_to_rad(_yaw)
	var pitch_r := deg_to_rad(clampf(_pitch, -18.0, 25.0))
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
			_cam_dist = clampf(_cam_dist * 0.9, 0.18, 0.7)
			_apply_camera()
			_svc.accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			_cam_dist = clampf(_cam_dist * 1.1, 0.18, 0.7)
			_apply_camera()
			_svc.accept_event()
	elif event is InputEventMouseMotion and _orbiting:
		var mm := event as InputEventMouseMotion
		var d := mm.position - _orbit_last
		_orbit_last = mm.position
		_yaw = clampf(_yaw - d.x * 0.35, -55.0, 55.0)
		_pitch = clampf(_pitch + d.y * 0.25, -18.0, 25.0)
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


func _bar_rect(i: int) -> Rect2:
	var r := _bars.size
	var n := float(OVR.size())
	var gap := 2.0
	var w := maxf(4.0, (r.x - gap * (n + 1.0)) / n)
	var x := gap + float(i) * (w + gap)
	return Rect2(x, 18.0, w, maxf(8.0, r.y - 36.0))


func _draw_bars() -> void:
	var r := _bars.size
	_bars.draw_rect(Rect2(Vector2.ZERO, r), Color(0.10, 0.11, 0.13))
	_bars.draw_string(
		ThemeDB.fallback_font, Vector2(6, 14),
		"blendshapes (drag) — tune hardener by eye",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.75, 0.78, 0.85)
	)
	for i in OVR.size():
		var br := _bar_rect(i)
		_bars.draw_rect(br, Color(0.18, 0.19, 0.22))
		var h := br.size.y * (_manual_ovr[i] if i < _manual_ovr.size() else 0.0)
		var fill := Rect2(br.position.x, br.position.y + br.size.y - h, br.size.x, h)
		var col := Color(0.35, 0.75, 0.95) if i != 0 else Color(0.5, 0.5, 0.55)
		_bars.draw_rect(fill, col)
		if br.size.x >= 10.0:
			_bars.draw_string(
				ThemeDB.fallback_font,
				Vector2(br.position.x, r.y - 4.0),
				OVR[i],
				HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.7, 0.72, 0.78)
			)


func _on_bars_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_bar_drag = _hit_bar(mb.position)
				if _bar_drag >= 0:
					_set_bar_from_y(_bar_drag, mb.position.y)
					_bars.accept_event()
			else:
				_bar_drag = -1
	elif event is InputEventMouseMotion and _bar_drag >= 0:
		_set_bar_from_y(_bar_drag, (event as InputEventMouseMotion).position.y)
		_bars.accept_event()


func _hit_bar(p: Vector2) -> int:
	for i in OVR.size():
		if _bar_rect(i).has_point(p):
			return i
	return -1


func _set_bar_from_y(i: int, y: float) -> void:
	var br := _bar_rect(i)
	var u := 1.0 - clampf((y - br.position.y) / maxf(1.0, br.size.y), 0.0, 1.0)
	_manual_ovr.resize(OVR.size())
	_manual_ovr[i] = u
	if viseme_face != null and viseme_face.has_method("set_visemes"):
		viseme_face.set_visemes(_manual_ovr)
	_bars.queue_redraw()
