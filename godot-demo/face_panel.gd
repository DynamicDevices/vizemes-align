extends Control
## Right-hand 3D face panel (Ready Player Me via goatchurchprime/lipsync morphs).

const VisemeFaceScript := preload("res://viseme_face.gd")

@onready var _viewport: SubViewport = $SubViewportContainer/SubViewport
@onready var _face_host: Node3D = $SubViewportContainer/SubViewport/World

var viseme_face: Node3D


func _ready() -> void:
	var cam := Camera3D.new()
	cam.position = Vector3(0.0, 1.55, 0.55)
	cam.look_at(Vector3(0.0, 1.52, 0.0))
	cam.fov = 35.0
	_face_host.add_child(cam)

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-35.0, -25.0, 0.0)
	light.light_energy = 1.1
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
			avatar.position = Vector3(0.0, 0.0, 0.0)
			viseme_face.add_child(avatar)
			# Re-bind after avatar enters tree.
			viseme_face.call_deferred("_bind_meshes")
		else:
			_add_fallback_label("avatar load failed")
	else:
		_add_fallback_label("Open once in Godot 4.6 to import RPM GLB")


func _add_fallback_label(msg: String) -> void:
	var lbl := Label3D.new()
	lbl.text = msg
	lbl.font_size = 32
	lbl.position = Vector3(0.0, 1.5, 0.0)
	_face_host.add_child(lbl)


func set_visemes(vv: Variant) -> void:
	if viseme_face != null and viseme_face.has_method("set_visemes"):
		viseme_face.set_visemes(vv)
