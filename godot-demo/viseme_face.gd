extends Node3D
class_name VisemeFace
## Drives Ready Player Me (goatchurchprime/lipsync) viseme_* blend shapes.
## API matches VisemeSystem.set_visemes(PackedFloat32Array) in OVR order.

const OVR := [
	"sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U", "LA"
]

var last_ovr: PackedFloat32Array = PackedFloat32Array()
var _meshes: Array = [] ## MeshInstance3D
var _shape_idx: Array = [] ## int per OVR name (-1 if missing)


func _ready() -> void:
	_bind_meshes()


func _bind_meshes() -> void:
	_meshes.clear()
	_shape_idx.clear()
	var head := find_child("Wolf3D_Head", true, false)
	var teeth := find_child("Wolf3D_Teeth", true, false)
	if head is MeshInstance3D:
		_meshes.append(head)
	if teeth is MeshInstance3D:
		_meshes.append(teeth)
	if _meshes.is_empty():
		push_warning("VisemeFace: no Wolf3D_Head/Teeth under %s (GLB not imported?)" % name)
		return
	var m0: MeshInstance3D = _meshes[0]
	for vs in OVR:
		_shape_idx.append(m0.find_blend_shape_by_name("viseme_%s" % vs))


func set_visemes(vv: Variant) -> void:
	if _meshes.is_empty():
		_bind_meshes()
	var n := OVR.size()
	last_ovr = PackedFloat32Array()
	last_ovr.resize(n)
	for i in n:
		var w := 0.0
		if vv is PackedFloat32Array and i < vv.size():
			w = float(vv[i])
		elif vv is Array and i < vv.size():
			w = float(vv[i])
		last_ovr[i] = clampf(w, 0.0, 1.0)
	for m in _meshes:
		var mesh: MeshInstance3D = m
		for i in n:
			var si: int = _shape_idx[i] if i < _shape_idx.size() else -1
			if si >= 0:
				mesh.set_blend_shape_value(si, last_ovr[i])
