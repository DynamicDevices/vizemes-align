extends Node
class_name VisemeSystemStub
## Stand-in for goatchurchprime/lipsync VisemeSystem — records set_visemes for smoke tests.

const VisemeUtils := preload("res://viseme_utils.gd")

var last_ovr: PackedFloat32Array = PackedFloat32Array()
var call_count: int = 0


func set_visemes(vv: Variant) -> void:
	call_count += 1
	if vv is PackedFloat32Array:
		last_ovr = vv
	elif vv is Dictionary:
		var ovr := PackedFloat32Array()
		ovr.resize(VisemeUtils.OVR_NAMES.size())
		for i in VisemeUtils.OVR_NAMES.size():
			var key: String = VisemeUtils.OVR_NAMES[i]
			ovr[i] = float(vv.get(key, 0.0))
		last_ovr = ovr
	else:
		push_error("set_visemes: expected PackedFloat32Array or Dictionary")
