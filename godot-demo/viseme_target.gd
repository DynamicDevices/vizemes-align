extends RefCounted
class_name VisemeTarget
## Resolves goatchurchprime/lipsync VisemeSystem when present, else VisemeSystemStub.
## Real VisemeSystem.set_visemes expects weights in OVR order (incl. trailing LA).

const VisemeUtils := preload("res://viseme_utils.gd")


static func resolve(host: Node) -> Node:
	var real := host.get_node_or_null("VisemeSystem")
	if real != null and real.has_method("set_visemes"):
		return real
	var stub := host.get_node_or_null("VisemeSystemStub")
	if stub != null:
		return stub
	push_error("VisemeTarget: need VisemeSystem or VisemeSystemStub under %s" % host.name)
	return null


static func feed(target: Object, ovr: PackedFloat32Array) -> void:
	if target == null or ovr.is_empty():
		return
	# Pad/truncate to VisemeSystem length (16 with LA) or stub (same after LA add).
	var n := VisemeUtils.OVR_NAMES.size()
	var vv := PackedFloat32Array()
	vv.resize(n)
	for i in n:
		vv[i] = ovr[i] if i < ovr.size() else 0.0
	target.set_visemes(vv)


static func last_ovr(target: Object) -> PackedFloat32Array:
	if target == null:
		return PackedFloat32Array()
	if "last_ovr" in target:
		return target.last_ovr
	return PackedFloat32Array()
