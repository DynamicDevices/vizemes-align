extends RefCounted
class_name VisemePipeline
## MelFrontend → OnnxLoader → OVR visemes (shared by lipsync + mic demos).

const VisemeUtils := preload("res://viseme_utils.gd")
const VisemeTarget := preload("res://viseme_target.gd")

var mel: Object
var loader: Object
var id_to_name: Array = []
var target_rate: int = 16000
var last_ovr: PackedFloat32Array = PackedFloat32Array()


func setup(json_path: String, onnx_path: String) -> bool:
	id_to_name = VisemeUtils.load_id_to_name(json_path)
	if id_to_name.is_empty():
		push_error("VisemePipeline: model.json visemes missing")
		return false
	mel = ClassDB.instantiate("MelFrontend")
	loader = ClassDB.instantiate("OnnxLoader")
	if mel == null or loader == null:
		push_error("VisemePipeline: GDExtensions missing")
		return false
	if not mel.configure_from_json(json_path):
		return false
	if not loader.load_model(onnx_path):
		return false
	target_rate = 16000
	return true


func begin_stream() -> void:
	mel.begin_stream()


func predict_ovr(ctx: PackedFloat32Array) -> PackedFloat32Array:
	var logits: PackedFloat32Array = loader.predict(ctx)
	if logits.is_empty():
		return PackedFloat32Array()
	var w := VisemeUtils.softmax(logits)
	return VisemeUtils.mlp_to_ovr(w, id_to_name)


func push_pcm_to_stub(pcm: PackedFloat32Array, stub: Object) -> int:
	var frames := 0
	for ctx_variant in mel.push_pcm_contexts(pcm):
		var ctx: PackedFloat32Array = ctx_variant
		var ovr := predict_ovr(ctx)
		if ovr.is_empty():
			continue
		VisemeTarget.feed(stub, ovr)
		last_ovr = ovr
		frames += 1
	return frames


func feed_pcm_mono_16k(pcm: PackedFloat32Array, target: Object) -> int:
	if pcm.is_empty():
		return 0
	return push_pcm_to_stub(pcm, target)
