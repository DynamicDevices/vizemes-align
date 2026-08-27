extends RefCounted
class_name VisemeUtils
## MLP softmax → OVR viseme weights ([goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync) shape).

const OVR_NAMES: Array[String] = [
	"sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U"
]

const MODEL_NAME_TO_OVR: Dictionary = {
	"silence": "sil",
	"ih": "I",
	"oh": "O",
	"ou": "U",
}


static func load_wav_pcm(path: String) -> PackedFloat32Array:
	var stream: AudioStreamWAV = AudioStreamWAV.load_from_file(path)
	if stream == null:
		push_error("load wav failed: %s" % path)
		return PackedFloat32Array()
	if stream.stereo:
		push_error("expected mono wav: %s" % path)
		return PackedFloat32Array()
	if stream.format != AudioStreamWAV.FORMAT_16_BITS:
		push_error("expected 16-bit pcm wav: %s" % path)
		return PackedFloat32Array()
	var raw: PackedByteArray = stream.data
	var n := raw.size() / 2
	var pcm := PackedFloat32Array()
	pcm.resize(n)
	for i in n:
		pcm[i] = float(raw.decode_s16(i * 2)) / 32768.0
	return pcm


static func softmax(logits: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(logits.size())
	if logits.is_empty():
		return out
	var m := logits[0]
	for i in range(1, logits.size()):
		if logits[i] > m:
			m = logits[i]
	var sum := 0.0
	for i in logits.size():
		out[i] = exp(logits[i] - m)
		sum += out[i]
	if sum <= 0.0:
		sum = 1.0
	for i in logits.size():
		out[i] /= sum
	return out


static func argmax(weights: PackedFloat32Array) -> int:
	var best := 0
	for i in range(1, weights.size()):
		if weights[i] > weights[best]:
			best = i
	return best


static func load_id_to_name(json_path: String) -> Array:
	var f := FileAccess.open(json_path, FileAccess.READ)
	if f == null:
		return []
	var data: Variant = JSON.parse_string(f.get_as_text())
	if typeof(data) != TYPE_DICTIONARY:
		return []
	var visemes: Dictionary = data.get("visemes", {})
	var n: int = int(data.get("n_visemes", visemes.size()))
	var names: Array = []
	names.resize(n)
	for key in visemes.keys():
		names[int(visemes[key])] = str(key)
	return names


static func mlp_to_ovr(weights: PackedFloat32Array, id_to_name: Array) -> PackedFloat32Array:
	var ovr := PackedFloat32Array()
	ovr.resize(OVR_NAMES.size())
	for i in OVR_NAMES.size():
		ovr[i] = 0.0
	for i in weights.size():
		if i >= id_to_name.size():
			break
		var model_name: String = str(id_to_name[i])
		var ovr_name: String = str(MODEL_NAME_TO_OVR.get(model_name, model_name))
		var idx := OVR_NAMES.find(ovr_name)
		if idx >= 0:
			ovr[idx] = weights[i]
	return ovr


static func find_nearest_context(
	ctx_ref: PackedFloat32Array, contexts: Array, max_l2: float
) -> int:
	var best_idx := -1
	var best_d := max_l2
	for ci in contexts.size():
		var ctx: PackedFloat32Array = contexts[ci]
		if ctx.size() != ctx_ref.size():
			continue
		var d := 0.0
		for j in ctx.size():
			var diff := ctx[j] - ctx_ref[j]
			d += diff * diff
		d = sqrt(d)
		if d < best_d:
			best_d = d
			best_idx = ci
	return best_idx
