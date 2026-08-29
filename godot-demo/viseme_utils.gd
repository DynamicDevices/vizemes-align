extends RefCounted
class_name VisemeUtils
## MLP softmax → OVR viseme weights ([goatchurchprime/lipsync](https://github.com/goatchurchprime/lipsync) shape).

const OVR_NAMES: Array[String] = [
	"sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U", "LA"
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


static func stereo_to_mono(buffer: PackedVector2Array) -> PackedFloat32Array:
	var pcm := PackedFloat32Array()
	pcm.resize(buffer.size())
	for i in buffer.size():
		pcm[i] = (buffer[i].x + buffer[i].y) * 0.5
	return pcm


static func resample_pcm(pcm: PackedFloat32Array, from_rate: int, to_rate: int) -> PackedFloat32Array:
	if from_rate <= 0 or to_rate <= 0 or pcm.is_empty():
		return PackedFloat32Array()
	if from_rate == to_rate:
		return pcm
	var out_len := maxi(1, int(round(float(pcm.size()) * float(to_rate) / float(from_rate))))
	var out := PackedFloat32Array()
	out.resize(out_len)
	for i in out_len:
		var src_pos := float(i) * float(from_rate) / float(to_rate)
		var idx := int(src_pos)
		var frac := src_pos - float(idx)
		var a := pcm[mini(idx, pcm.size() - 1)]
		var b := pcm[mini(idx + 1, pcm.size() - 1)]
		out[i] = a + (b - a) * frac
	return out


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


## Pack soft weights → 1 byte: high nibble = viseme id (0–15), low nibble = blend (0–15).
## Designed for ~1 byte / 20 ms Opus sideband (MPEG-4-style hard path).
static func soft_to_hard_byte(weights: PackedFloat32Array) -> int:
	if weights.is_empty():
		return 0
	var id := clampi(argmax(weights), 0, 15)
	var peak := weights[id]
	var blend := clampi(int(round(peak * 15.0)), 0, 15)
	return (id << 4) | blend


static func hard_byte_id(b: int) -> int:
	return (b >> 4) & 0x0F


static func hard_byte_blend01(b: int) -> float:
	return float(b & 0x0F) / 15.0


## Expand a hard byte to a sparse weight vector (length n) for VisemeSystem / plots.
static func hard_byte_to_weights(b: int, n: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(maxi(1, n))
	for i in out.size():
		out[i] = 0.0
	var id := hard_byte_id(b)
	if id < out.size():
		out[id] = hard_byte_blend01(b)
	return out


## Encode a softmax/OVR series (one PackedFloat32Array per hop) to 20 ms hard bytes.
static func series_to_hard_bytes(
	series: Array, hop_s: float, t_series0: float, frame_s: float = 0.02
) -> PackedByteArray:
	var out := PackedByteArray()
	if series.is_empty() or hop_s <= 0.0 or frame_s <= 0.0:
		return out
	var t_end := t_series0 + float(series.size() - 1) * hop_s
	var t := t_series0
	while t <= t_end + 1e-6:
		var idx := int(round((t - t_series0) / hop_s))
		idx = clampi(idx, 0, series.size() - 1)
		var w: PackedFloat32Array = series[idx]
		out.append(soft_to_hard_byte(w))
		t += frame_s
	return out


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


## Parse model.json in GDScript (res:// or absolute) and call MelFrontend.configure(...).
static func configure_mel_from_json(mel: Object, json_path: String) -> bool:
	if mel == null:
		return false
	var f := FileAccess.open(json_path, FileAccess.READ)
	if f == null:
		push_error("configure_mel_from_json: open failed %s" % json_path)
		return false
	var data: Variant = JSON.parse_string(f.get_as_text())
	if typeof(data) != TYPE_DICTIONARY:
		push_error("configure_mel_from_json: bad JSON %s" % json_path)
		return false
	var audio: Dictionary = data.get("audio", {})
	var ctx := int(data.get("context_frames", 20))
	var n_mels := int(audio.get("n_mels", data.get("n_mels", 80)))
	var sr := int(audio.get("sample_rate", 16000))
	var hop := int(audio.get("hop_length_samples", 160))
	var win := int(audio.get("window_length_samples", 400))
	var n_fft := int(audio.get("n_fft", 1024))
	var fmin := float(audio.get("fmin", 50.0))
	var fmax := float(audio.get("fmax", 8000.0))
	var visemes: Variant = data.get("visemes", {})
	var n_vis := int(data.get("n_visemes", visemes.size() if visemes is Dictionary else 15))
	var feats := int(data.get("input_features", ctx * n_mels))
	return mel.configure(ctx, n_mels, sr, hop, win, n_fft, fmin, fmax, n_vis, feats)


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
