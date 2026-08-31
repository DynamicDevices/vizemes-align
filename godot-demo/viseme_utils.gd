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


static var _speex_rs = null
static var _speex_in_rate: int = 0
static var _speex_out_rate: int = 0


static func resample_pcm(pcm: PackedFloat32Array, from_rate: int, to_rate: int) -> PackedFloat32Array:
	if from_rate <= 0 or to_rate <= 0 or pcm.is_empty():
		return PackedFloat32Array()
	if from_rate == to_rate:
		return pcm
	# Prefer SpeexDSP GDExtension when present (DynamicDevices/godot-speexdsp).
	# Reuse one SpeexResampler and set_rate() when rates change (live mic).
	if ClassDB.class_exists("SpeexResampler"):
		if _speex_rs == null:
			_speex_rs = ClassDB.instantiate("SpeexResampler")
			if _speex_rs.setup(1, from_rate, to_rate, 5) != OK:
				_speex_rs = null
		elif _speex_in_rate != from_rate or _speex_out_rate != to_rate:
			if _speex_rs.set_rate(from_rate, to_rate) != OK:
				if _speex_rs.setup(1, from_rate, to_rate, 5) != OK:
					_speex_rs = null
		if _speex_rs != null:
			_speex_in_rate = from_rate
			_speex_out_rate = to_rate
			return _speex_rs.process(pcm)
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


## Local preview only: high nibble = viseme id, low nibble = confidence.
## This is deliberately not the Opus side-channel boundary-event contract.
static func soft_to_preview_byte(weights: PackedFloat32Array) -> int:
	if weights.is_empty():
		return 0
	var id := clampi(argmax(weights), 0, 15)
	var peak := weights[id]
	var blend := clampi(int(round(peak * 15.0)), 0, 15)
	return (id << 4) | blend


static func preview_byte_id(b: int) -> int:
	return (b >> 4) & 0x0F


static func preview_byte_confidence01(b: int) -> float:
	return float(b & 0x0F) / 15.0


## Expand a hard byte to a sparse weight vector (length n) for VisemeSystem / plots.
static func preview_byte_to_weights(b: int, n: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(maxi(1, n))
	for i in out.size():
		out[i] = 0.0
	var id := preview_byte_id(b)
	if id < out.size():
		out[id] = preview_byte_confidence01(b)
	return out


## Encode a softmax/OVR series (one PackedFloat32Array per hop) to 20 ms hard bytes.
static func series_to_preview_bytes(
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
		out.append(soft_to_preview_byte(w))
		t += frame_s
	return out


## Sparse one-byte boundary events carried once per 20 ms Opus packet.
const SIDECHANNEL_FRAME_SECONDS := 0.02
const SIDECHANNEL_MAX_VISEME_ID := 14
const SIDECHANNEL_MAX_PACKETS_AGO := 7
const SIDECHANNEL_NO_EVENT := 0xF0
const SIDECHANNEL_RESYNC_BASE := 0xF1
const SIDECHANNEL_START := 0
const SIDECHANNEL_END := 1


static func sidechannel_pack_boundary(target_id: int, kind: int, packets_ago: int) -> int:
	assert(target_id >= 0 and target_id <= SIDECHANNEL_MAX_VISEME_ID)
	assert(kind == SIDECHANNEL_START or kind == SIDECHANNEL_END)
	assert(packets_ago >= 0 and packets_ago <= SIDECHANNEL_MAX_PACKETS_AGO)
	var low := packets_ago | (0x08 if kind == SIDECHANNEL_END else 0)
	return (target_id << 4) | low


static func sidechannel_unpack_boundary(packet: int) -> Dictionary:
	assert(packet >= 0 and packet <= 0xFF)
	assert((packet >> 4) != 0x0F)
	var low := packet & 0x0F
	return {
		"target_id": packet >> 4,
		"kind": SIDECHANNEL_END if (low & 0x08) else SIDECHANNEL_START,
		"packets_ago": low & 0x07,
	}


static func sidechannel_pack_resync(current_id: int) -> int:
	assert(current_id >= 0 and current_id <= SIDECHANNEL_MAX_VISEME_ID)
	return SIDECHANNEL_RESYNC_BASE + current_id


static func sidechannel_unpack_resync(packet: int) -> int:
	assert(packet >= SIDECHANNEL_RESYNC_BASE and packet <= 0xFF)
	return packet - SIDECHANNEL_RESYNC_BASE


class SidechannelDecoder:
	var viseme_count: int = 15
	var base_id: int = 0
	var transition: Dictionary = {}

	func _init(p_viseme_count: int = 15) -> void:
		assert(p_viseme_count > 0 and p_viseme_count <= 15)
		viseme_count = p_viseme_count

	func ingest(packet: int, stream_packet: int) -> void:
		assert(packet >= 0 and packet <= 0xFF)
		if packet == 0xF0:
			return
		if (packet >> 4) == 0x0F:
			assert(packet >= 0xF1)
			base_id = packet - 0xF1
			assert(base_id < viseme_count)
			transition = {}
			return
		var low := packet & 0x0F
		var event: Dictionary = {
			"target_id": packet >> 4,
			"kind": 1 if (low & 0x08) else 0,
			"packets_ago": low & 0x07,
		}
		assert(int(event["target_id"]) < viseme_count)
		var boundary_packet := stream_packet - int(event["packets_ago"])
		if int(event["kind"]) == 0:
			transition = {
				"base_id": base_id,
				"target_id": int(event["target_id"]),
				"start_packet": boundary_packet,
			}
			return
		if transition.is_empty() or int(transition["target_id"]) != int(event["target_id"]):
			return # orphan END after packet loss; periodic resync recovers state
		assert(boundary_packet > int(transition["start_packet"]))
		transition["end_packet"] = boundary_packet

	func weights_at(audio_packet: float) -> PackedFloat32Array:
		var weights := PackedFloat32Array()
		weights.resize(viseme_count)
		if transition.is_empty() or not transition.has("end_packet"):
			weights[base_id] = 1.0
			return weights
		var start := float(transition["start_packet"])
		var end := float(transition["end_packet"])
		var progress := clampf((audio_packet - start) / (end - start), 0.0, 1.0)
		weights[int(transition["base_id"])] = 1.0 - progress
		weights[int(transition["target_id"])] += progress
		if progress >= 1.0:
			base_id = int(transition["target_id"])
			transition = {}
		return weights


## Configure solely from canonical ONNX vizemes_* metadata.
static func configure_mel_from_onnx(mel: Object, loader: Object) -> bool:
	if mel == null or loader == null:
		return false
	if not loader.has_method("get_metadata_value"):
		push_error("configure_mel_from_onnx: loader lacks metadata API")
		return false
	var blob := str(loader.get_metadata_value("vizemes_meta_json"))
	if not blob.is_empty():
		var data: Variant = JSON.parse_string(blob)
		if typeof(data) == TYPE_DICTIONARY:
			return _configure_mel_from_dict(mel, data)
	# Flat keys if full JSON missing
	var ctx := int(str(loader.get_metadata_value("vizemes_context_frames")))
	var n_mels := int(str(loader.get_metadata_value("vizemes_n_mels")))
	var sr := int(str(loader.get_metadata_value("vizemes_sample_rate")))
	if ctx > 0 and n_mels > 0 and sr > 0:
		var hop := int(str(loader.get_metadata_value("vizemes_hop_length_samples")))
		var win := int(str(loader.get_metadata_value("vizemes_window_length_samples")))
		var n_fft := int(str(loader.get_metadata_value("vizemes_n_fft")))
		var fmin := float(str(loader.get_metadata_value("vizemes_fmin")))
		var fmax := float(str(loader.get_metadata_value("vizemes_fmax")))
		var n_vis := int(str(loader.get_metadata_value("vizemes_n_visemes")))
		var feats := int(str(loader.get_metadata_value("vizemes_input_features")))
		if hop <= 0:
			hop = 160
		if win <= 0:
			win = 400
		if n_fft <= 0:
			n_fft = 1024
		if fmin <= 0.0:
			fmin = 50.0
		if fmax <= 0.0:
			fmax = 8000.0
		if n_vis <= 0:
			n_vis = 15
		if feats <= 0:
			feats = ctx * n_mels
		return mel.configure(ctx, n_mels, sr, hop, win, n_fft, fmin, fmax, n_vis, feats)
	push_error("configure_mel_from_onnx: no vizemes metadata on model")
	return false


## Print the ONNX input contract once at load, including the time history implied
## by a causal TCN's dilated convolution stack.
static func print_onnx_input_summary(loader: Object) -> void:
	if loader == null or not loader.has_method("get_metadata_value"):
		return
	var blob := str(loader.get_metadata_value("vizemes_meta_json"))
	var data: Variant = JSON.parse_string(blob)
	if typeof(data) != TYPE_DICTIONARY:
		return
	var meta: Dictionary = data
	var audio: Dictionary = meta.get("audio", {})
	var sample_rate := int(audio.get("sample_rate", 0))
	var hop_samples := int(audio.get("hop_length_samples", 0))
	var window_samples := int(audio.get("window_length_samples", 0))
	var n_mels := int(meta.get("n_mels", audio.get("n_mels", 0)))
	var context_frames := int(meta.get("context_frames", 0))
	var model_name := str(meta.get("model", "unknown"))
	if sample_rate <= 0 or hop_samples <= 0 or window_samples <= 0 or n_mels <= 0:
		return
	var hop_ms := 1000.0 * float(hop_samples) / float(sample_rate)
	var window_ms := 1000.0 * float(window_samples) / float(sample_rate)
	var overlap_ms := window_ms - hop_ms
	if model_name == "viseme_tcn":
		var layers := int(meta.get("layers", 0))
		var kernel := int(meta.get("kernel_size", 0))
		var receptive_frames := 1
		if layers > 0 and kernel > 0:
			receptive_frames += 2 * (kernel - 1) * ((1 << layers) - 1)
		print(
			"Vizemes ONNX input: model=%s mel=[time,%d]; window=%.1fms hop=%.1fms "
			% [model_name, n_mels, window_ms, hop_ms]
			+ "overlap=%.1fms; TCN input=[batch,time,%d], receptive_history=%d hops (%.1fms)."
			% [overlap_ms, n_mels, receptive_frames, float(receptive_frames) * hop_ms]
		)
		return
	var input_features := int(meta.get("input_features", context_frames * n_mels))
	var context_span_ms := window_ms + float(maxi(0, context_frames - 1)) * hop_ms
	print(
		"Vizemes ONNX input: model=%s mel=[%d,%d] flattened=%d; window=%.1fms hop=%.1fms "
		% [model_name, context_frames, n_mels, input_features, window_ms, hop_ms]
		+ "overlap=%.1fms; context_span=%.1fms."
		% [overlap_ms, context_span_ms]
	)


static func load_id_to_name_from_onnx(loader: Object) -> Array:
	if loader == null or not loader.has_method("get_metadata_value"):
		return []
	var blob := str(loader.get_metadata_value("vizemes_meta_json"))
	if blob.is_empty():
		return []
	var data: Variant = JSON.parse_string(blob)
	if typeof(data) != TYPE_DICTIONARY:
		return []
	var visemes: Dictionary = data.get("visemes", {})
	if visemes.is_empty():
		var mapper: Dictionary = data.get("phone_to_viseme", {})
		visemes = mapper.get("visemes", {})
	var n: int = int(data.get("n_visemes", visemes.size()))
	var names: Array = []
	names.resize(n)
	for key in visemes.keys():
		names[int(visemes[key])] = str(key)
	return names


## Deterministic stage B from a phone ONNX's embedded contract.
## Input is one phone-logit row per Mel hop; output is sparse viseme weights.
static func phone_logits_to_viseme_series(logit_series: Array, mapper: Dictionary) -> Array:
	var out: Array = []
	var ids: Array = mapper.get("phone_to_viseme_ids", [])
	var visemes: Dictionary = mapper.get("visemes", {})
	var n_visemes := visemes.size()
	var hop_s := float(mapper.get("hop_seconds", 0.0))
	var smoothing_s := float(mapper.get("smoothing_seconds", -1.0))
	var top_k := int(mapper.get("top_k", -1))
	if ids.is_empty() or n_visemes <= 0 or hop_s <= 0.0 or smoothing_s < 0.0 or top_k < 0:
		push_error("phone mapper contract is incomplete")
		return out
	var alpha := 1.0 if smoothing_s == 0.0 else 1.0 - exp(-hop_s / smoothing_s)
	var state := PackedFloat32Array()
	state.resize(n_visemes)
	for logits_variant in logit_series:
		var logits: PackedFloat32Array = logits_variant
		if logits.size() != ids.size():
			push_error("phone mapper logits=%d mapping=%d" % [logits.size(), ids.size()])
			return []
		var phones := softmax(logits)
		var mapped := PackedFloat32Array()
		mapped.resize(n_visemes)
		for phone_id in phones.size():
			var viseme_id := int(ids[phone_id])
			if viseme_id < 0 or viseme_id >= n_visemes:
				push_error("phone mapper viseme id out of range: %d" % viseme_id)
				return []
			mapped[viseme_id] += phones[phone_id]
		for viseme_id in n_visemes:
			state[viseme_id] += alpha * (mapped[viseme_id] - state[viseme_id])
		var sparse := state.duplicate()
		if top_k > 0 and top_k < n_visemes:
			var keep := PackedInt32Array()
			for _rank in top_k:
				var best := -1
				for candidate in n_visemes:
					if candidate not in keep and (best < 0 or state[candidate] > state[best]):
						best = candidate
				keep.append(best)
			for candidate in n_visemes:
				if candidate not in keep:
					sparse[candidate] = 0.0
		var total := 0.0
		for value in sparse:
			total += value
		if total > 0.0:
			for viseme_id in n_visemes:
				sparse[viseme_id] /= total
		out.append(sparse)
	return out


static func load_phone_mapper_from_onnx(loader: Object) -> Dictionary:
	if loader == null or not loader.has_method("get_metadata_value"):
		return {}
	var blob := str(loader.get_metadata_value("vizemes_meta_json"))
	var data: Variant = JSON.parse_string(blob)
	if typeof(data) != TYPE_DICTIONARY:
		return {}
	var mapper: Variant = data.get("phone_to_viseme", {})
	return mapper if typeof(mapper) == TYPE_DICTIONARY else {}


static func _configure_mel_from_dict(mel: Object, data: Dictionary) -> bool:
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
	var n_vis_default: int = 15
	if visemes is Dictionary:
		n_vis_default = visemes.size()
	var n_vis := int(data.get("n_visemes", n_vis_default))
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


## MFA / training-timeline soft weights at time t (model class space).
## Inside a box → one-hot expect_id. Near a box boundary → monotonic crossfade
## between the two neighbouring expect ids over `transition_s` (default 60 ms).
static func expect_soft_at_time(
	boxes: Array, n_visemes: int, t: float, transition_s: float = 0.06
) -> PackedFloat32Array:
	var n := maxi(1, n_visemes)
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		out[i] = 0.0
	if boxes.is_empty():
		# silence / empty
		out[0] = 1.0
		return out
	var half := maxf(0.0, transition_s) * 0.5
	var cur_i := -1
	var prev_i := -1
	var next_i := -1
	for i in boxes.size():
		var b: Variant = boxes[i]
		if typeof(b) != TYPE_DICTIONARY:
			continue
		var t0 := float(b.get("start", 0.0))
		var t1 := float(b.get("end", 0.0))
		if t >= t0 and t < t1:
			cur_i = i
			break
		if t1 <= t:
			prev_i = i
		elif next_i < 0 and t0 > t:
			next_i = i
	# Gap between boxes: treat as silence unless still inside a transition.
	if cur_i < 0:
		if prev_i >= 0 and next_i >= 0 and half > 0.0:
			var t_b := float((boxes[prev_i] as Dictionary).get("end", t))
			# Prefer abut mid-point if next starts later.
			var t_n0 := float((boxes[next_i] as Dictionary).get("start", t_b))
			t_b = 0.5 * (t_b + t_n0)
			if absf(t - t_b) <= half:
				var alpha := clampf((t - (t_b - half)) / maxf(1e-6, transition_s), 0.0, 1.0)
				_blend_expect_ids(out, boxes, prev_i, next_i, alpha)
				return out
		out[0] = 1.0
		return out
	var cur: Dictionary = boxes[cur_i]
	var id_cur := clampi(int(cur.get("expect_id", 0)), 0, n - 1)
	if half <= 0.0:
		out[id_cur] = 1.0
		return out
	var t0c := float(cur.get("start", 0.0))
	var t1c := float(cur.get("end", 0.0))
	# Don't let blend eat an entire short box.
	var local_half := minf(half, maxf(0.0, (t1c - t0c) * 0.45))
	var local_trans := maxf(1e-6, local_half * 2.0)
	# Leading edge: blend from previous box.
	if cur_i > 0 and local_half > 0.0 and t < t0c + local_half:
		var alpha := clampf((t - (t0c - local_half)) / local_trans, 0.0, 1.0)
		_blend_expect_ids(out, boxes, cur_i - 1, cur_i, alpha)
		return out
	# Trailing edge: blend into next box.
	if cur_i + 1 < boxes.size() and local_half > 0.0 and t > t1c - local_half:
		var alpha := clampf((t - (t1c - local_half)) / local_trans, 0.0, 1.0)
		_blend_expect_ids(out, boxes, cur_i, cur_i + 1, alpha)
		return out
	out[id_cur] = 1.0
	return out


static func _blend_expect_ids(
	out: PackedFloat32Array, boxes: Array, i_a: int, i_b: int, alpha: float
) -> void:
	var n := out.size()
	for i in n:
		out[i] = 0.0
	var a: Dictionary = boxes[i_a]
	var b: Dictionary = boxes[i_b]
	var id_a := clampi(int(a.get("expect_id", 0)), 0, n - 1)
	var id_b := clampi(int(b.get("expect_id", 0)), 0, n - 1)
	var u := clampf(alpha, 0.0, 1.0)
	# Smoothstep for less mechanical mid-point hold.
	u = u * u * (3.0 - 2.0 * u)
	if id_a == id_b:
		out[id_a] = 1.0
	else:
		out[id_a] = 1.0 - u
		out[id_b] = u


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
