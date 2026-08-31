#include "MelFrontend.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include "mel_spectrogram.h"
#include "speex_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

MelFrontend::MelFrontend()
{
	ops = vizemes_frontend_mel();
}

MelFrontend::~MelFrontend()
{
	reset();
}

void MelFrontend::clear_stream()
{
	::free(stream_pcm);
	stream_pcm = nullptr;
	stream_pcm_n = 0;
	stream_pcm_cap = 0;
	stream_contexts_emitted = 0;
	context_queue.clear();
}

void MelFrontend::destroy_resampler()
{
	if (resampler) {
		speex_resampler_destroy(resampler);
		resampler = nullptr;
	}
	resampler_in_rate = 0;
}

bool MelFrontend::ensure_resampler(int from_rate)
{
	if (from_rate <= 0 || cfg.sample_rate <= 0) {
		return false;
	}
	if (from_rate == cfg.sample_rate) {
		destroy_resampler();
		return true;
	}
	if (resampler && resampler_in_rate == from_rate) {
		return true;
	}
	if (resampler) {
		/* Live mic mix_rate can change (device switch) — adapt without teardown. */
		int err = speex_resampler_set_rate(resampler, (spx_uint32_t)from_rate,
				(spx_uint32_t)cfg.sample_rate);
		if (err == RESAMPLER_ERR_SUCCESS) {
			resampler_in_rate = from_rate;
			return true;
		}
		UtilityFunctions::push_warning(
				"MelFrontend: speex_resampler_set_rate failed; recreating resampler");
		destroy_resampler();
	}
	int err = 0;
	resampler = speex_resampler_init(1, (spx_uint32_t)from_rate, (spx_uint32_t)cfg.sample_rate, 5, &err);
	if (!resampler || err != RESAMPLER_ERR_SUCCESS) {
		resampler = nullptr;
		UtilityFunctions::push_error("MelFrontend: Speex resampler init failed");
		return false;
	}
	resampler_in_rate = from_rate;
	return true;
}

void MelFrontend::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("configure", "context_frames", "n_mels", "sample_rate",
						   "hop_length_samples", "window_length_samples", "n_fft", "fmin", "fmax",
						   "n_visemes", "input_features"),
			&MelFrontend::configure, DEFVAL(15), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("configure_preprocess", "agc", "vad", "denoise", "agc_level",
						   "gate_on_vad", "frame_size_ms"),
			&MelFrontend::configure_preprocess, DEFVAL(true), DEFVAL(true), DEFVAL(true),
			DEFVAL(8000.f), DEFVAL(false), DEFVAL(10));
	ClassDB::bind_method(D_METHOD("disable_preprocess"), &MelFrontend::disable_preprocess);
	ClassDB::bind_method(D_METHOD("get_last_vad"), &MelFrontend::get_last_vad);
	ClassDB::bind_method(D_METHOD("is_preprocess_enabled"), &MelFrontend::is_preprocess_enabled);
	ClassDB::bind_method(D_METHOD("configure_aec", "filter_ms", "frame_size_ms"),
			&MelFrontend::configure_aec, DEFVAL(100), DEFVAL(10));
	ClassDB::bind_method(D_METHOD("disable_aec"), &MelFrontend::disable_aec);
	ClassDB::bind_method(D_METHOD("is_aec_enabled"), &MelFrontend::is_aec_enabled);
	ClassDB::bind_method(D_METHOD("push_far_end_pcm", "pcm"), &MelFrontend::push_far_end_pcm);
	ClassDB::bind_method(D_METHOD("push_far_end_stereo", "frames", "mix_rate"),
			&MelFrontend::push_far_end_stereo);
	ClassDB::bind_method(D_METHOD("reset"), &MelFrontend::reset);
	ClassDB::bind_method(D_METHOD("begin_stream"), &MelFrontend::begin_stream);
	ClassDB::bind_method(D_METHOD("push_pcm", "pcm"), &MelFrontend::push_pcm);
	ClassDB::bind_method(D_METHOD("push_pcm_stereo", "frames", "mix_rate"),
			&MelFrontend::push_pcm_stereo);
	ClassDB::bind_method(D_METHOD("count_available_contexts"), &MelFrontend::count_available_contexts);
	ClassDB::bind_method(D_METHOD("get_next_context"), &MelFrontend::get_next_context);
	ClassDB::bind_method(D_METHOD("last_context_time_offset"), &MelFrontend::last_context_time_offset);
	ClassDB::bind_method(D_METHOD("build_utterance_contexts", "pcm"),
			&MelFrontend::build_utterance_contexts);
	ClassDB::bind_method(D_METHOD("build_utterance_mels", "pcm"),
			&MelFrontend::build_utterance_mels);
	ClassDB::bind_method(D_METHOD("get_input_features"), &MelFrontend::get_input_features);
	ClassDB::bind_method(D_METHOD("get_context_frames"), &MelFrontend::get_context_frames);
	ClassDB::bind_method(D_METHOD("get_n_mels"), &MelFrontend::get_n_mels);
	ClassDB::bind_method(D_METHOD("get_sample_rate"), &MelFrontend::get_sample_rate);
	ClassDB::bind_method(D_METHOD("get_hop_length_samples"), &MelFrontend::get_hop_length_samples);
	ClassDB::bind_method(D_METHOD("get_resampler_input_latency"),
			&MelFrontend::get_resampler_input_latency);
	ClassDB::bind_method(D_METHOD("get_resampler_output_latency"),
			&MelFrontend::get_resampler_output_latency);
	ClassDB::bind_method(D_METHOD("get_preprocess_frame_size"), &MelFrontend::get_preprocess_frame_size);
	ClassDB::bind_method(D_METHOD("get_aec_frame_size"), &MelFrontend::get_aec_frame_size);
	ClassDB::bind_method(D_METHOD("get_dsp_latency_samples"), &MelFrontend::get_dsp_latency_samples);
	ClassDB::bind_method(D_METHOD("get_dsp_latency_seconds"), &MelFrontend::get_dsp_latency_seconds);
}

bool MelFrontend::apply_config()
{
	if (!ops) {
		return false;
	}
	cfg.n_features = n_mels;
	cfg.top_db = 80.f;
	if (ops->init && ops->init(&cfg) != 0) {
		return false;
	}
	configured = true;
	return true;
}

bool MelFrontend::configure(int p_context_frames, int p_n_mels, int p_sample_rate,
		int p_hop_length_samples, int p_window_length_samples, int p_n_fft, float p_fmin,
		float p_fmax, int p_n_visemes, int p_input_features)
{
	reset();
	if (p_context_frames <= 0 || p_n_mels <= 0 || p_sample_rate <= 0 || p_hop_length_samples <= 0 ||
			p_window_length_samples <= 0 || p_n_fft <= 0) {
		return false;
	}
	context_frames = p_context_frames;
	n_mels = p_n_mels;
	n_visemes = p_n_visemes > 0 ? p_n_visemes : 15;
	input_features = p_input_features > 0 ? p_input_features : (p_context_frames * p_n_mels);
	cfg.sample_rate = p_sample_rate;
	cfg.hop_length_samples = p_hop_length_samples;
	cfg.window_length_samples = p_window_length_samples;
	cfg.n_fft = p_n_fft;
	cfg.fmin = p_fmin;
	cfg.fmax = p_fmax;
	return apply_config();
}

void MelFrontend::reset()
{
	if (ops && ops->free) {
		ops->free();
	}
	clear_stream();
	destroy_resampler();
	preprocess.destroy();
	aec.destroy();
	cfg = {};
	context_frames = 0;
	n_mels = 0;
	input_features = 0;
	n_visemes = 0;
	configured = false;
}

bool MelFrontend::configure_preprocess(bool agc, bool vad, bool denoise, float agc_level,
		bool gate_on_vad, int frame_size_ms)
{
	if (!configured || cfg.sample_rate <= 0) {
		UtilityFunctions::push_error("MelFrontend.configure_preprocess: call configure() first");
		return false;
	}
	if (!preprocess.setup(cfg.sample_rate, frame_size_ms, agc, vad, denoise, agc_level, gate_on_vad)) {
		UtilityFunctions::push_error("MelFrontend: Speex preprocess init failed");
		return false;
	}
	return true;
}

void MelFrontend::disable_preprocess()
{
	preprocess.destroy();
}

bool MelFrontend::get_last_vad() const
{
	return preprocess.get_last_vad();
}

bool MelFrontend::is_preprocess_enabled() const
{
	return preprocess.is_enabled();
}

bool MelFrontend::configure_aec(int filter_ms, int frame_size_ms)
{
	if (!configured || cfg.sample_rate <= 0) {
		UtilityFunctions::push_error("MelFrontend.configure_aec: call configure() first");
		return false;
	}
	if (!aec.setup(cfg.sample_rate, frame_size_ms, filter_ms)) {
		UtilityFunctions::push_error("MelFrontend: Speex AEC init failed");
		return false;
	}
	return true;
}

void MelFrontend::disable_aec()
{
	aec.destroy();
}

bool MelFrontend::is_aec_enabled() const
{
	return aec.is_enabled();
}

void MelFrontend::push_far_end_pcm(const PackedFloat32Array &pcm)
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (!aec.is_enabled() || pcm.is_empty()) {
		return;
	}
	aec.push_far_end(pcm.ptr(), (size_t)pcm.size());
}

void MelFrontend::push_far_end_stereo(const PackedVector2Array &frames, int mix_rate)
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (!configured || !aec.is_enabled() || frames.is_empty() || mix_rate <= 0) {
		return;
	}
	const int n = (int)frames.size();
	std::vector<float> mono((size_t)n);
	for (int i = 0; i < n; i++) {
		Vector2 v = frames[i];
		mono[(size_t)i] = 0.5f * (v.x + v.y);
	}
	PackedFloat32Array pcm16 = resample_mono(mono.data(), (size_t)n, mix_rate);
	if (!pcm16.is_empty()) {
		aec.push_far_end(pcm16.ptr(), (size_t)pcm16.size());
	}
}

void MelFrontend::begin_stream()
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (!configured) {
		return;
	}
	clear_stream();
	if (resampler) {
		speex_resampler_reset_mem(resampler);
	}
	preprocess.clear_pending();
	aec.clear_pending();
}

void MelFrontend::enqueue_new_contexts(const Array &fresh)
{
	for (int i = 0; i < fresh.size(); i++) {
		context_queue.push_back(fresh[i]);
	}
}

Array MelFrontend::contexts_from_pcm(const float *pcm, size_t n_samples, size_t skip_contexts) const
{
	Array out;
	if (!configured || !pcm || n_samples == 0) {
		return out;
	}

	const int nm = n_mels;
	const int ctx = context_frames;

	size_t max_frames = n_samples / (size_t)cfg.hop_length_samples + 16;
	float *mel = (float *)calloc(max_frames * (size_t)nm, sizeof(float));
	if (!mel) {
		return out;
	}

	size_t nframes = 0;
	if (mel_spectrogram_process(pcm, n_samples, mel, &nframes) < 0 || nframes < (size_t)ctx) {
		::free(mel);
		return out;
	}

	float *mu = (float *)calloc((size_t)nm, sizeof(float));
	float *sd = (float *)calloc((size_t)nm, sizeof(float));
	if (!mu || !sd) {
		::free(mu);
		::free(sd);
		::free(mel);
		return out;
	}
	for (int j = 0; j < nm; j++) {
		double sum = 0.0;
		double sum2 = 0.0;
		for (size_t t = 0; t < nframes; t++) {
			double v = (double)mel[t * (size_t)nm + (size_t)j];
			sum += v;
			sum2 += v * v;
		}
		mu[j] = (float)(sum / (double)nframes);
		double var = sum2 / (double)nframes - (double)mu[j] * (double)mu[j];
		if (var < 0.0) {
			var = 0.0;
		}
		sd[j] = (float)sqrt(var) + 1e-5f;
	}
	for (size_t t = 0; t < nframes; t++) {
		for (int j = 0; j < nm; j++) {
			size_t idx = t * (size_t)nm + (size_t)j;
			mel[idx] = (mel[idx] - mu[j]) / sd[j];
		}
	}
	::free(mu);
	::free(sd);

	const size_t n_contexts = nframes - (size_t)ctx + 1;
	for (size_t ci = skip_contexts; ci < n_contexts; ci++) {
		size_t i = ci + (size_t)ctx - 1;
		PackedFloat32Array flat;
		flat.resize(input_features);
		for (int f = 0; f < ctx; f++) {
			size_t frame_idx = i - (size_t)ctx + 1 + (size_t)f;
			for (int j = 0; j < nm; j++) {
				flat[f * nm + j] = mel[frame_idx * (size_t)nm + (size_t)j];
			}
		}
		out.push_back(flat);
	}

	::free(mel);
	return out;
}

void MelFrontend::append_stream_pcm(const float *pcm, size_t n)
{
	if (!pcm || n == 0) {
		return;
	}
	if (stream_pcm_n + n > stream_pcm_cap) {
		size_t need = stream_pcm_n + n;
		float *nb = (float *)realloc(stream_pcm, need * sizeof(float));
		if (!nb) {
			return;
		}
		stream_pcm = nb;
		stream_pcm_cap = need;
	}
	memcpy(stream_pcm + stream_pcm_n, pcm, n * sizeof(float));
	stream_pcm_n += n;

	Array fresh = contexts_from_pcm(stream_pcm, stream_pcm_n, stream_contexts_emitted);
	stream_contexts_emitted += (size_t)fresh.size();
	enqueue_new_contexts(fresh);
}

void MelFrontend::push_pcm(const PackedFloat32Array &pcm)
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (!configured) {
		return;
	}
	const int n_in = (int)pcm.size();
	if (n_in <= 0) {
		return;
	}

	const float *near = pcm.ptr();
	size_t near_n = (size_t)n_in;
	std::vector<float> after_aec;
	if (aec.is_enabled()) {
		after_aec.reserve(near_n);
		aec.process_and_append(near, near_n, after_aec);
		near = after_aec.data();
		near_n = after_aec.size();
		if (near_n == 0) {
			return;
		}
	}

	if (preprocess.is_enabled()) {
		std::vector<float> processed;
		processed.reserve(near_n);
		preprocess.process_and_append(near, near_n, processed);
		if (!processed.empty()) {
			append_stream_pcm(processed.data(), processed.size());
		}
		return;
	}
	append_stream_pcm(near, near_n);
}

PackedFloat32Array MelFrontend::resample_mono(const float *mono, size_t n, int from_rate)
{
	PackedFloat32Array out;
	if (!mono || n == 0 || from_rate <= 0 || cfg.sample_rate <= 0) {
		return out;
	}
	if (from_rate == cfg.sample_rate) {
		out.resize((int)n);
		memcpy(out.ptrw(), mono, n * sizeof(float));
		return out;
	}
	if (!ensure_resampler(from_rate) || !resampler) {
		return out;
	}
	spx_uint32_t in_len = (spx_uint32_t)n;
	const double ratio = (double)cfg.sample_rate / (double)from_rate;
	spx_uint32_t out_cap = (spx_uint32_t)std::ceil(n * ratio) + 16u +
			(spx_uint32_t)speex_resampler_get_output_latency(resampler);
	if (out_cap < 1) {
		out_cap = 1;
	}
	std::vector<float> in_buf(mono, mono + n);
	std::vector<float> out_buf(out_cap);
	spx_uint32_t out_len = out_cap;
	int err = speex_resampler_process_float(resampler, 0, in_buf.data(), &in_len, out_buf.data(),
			&out_len);
	if (err != RESAMPLER_ERR_SUCCESS) {
		UtilityFunctions::push_error("MelFrontend: Speex resample failed");
		return out;
	}
	out.resize((int)out_len);
	if (out_len > 0) {
		memcpy(out.ptrw(), out_buf.data(), (size_t)out_len * sizeof(float));
	}
	return out;
}

void MelFrontend::push_pcm_stereo(const PackedVector2Array &frames, int mix_rate)
{
	if (!configured || frames.is_empty() || mix_rate <= 0) {
		return;
	}
	const int n = (int)frames.size();
	std::vector<float> mono((size_t)n);
	for (int i = 0; i < n; i++) {
		Vector2 v = frames[i];
		mono[(size_t)i] = 0.5f * (v.x + v.y);
	}
	PackedFloat32Array pcm16 = resample_mono(mono.data(), (size_t)n, mix_rate);
	push_pcm(pcm16);
}

int MelFrontend::count_available_contexts() const
{
	std::lock_guard<std::mutex> lock(stream_mu);
	return (int)context_queue.size();
}

PackedFloat32Array MelFrontend::get_next_context()
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (context_queue.empty()) {
		return PackedFloat32Array();
	}
	PackedFloat32Array front = context_queue.front();
	context_queue.erase(context_queue.begin());
	return front;
}

float MelFrontend::last_context_time_offset() const
{
	std::lock_guard<std::mutex> lock(stream_mu);
	if (!configured || cfg.sample_rate <= 0 || cfg.hop_length_samples <= 0) {
		return 0.f;
	}
	const size_t queued = context_queue.size();
	if (stream_contexts_emitted == 0 && queued == 0) {
		return (float)stream_pcm_n / (float)cfg.sample_rate;
	}
	const size_t next_ci = stream_contexts_emitted - queued;
	const size_t end_frame = next_ci + (size_t)context_frames - 1;
	const double context_end_s =
			((double)(end_frame + 1) * (double)cfg.hop_length_samples) / (double)cfg.sample_rate;
	const double now_s = (double)stream_pcm_n / (double)cfg.sample_rate;
	double off = now_s - context_end_s;
	if (off < 0.0) {
		off = 0.0;
	}
	return (float)off;
}

Array MelFrontend::build_utterance_contexts(const PackedFloat32Array &pcm)
{
	begin_stream();
	push_pcm(pcm);
	Array out;
	while (count_available_contexts() > 0) {
		out.push_back(get_next_context());
	}
	return out;
}

Dictionary MelFrontend::build_utterance_mels(const PackedFloat32Array &pcm) const
{
	std::lock_guard<std::mutex> lock(stream_mu);
	Dictionary out;
	PackedFloat32Array frames;
	out["frames"] = frames;
	out["n_frames"] = 0;
	out["n_mels"] = 0;
	if (!configured || pcm.is_empty() || n_mels <= 0 || cfg.hop_length_samples <= 0) {
		return out;
	}

	const size_t max_frames = (size_t)pcm.size() / (size_t)cfg.hop_length_samples + 16;
	std::vector<float> mel(max_frames * (size_t)n_mels, 0.f);
	size_t frame_count = 0;
	if (mel_spectrogram_process(pcm.ptr(), (size_t)pcm.size(), mel.data(), &frame_count) < 0 ||
			frame_count == 0 || frame_count > max_frames) {
		return out;
	}

	// Match scripts/build_train_tensors.py: normalize each utterance per mel bin.
	for (int m = 0; m < n_mels; m++) {
		double sum = 0.0;
		double sum2 = 0.0;
		for (size_t t = 0; t < frame_count; t++) {
			const double v = (double)mel[t * (size_t)n_mels + (size_t)m];
			sum += v;
			sum2 += v * v;
		}
		const double mean = sum / (double)frame_count;
		double variance = sum2 / (double)frame_count - mean * mean;
		if (variance < 0.0) {
			variance = 0.0;
		}
		const double scale = std::sqrt(variance) + 1e-5;
		for (size_t t = 0; t < frame_count; t++) {
			const size_t idx = t * (size_t)n_mels + (size_t)m;
			mel[idx] = (float)(((double)mel[idx] - mean) / scale);
		}
	}

	frames.resize((int)(frame_count * (size_t)n_mels));
	std::memcpy(frames.ptrw(), mel.data(), frame_count * (size_t)n_mels * sizeof(float));
	out["frames"] = frames;
	out["n_frames"] = (int)frame_count;
	out["n_mels"] = n_mels;
	return out;
}

int MelFrontend::get_input_features() const
{
	return configured ? input_features : 0;
}

int MelFrontend::get_context_frames() const
{
	return configured ? context_frames : 0;
}

int MelFrontend::get_n_mels() const
{
	return configured ? n_mels : 0;
}

int MelFrontend::get_sample_rate() const
{
	return configured ? cfg.sample_rate : 0;
}

int MelFrontend::get_hop_length_samples() const
{
	return configured ? cfg.hop_length_samples : 0;
}

int MelFrontend::get_resampler_input_latency() const
{
	return resampler ? (int)speex_resampler_get_input_latency(resampler) : 0;
}

int MelFrontend::get_resampler_output_latency() const
{
	return resampler ? (int)speex_resampler_get_output_latency(resampler) : 0;
}

int MelFrontend::get_preprocess_frame_size() const
{
	return preprocess.is_enabled() ? preprocess.get_frame_size() : 0;
}

int MelFrontend::get_aec_frame_size() const
{
	return aec.is_enabled() ? aec.get_frame_size() : 0;
}

int MelFrontend::get_dsp_latency_samples() const
{
	if (!configured) {
		return 0;
	}
	/* Model-rate samples of fixed Speex delay before mel sees audio. */
	int n = get_resampler_output_latency();
	n += get_aec_frame_size();
	n += get_preprocess_frame_size();
	return n;
}

float MelFrontend::get_dsp_latency_seconds() const
{
	if (!configured || cfg.sample_rate <= 0) {
		return 0.f;
	}
	return (float)get_dsp_latency_samples() / (float)cfg.sample_rate;
}
