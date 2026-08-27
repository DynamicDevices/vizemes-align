#include "MelFrontend.hpp"

#include <godot_cpp/core/class_db.hpp>

#include "mel_spectrogram.h"

#include <cmath>
#include <cstring>
#include <cstdlib>

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
}

void MelFrontend::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("configure_from_json", "model_json_path"),
			&MelFrontend::configure_from_json);
	ClassDB::bind_method(D_METHOD("reset"), &MelFrontend::reset);
	ClassDB::bind_method(D_METHOD("begin_stream"), &MelFrontend::begin_stream);
	ClassDB::bind_method(D_METHOD("push_pcm", "pcm"), &MelFrontend::push_pcm);
	ClassDB::bind_method(D_METHOD("push_pcm_contexts", "pcm"), &MelFrontend::push_pcm_contexts);
	ClassDB::bind_method(D_METHOD("build_utterance_contexts", "pcm"), &MelFrontend::build_utterance_contexts);
	ClassDB::bind_method(D_METHOD("get_input_features"), &MelFrontend::get_input_features);
	ClassDB::bind_method(D_METHOD("get_context_frames"), &MelFrontend::get_context_frames);
	ClassDB::bind_method(D_METHOD("get_n_mels"), &MelFrontend::get_n_mels);
}

bool MelFrontend::configure_from_json(const String &model_json_path)
{
	reset();
	if (!ops) {
		return false;
	}

	CharString path = model_json_path.utf8();
	if (vizemes_sidecar_load(path.get_data(), &meta) != 0) {
		return false;
	}

	VizemesFrontendConfig cfg{};
	cfg.sample_rate = meta.sample_rate;
	cfg.hop_length_samples = meta.hop_length_samples;
	cfg.window_length_samples = meta.window_length_samples;
	cfg.n_features = meta.n_mels;
	cfg.n_fft = meta.n_fft;
	cfg.fmin = meta.fmin;
	cfg.fmax = meta.fmax;
	cfg.top_db = 80.f;
	if (ops->init && ops->init(&cfg) != 0) {
		return false;
	}

	configured = true;
	return true;
}

void MelFrontend::reset()
{
	if (ops && ops->free) {
		ops->free();
	}
	clear_stream();
	meta = {};
	configured = false;
}

void MelFrontend::begin_stream()
{
	if (!configured) {
		return;
	}
	clear_stream();
}

Array MelFrontend::contexts_from_pcm(const float *pcm, size_t n_samples, size_t skip_contexts) const
{
	Array out;
	if (!configured || !pcm || n_samples == 0) {
		return out;
	}

	const int nm = meta.n_mels;
	const int ctx = meta.context_frames;

	size_t max_frames = n_samples / (size_t)meta.hop_length_samples + 16;
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
		flat.resize(meta.input_features);
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

PackedFloat32Array MelFrontend::push_pcm(const PackedFloat32Array &pcm)
{
	Array contexts = push_pcm_contexts(pcm);
	if (contexts.is_empty()) {
		return PackedFloat32Array();
	}
	return contexts[contexts.size() - 1];
}

Array MelFrontend::push_pcm_contexts(const PackedFloat32Array &pcm)
{
	Array out;
	if (!configured) {
		return out;
	}

	const int n_in = (int)pcm.size();
	if (n_in <= 0) {
		return out;
	}

	if (stream_pcm_n + (size_t)n_in > stream_pcm_cap) {
		size_t need = stream_pcm_n + (size_t)n_in;
		float *nb = (float *)realloc(stream_pcm, need * sizeof(float));
		if (!nb) {
			return out;
		}
		stream_pcm = nb;
		stream_pcm_cap = need;
	}
	memcpy(stream_pcm + stream_pcm_n, pcm.ptr(), (size_t)n_in * sizeof(float));
	stream_pcm_n += (size_t)n_in;

	out = contexts_from_pcm(stream_pcm, stream_pcm_n, stream_contexts_emitted);
	stream_contexts_emitted += (size_t)out.size();
	return out;
}

Array MelFrontend::build_utterance_contexts(const PackedFloat32Array &pcm)
{
	return contexts_from_pcm(pcm.ptr(), (size_t)pcm.size(), 0);
}

int MelFrontend::get_input_features() const
{
	return configured ? meta.input_features : 0;
}

int MelFrontend::get_context_frames() const
{
	return configured ? meta.context_frames : 0;
}

int MelFrontend::get_n_mels() const
{
	return configured ? meta.n_mels : 0;
}
