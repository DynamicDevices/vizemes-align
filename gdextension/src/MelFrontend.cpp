#include "MelFrontend.hpp"

#include <godot_cpp/core/class_db.hpp>

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

void MelFrontend::clear_buffers()
{
	::free(mel_ring);
	mel_ring = nullptr;
	mel_filled = 0;
	::free(pcm_pending);
	pcm_pending = nullptr;
	pcm_pending_n = 0;
	pcm_pending_cap = 0;
}

void MelFrontend::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("configure_from_json", "model_json_path"),
			&MelFrontend::configure_from_json);
	ClassDB::bind_method(D_METHOD("reset"), &MelFrontend::reset);
	ClassDB::bind_method(D_METHOD("push_pcm", "pcm"), &MelFrontend::push_pcm);
	ClassDB::bind_method(D_METHOD("push_pcm_contexts", "pcm"), &MelFrontend::push_pcm_contexts);
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

	mel_ring = (float *)calloc((size_t)meta.context_frames * (size_t)meta.n_mels, sizeof(float));
	pcm_pending_cap = meta.window_length_samples * 4;
	pcm_pending = (float *)calloc((size_t)pcm_pending_cap, sizeof(float));
	if (!mel_ring || !pcm_pending) {
		reset();
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
	clear_buffers();
	meta = {};
	configured = false;
}

void MelFrontend::ring_push(const float *mel_frame)
{
	const int ctx = meta.context_frames;
	const int nm = meta.n_mels;
	if (mel_filled < ctx) {
		memcpy(mel_ring + (size_t)mel_filled * (size_t)nm, mel_frame, (size_t)nm * sizeof(float));
		mel_filled++;
		return;
	}
	memmove(mel_ring, mel_ring + nm, (size_t)(ctx - 1) * (size_t)nm * sizeof(float));
	memcpy(mel_ring + (size_t)(ctx - 1) * (size_t)nm, mel_frame, (size_t)nm * sizeof(float));
}

PackedFloat32Array MelFrontend::build_flat_context() const
{
	PackedFloat32Array flat;
	if (!configured || mel_filled < meta.context_frames) {
		return flat;
	}
	flat.resize(meta.input_features);
	const int ctx = meta.context_frames;
	const int nm = meta.n_mels;
	for (int i = 0; i < ctx; i++) {
		for (int j = 0; j < nm; j++) {
			flat[i * nm + j] = mel_ring[(size_t)i * (size_t)nm + (size_t)j];
		}
	}
	return flat;
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
	if (!configured || !ops || !ops->process_frame) {
		return out;
	}

	const int win = meta.window_length_samples;
	const int hop = meta.hop_length_samples;
	const int nm = meta.n_mels;
	const int n_in = (int)pcm.size();
	if (n_in <= 0) {
		return out;
	}

	if (pcm_pending_n + n_in > pcm_pending_cap) {
		int need = pcm_pending_n + n_in;
		float *nb = (float *)realloc(pcm_pending, (size_t)need * sizeof(float));
		if (!nb) {
			return out;
		}
		pcm_pending = nb;
		pcm_pending_cap = need;
	}
	memcpy(pcm_pending + pcm_pending_n, pcm.ptr(), (size_t)n_in * sizeof(float));
	pcm_pending_n += n_in;

	float *frame = (float *)malloc((size_t)nm * sizeof(float));
	if (!frame) {
		return out;
	}

	while (pcm_pending_n >= win) {
		if (ops->process_frame(pcm_pending, frame) != 0) {
			::free(frame);
			return out;
		}
		ring_push(frame);
		int drop = hop;
		if (drop > pcm_pending_n) {
			drop = pcm_pending_n;
		}
		memmove(pcm_pending, pcm_pending + drop, (size_t)(pcm_pending_n - drop) * sizeof(float));
		pcm_pending_n -= drop;

		if (mel_filled >= meta.context_frames) {
			out.push_back(build_flat_context());
		}
	}

	::free(frame);
	return out;
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
