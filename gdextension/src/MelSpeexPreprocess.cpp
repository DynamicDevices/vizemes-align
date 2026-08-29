#include "MelSpeexPreprocess.hpp"

#include "speex/speex_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

MelSpeexPreprocess::~MelSpeexPreprocess()
{
	destroy();
}

void MelSpeexPreprocess::destroy()
{
	if (st) {
		speex_preprocess_state_destroy(st);
		st = nullptr;
	}
	pending.clear();
	enabled = false;
	frame_size = 0;
	sample_rate = 0;
	last_vad = false;
}

bool MelSpeexPreprocess::setup(int p_sample_rate, int frame_size_ms, bool agc, bool vad, bool denoise,
		float agc_level, bool p_gate_on_vad)
{
	destroy();
	if (p_sample_rate <= 0 || frame_size_ms <= 0) {
		return false;
	}
	if (!agc && !vad && !denoise) {
		enabled = false;
		return true;
	}
	frame_size = (p_sample_rate * frame_size_ms) / 1000;
	if (frame_size < 80) {
		frame_size = 80;
	}
	sample_rate = p_sample_rate;
	st = speex_preprocess_state_init(frame_size, sample_rate);
	if (!st) {
		return false;
	}
	int on;
	on = denoise ? 1 : 0;
	speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DENOISE, &on);
	on = agc ? 1 : 0;
	speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC, &on);
	on = vad ? 1 : 0;
	speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_VAD, &on);
	if (agc) {
		float lvl = agc_level;
		speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC_LEVEL, &lvl);
	}
	gate_on_vad = p_gate_on_vad && vad;
	enabled = true;
	pending.clear();
	pending.reserve((size_t)frame_size * 2);
	return true;
}

void MelSpeexPreprocess::process_and_append(const float *in, size_t n, std::vector<float> &out)
{
	if (!enabled || !st || !in || n == 0) {
		if (in && n > 0) {
			out.insert(out.end(), in, in + n);
		}
		return;
	}
	pending.insert(pending.end(), in, in + n);
	std::vector<spx_int16_t> frame((size_t)frame_size);
	while ((int)pending.size() >= frame_size) {
		for (int i = 0; i < frame_size; i++) {
			float s = pending[(size_t)i];
			s = std::max(-1.f, std::min(1.f, s));
			frame[(size_t)i] = (spx_int16_t)std::lround(s * 32767.f);
		}
		pending.erase(pending.begin(), pending.begin() + frame_size);
		int vad = speex_preprocess_run(st, frame.data());
		last_vad = vad != 0;
		if (gate_on_vad && !last_vad) {
			continue;
		}
		for (int i = 0; i < frame_size; i++) {
			out.push_back((float)frame[(size_t)i] / 32768.f);
		}
	}
}
