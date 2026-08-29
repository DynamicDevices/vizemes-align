#include "MelSpeexEcho.hpp"

#include "speex/speex_echo.h"

#include <algorithm>
#include <cmath>
#include <cstring>

MelSpeexEcho::~MelSpeexEcho()
{
	destroy();
}

void MelSpeexEcho::destroy()
{
	if (st) {
		speex_echo_state_destroy(st);
		st = nullptr;
	}
	rec_pending.clear();
	far_pending.clear();
	enabled = false;
	frame_size = 0;
	filter_length = 0;
	sample_rate = 0;
}

void MelSpeexEcho::clear_pending()
{
	rec_pending.clear();
	far_pending.clear();
	if (st) {
		speex_echo_state_reset(st);
	}
}

bool MelSpeexEcho::setup(int p_sample_rate, int frame_size_ms, int filter_ms)
{
	destroy();
	if (p_sample_rate <= 0 || frame_size_ms <= 0 || filter_ms <= 0) {
		return false;
	}
	frame_size = (p_sample_rate * frame_size_ms) / 1000;
	if (frame_size < 80) {
		frame_size = 80;
	}
	filter_length = (p_sample_rate * filter_ms) / 1000;
	if (filter_length < frame_size * 2) {
		filter_length = frame_size * 2;
	}
	sample_rate = p_sample_rate;
	st = speex_echo_state_init(frame_size, filter_length);
	if (!st) {
		return false;
	}
	int rate = sample_rate;
	speex_echo_ctl(st, SPEEX_ECHO_SET_SAMPLING_RATE, &rate);
	enabled = true;
	rec_pending.reserve((size_t)frame_size * 2);
	return true;
}

void MelSpeexEcho::push_far_end(const float *in, size_t n)
{
	if (!enabled || !in || n == 0) {
		return;
	}
	for (size_t i = 0; i < n; i++) {
		far_pending.push_back(in[i]);
	}
}

void MelSpeexEcho::process_and_append(const float *rec, size_t n, std::vector<float> &out)
{
	if (!enabled || !st || !rec || n == 0) {
		if (rec && n > 0) {
			out.insert(out.end(), rec, rec + n);
		}
		return;
	}
	rec_pending.insert(rec_pending.end(), rec, rec + n);
	std::vector<spx_int16_t> rec_i((size_t)frame_size);
	std::vector<spx_int16_t> play_i((size_t)frame_size);
	std::vector<spx_int16_t> out_i((size_t)frame_size);
	while ((int)rec_pending.size() >= frame_size) {
		for (int i = 0; i < frame_size; i++) {
			float s = rec_pending[(size_t)i];
			s = std::max(-1.f, std::min(1.f, s));
			rec_i[(size_t)i] = (spx_int16_t)std::lround(s * 32767.f);
			float p = 0.f;
			if (!far_pending.empty()) {
				p = far_pending.front();
				far_pending.pop_front();
			}
			p = std::max(-1.f, std::min(1.f, p));
			play_i[(size_t)i] = (spx_int16_t)std::lround(p * 32767.f);
		}
		rec_pending.erase(rec_pending.begin(), rec_pending.begin() + frame_size);
		speex_echo_cancellation(st, rec_i.data(), play_i.data(), out_i.data());
		for (int i = 0; i < frame_size; i++) {
			out.push_back((float)out_i[(size_t)i] / 32768.f);
		}
	}
}
