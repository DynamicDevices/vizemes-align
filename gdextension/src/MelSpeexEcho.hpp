#pragma once

#include <cstddef>
#include <deque>
#include <vector>

struct SpeexEchoState_;
typedef struct SpeexEchoState_ SpeexEchoState;

/** SpeexDSP AEC for MelFrontend mic path. Speex sources unmodified. */
class MelSpeexEcho {
	SpeexEchoState *st = nullptr;
	int frame_size = 0;
	int filter_length = 0;
	int sample_rate = 0;
	bool enabled = false;
	std::vector<float> rec_pending;
	std::deque<float> far_pending;

public:
	~MelSpeexEcho();

	void destroy();
	/**
	 * frame_size_ms typically 10; filter_ms typically 100–500 (echo tail).
	 * Returns false on Speex init failure.
	 */
	bool setup(int p_sample_rate, int frame_size_ms, int filter_ms);

	bool is_enabled() const { return enabled && st != nullptr; }
	int get_frame_size() const { return frame_size; }

	/** Queue far-end (speaker) mono at model sample_rate. */
	void push_far_end(const float *in, size_t n);
	void clear_pending();

	/**
	 * AEC on near-end mic mono @ sample_rate. Appends cleaned samples to out.
	 * Uses queued far-end (zeros when none yet).
	 */
	void process_and_append(const float *rec, size_t n, std::vector<float> &out);
};
