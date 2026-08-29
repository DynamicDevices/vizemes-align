#pragma once

#include <cstddef>
#include <vector>

struct SpeexPreprocessState_;
typedef struct SpeexPreprocessState_ SpeexPreprocessState;

/** SpeexDSP preprocess (AGC/VAD/denoise) for MelFrontend mic path. Speex sources unmodified. */
class MelSpeexPreprocess {
	SpeexPreprocessState *st = nullptr;
	int frame_size = 0;
	int sample_rate = 0;
	bool enabled = false;
	bool gate_on_vad = false;
	bool last_vad = false;
	std::vector<float> pending;

public:
	~MelSpeexPreprocess();

	void destroy();
	/** frame_size_ms typically 10. Returns false on Speex init failure. */
	bool setup(int p_sample_rate, int frame_size_ms, bool agc, bool vad, bool denoise,
			float agc_level, bool p_gate_on_vad);

	bool is_enabled() const { return enabled && st != nullptr; }
	bool get_last_vad() const { return last_vad; }
	bool get_gate_on_vad() const { return gate_on_vad; }
	int get_frame_size() const { return frame_size; }

	/**
	 * Process mono float @ sample_rate. Appends processed samples to out
	 * (skips frames when gate_on_vad && !speech).
	 */
	void process_and_append(const float *in, size_t n, std::vector<float> &out);
	void clear_pending() { pending.clear(); }
};
