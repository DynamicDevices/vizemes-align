#pragma once

#include "feature_frontend.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>

#include <mutex>
#include <vector>

struct SpeexResamplerState_;
typedef struct SpeexResamplerState_ SpeexResamplerState;

using namespace godot;

/** PCM → flattened mel context tensor (for OnnxLoader.predict). No ONNX here. */
class MelFrontend : public RefCounted {
	GDCLASS(MelFrontend, RefCounted)

	const VizemesFrontendOps *ops = nullptr;
	VizemesFrontendConfig cfg{};
	int context_frames = 0;
	int n_mels = 0;
	int input_features = 0;
	int n_visemes = 0;
	bool configured = false;

	float *stream_pcm = nullptr;
	size_t stream_pcm_n = 0;
	size_t stream_pcm_cap = 0;
	size_t stream_contexts_emitted = 0;
	std::vector<PackedFloat32Array> context_queue;
	mutable std::mutex stream_mu;

	/** SpeexDSP resampler for mic mix_rate → model sample_rate (stateful). */
	SpeexResamplerState *resampler = nullptr;
	int resampler_in_rate = 0;

	void clear_stream();
	void destroy_resampler();
	bool ensure_resampler(int from_rate);
	bool apply_config();
	Array contexts_from_pcm(const float *pcm, size_t n_samples, size_t skip_contexts) const;
	void enqueue_new_contexts(const Array &fresh);
	PackedFloat32Array resample_mono(const float *mono, size_t n, int from_rate);

protected:
	static void _bind_methods();

public:
	MelFrontend();
	~MelFrontend() override;

	/**
	 * Configure from GDScript-owned params (parse model.json with FileAccess + JSON.parse_string).
	 * input_features is derived as context_frames * n_mels when <= 0.
	 */
	bool configure(int p_context_frames, int p_n_mels, int p_sample_rate, int p_hop_length_samples,
			int p_window_length_samples, int p_n_fft, float p_fmin, float p_fmax,
			int p_n_visemes = 15, int p_input_features = 0);

	void reset();
	/** Clear streaming PCM + context queue (keep configure). */
	void begin_stream();

	/** Append mono PCM already at configured sample_rate; produce contexts into the queue. */
	void push_pcm(const PackedFloat32Array &pcm);
	/** Mic path: stereo frames at mix_rate → mono + resample → push_pcm. */
	void push_pcm_stereo(const PackedVector2Array &frames, int mix_rate);

	int count_available_contexts() const;
	PackedFloat32Array get_next_context();
	/**
	 * Seconds from the newest PCM sample back to the next queued context
	 * (or last emitted if queue empty). Includes unfetched contexts + leftover PCM.
	 */
	float last_context_time_offset() const;

	/** Convenience: push whole utterance and return all contexts (drains via stream queue). */
	Array build_utterance_contexts(const PackedFloat32Array &pcm);

	int get_input_features() const;
	int get_context_frames() const;
	int get_n_mels() const;
	int get_sample_rate() const;
	int get_hop_length_samples() const;
};
