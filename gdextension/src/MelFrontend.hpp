#pragma once

#include "feature_frontend.h"
#include "MelSpeexPreprocess.hpp"
#include "MelSpeexEcho.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>

#include <mutex>
#include <deque>
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

	/** Causal live path: retain only one analysis window and the current Mel context. */
	std::deque<float> stream_pcm;
	size_t stream_pcm_n = 0;
	size_t stream_contexts_emitted = 0;
	std::deque<PackedFloat32Array> mel_ring;
	std::vector<double> running_mel_sum;
	std::vector<double> running_mel_sum2;
	size_t running_mel_frames = 0;
	std::deque<PackedFloat32Array> context_queue;
	mutable std::mutex stream_mu;

	/** SpeexDSP resampler for mic mix_rate → model sample_rate (stateful). */
	SpeexResamplerState *resampler = nullptr;
	int resampler_in_rate = 0;

	/** SpeexDSP preprocess (AGC/VAD/denoise) on model-rate mono before mel. */
	MelSpeexPreprocess preprocess;
	/** SpeexDSP AEC before preprocess; far-end via push_far_end_*. */
	MelSpeexEcho aec;

	void clear_stream();
	void destroy_resampler();
	bool ensure_resampler(int from_rate);
	bool apply_config();
	PackedFloat32Array resample_mono(const float *mono, size_t n, int from_rate);
	void append_stream_pcm(const float *pcm, size_t n);

protected:
	static void _bind_methods();

public:
	MelFrontend();
	~MelFrontend() override;

	/**
	 * Configure from GDScript-owned params read from canonical ONNX metadata.
	 * input_features is derived as context_frames * n_mels when <= 0.
	 */
	bool configure(int p_context_frames, int p_n_mels, int p_sample_rate, int p_hop_length_samples,
			int p_window_length_samples, int p_n_fft, float p_fmin, float p_fmax,
			int p_n_visemes = 15, int p_input_features = 0);

	/**
	 * SpeexDSP preprocess on mic/model-rate mono (after resample).
	 * gate_on_vad: drop non-speech frames from the stream (silence → no mel contexts).
	 */
	bool configure_preprocess(bool agc = true, bool vad = true, bool denoise = true,
			float agc_level = 8000.f, bool gate_on_vad = false, int frame_size_ms = 10);
	void disable_preprocess();
	bool get_last_vad() const;
	bool is_preprocess_enabled() const;

	/**
	 * SpeexDSP AEC on model-rate mono (before preprocess).
	 * Push far-end (speaker) with push_far_end_pcm / push_far_end_stereo.
	 */
	bool configure_aec(int filter_ms = 100, int frame_size_ms = 10);
	void disable_aec();
	bool is_aec_enabled() const;
	void push_far_end_pcm(const PackedFloat32Array &pcm);
	void push_far_end_stereo(const PackedVector2Array &frames, int mix_rate);

	void reset();
	/** Clear streaming PCM + context queue (keep configure). */
	void begin_stream();

	/** Append mono PCM already at configured sample_rate; produce contexts into the queue. */
	void push_pcm(const PackedFloat32Array &pcm);
	/** Mic path: stereo frames at mix_rate → mono + resample (+ preprocess) → push_pcm. */
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
	/** Full normalized [time, mel] matrix for dynamic-time models such as TCN. */
	Dictionary build_utterance_mels(const PackedFloat32Array &pcm) const;

	int get_input_features() const;
	int get_context_frames() const;
	int get_n_mels() const;
	int get_sample_rate() const;
	int get_hop_length_samples() const;

	/**
	 * Speex resampler latencies (model-rate / mix-rate samples). 0 if no resampler.
	 * Use with get_dsp_latency_samples() for viseme↔playback realignment.
	 */
	int get_resampler_input_latency() const;
	int get_resampler_output_latency() const;
	int get_preprocess_frame_size() const;
	int get_aec_frame_size() const;
	/**
	 * Fixed DSP delay at model sample_rate before mel contexts: Speex resampler
	 * output latency + one AEC frame (if on) + one preprocess frame (if on).
	 * Does not include mel context_frames / hop (see last_context_time_offset).
	 */
	int get_dsp_latency_samples() const;
	/** get_dsp_latency_samples() / sample_rate (0 if unconfigured). */
	float get_dsp_latency_seconds() const;
};
