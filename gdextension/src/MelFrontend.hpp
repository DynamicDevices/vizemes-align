#pragma once

#include "feature_frontend.h"
#include "sidecar_json.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

using namespace godot;

/** PCM → flattened mel context tensor (for OnnxLoader.predict). No ONNX here. */
class MelFrontend : public RefCounted {
	GDCLASS(MelFrontend, RefCounted)

	const VizemesFrontendOps *ops = nullptr;
	VizemesModelMeta meta{};
	bool configured = false;
	float *mel_ring = nullptr;
	int mel_filled = 0;
	float *pcm_pending = nullptr;
	int pcm_pending_n = 0;
	int pcm_pending_cap = 0;
	float *norm_sum = nullptr;
	float *norm_sum2 = nullptr;
	int norm_count = 0;

	void clear_buffers();
	void clear_norm_stats();
	void normalize_frame_causal(float *mel_frame);
	void ring_push(const float *mel_frame);
	PackedFloat32Array build_flat_context() const;

protected:
	static void _bind_methods();

public:
	MelFrontend();
	~MelFrontend() override;

	bool configure_from_json(const String &model_json_path);
	void reset();

	PackedFloat32Array push_pcm(const PackedFloat32Array &pcm);
	/** Streaming chunks (causal normalize; mel padding differs from batch train path). */
	Array push_pcm_contexts(const PackedFloat32Array &pcm);
	/** Full utterance: batch mel + per-utterance normalize (matches train path). */
	Array build_utterance_contexts(const PackedFloat32Array &pcm);

	int get_input_features() const;
	int get_context_frames() const;
	int get_n_mels() const;
};
