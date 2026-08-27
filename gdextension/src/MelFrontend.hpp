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
	float *stream_pcm = nullptr;
	size_t stream_pcm_n = 0;
	size_t stream_pcm_cap = 0;
	size_t stream_contexts_emitted = 0;

	void clear_stream();
	Array contexts_from_pcm(const float *pcm, size_t n_samples, size_t skip_contexts) const;

protected:
	static void _bind_methods();

public:
	MelFrontend();
	~MelFrontend() override;

	bool configure_from_json(const String &model_json_path);
	void reset();
	/** Clear streaming PCM buffer (keep configure/model). */
	void begin_stream();

	PackedFloat32Array push_pcm(const PackedFloat32Array &pcm);
	/** Append PCM; emit new contexts via batch mel (matches train path). */
	Array push_pcm_contexts(const PackedFloat32Array &pcm);
	/** One-shot full utterance (same mel path as push_pcm over whole buffer). */
	Array build_utterance_contexts(const PackedFloat32Array &pcm);

	int get_input_features() const;
	int get_context_frames() const;
	int get_n_mels() const;
};
