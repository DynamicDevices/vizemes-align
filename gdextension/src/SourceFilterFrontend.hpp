#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

using namespace godot;

/** Thin Godot binding for the stateless 16 kHz source/filter C extractor. */
class SourceFilterFrontend : public RefCounted {
	GDCLASS(SourceFilterFrontend, RefCounted)

	Dictionary frame_dictionary(const float *pcm) const;

protected:
	static void _bind_methods();

public:
	Dictionary analyze_frame(const PackedFloat32Array &pcm) const;
	Dictionary analyze_utterance(const PackedFloat32Array &pcm) const;
	int get_sample_rate() const;
	int get_window_samples() const;
	int get_hop_samples() const;
	int get_lpc_order() const;
};
