#pragma once

#include "MelFrontend.hpp"

#include <godot_cpp/classes/audio_effect.hpp>
#include <godot_cpp/classes/audio_effect_instance.hpp>
#include <godot_cpp/classes/audio_frame.hpp>
#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

using namespace godot;

/** Bus effect: passthrough audio + feed MelFrontend from the mix (like Capture → contexts). */
class MelAudioEffectInstance : public AudioEffectInstance {
	GDCLASS(MelAudioEffectInstance, AudioEffectInstance)

	Ref<MelFrontend> mel;

protected:
	static void _bind_methods() {}

public:
	void set_mel(const Ref<MelFrontend> &p_mel) { mel = p_mel; }

	void _process(const void *p_src_buffer, AudioFrame *p_dst_buffer, int32_t p_frame_count) override
	{
		const AudioFrame *src = static_cast<const AudioFrame *>(p_src_buffer);
		for (int32_t i = 0; i < p_frame_count; i++) {
			p_dst_buffer[i] = src[i];
		}
		if (mel.is_null() || p_frame_count <= 0) {
			return;
		}
		PackedVector2Array frames;
		frames.resize(p_frame_count);
		for (int32_t i = 0; i < p_frame_count; i++) {
			frames[i] = Vector2(src[i].left, src[i].right);
		}
		const int mix = (int)AudioServer::get_singleton()->get_mix_rate();
		mel->push_pcm_stereo(frames, mix > 0 ? mix : 48000);
	}
};

class MelAudioEffect : public AudioEffect {
	GDCLASS(MelAudioEffect, AudioEffect)

	Ref<MelFrontend> mel;

protected:
	static void _bind_methods();

public:
	MelAudioEffect();

	void set_mel_frontend(const Ref<MelFrontend> &p_mel);
	Ref<MelFrontend> get_mel_frontend() const;

	int count_available_contexts() const;
	PackedFloat32Array get_next_context();
	float last_context_time_offset() const;

	Ref<AudioEffectInstance> _instantiate() override;
};
