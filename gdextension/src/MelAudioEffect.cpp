#include "MelAudioEffect.hpp"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

MelAudioEffect::MelAudioEffect()
{
	mel.instantiate();
}

void MelAudioEffect::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_mel_frontend", "mel"), &MelAudioEffect::set_mel_frontend);
	ClassDB::bind_method(D_METHOD("get_mel_frontend"), &MelAudioEffect::get_mel_frontend);
	ClassDB::bind_method(D_METHOD("count_available_contexts"), &MelAudioEffect::count_available_contexts);
	ClassDB::bind_method(D_METHOD("get_next_context"), &MelAudioEffect::get_next_context);
	ClassDB::bind_method(D_METHOD("last_context_time_offset"), &MelAudioEffect::last_context_time_offset);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mel_frontend", PROPERTY_HINT_RESOURCE_TYPE, "MelFrontend"),
			"set_mel_frontend", "get_mel_frontend");
}

void MelAudioEffect::set_mel_frontend(const Ref<MelFrontend> &p_mel)
{
	mel = p_mel;
	if (mel.is_null()) {
		mel.instantiate();
	}
}

Ref<MelFrontend> MelAudioEffect::get_mel_frontend() const
{
	return mel;
}

int MelAudioEffect::count_available_contexts() const
{
	return mel.is_valid() ? mel->count_available_contexts() : 0;
}

PackedFloat32Array MelAudioEffect::get_next_context()
{
	return mel.is_valid() ? mel->get_next_context() : PackedFloat32Array();
}

float MelAudioEffect::last_context_time_offset() const
{
	return mel.is_valid() ? mel->last_context_time_offset() : 0.f;
}

Ref<AudioEffectInstance> MelAudioEffect::_instantiate()
{
	Ref<MelAudioEffectInstance> inst;
	inst.instantiate();
	inst->set_mel(mel);
	return inst;
}
