#include "SourceFilterFrontend.hpp"

#include "source_filter.h"

#include <godot_cpp/core/class_db.hpp>

static const int HOP_SAMPLES = 160;

void SourceFilterFrontend::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("analyze_frame", "pcm"), &SourceFilterFrontend::analyze_frame);
	ClassDB::bind_method(D_METHOD("analyze_utterance", "pcm"), &SourceFilterFrontend::analyze_utterance);
	ClassDB::bind_method(D_METHOD("get_sample_rate"), &SourceFilterFrontend::get_sample_rate);
	ClassDB::bind_method(D_METHOD("get_window_samples"), &SourceFilterFrontend::get_window_samples);
	ClassDB::bind_method(D_METHOD("get_hop_samples"), &SourceFilterFrontend::get_hop_samples);
	ClassDB::bind_method(D_METHOD("get_lpc_order"), &SourceFilterFrontend::get_lpc_order);
}

Dictionary SourceFilterFrontend::frame_dictionary(const float *pcm) const
{
	VizemesSourceFilterFrame frame;
	Dictionary result;
	if (vizemes_source_filter_process(pcm, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0) return result;

	PackedFloat32Array reflection;
	reflection.resize(VIZEMES_SOURCE_FILTER_LPC_ORDER);
	for (int i = 0; i < reflection.size(); i++) reflection[i] = frame.reflection[i];
	PackedFloat32Array envelope;
	envelope.resize(VIZEMES_SOURCE_FILTER_ENVELOPE_BINS);
	for (int i = 0; i < envelope.size(); i++) envelope[i] = frame.envelope_db[i];

	result["reflection"] = reflection;
	result["envelope_db"] = envelope;
	result["rms_dbfs"] = frame.rms_dbfs;
	result["prediction_gain_db"] = frame.prediction_gain_db;
	result["periodicity"] = frame.periodicity;
	result["pitch_hz"] = frame.pitch_hz;
	result["pitch_confidence"] = frame.pitch_confidence;
	result["pitch_valid"] = frame.pitch_valid != 0;
	result["hnr_db"] = frame.hnr_db;
	result["residual_tilt_db_octave"] = frame.residual_tilt_db_octave;
	return result;
}

Dictionary SourceFilterFrontend::analyze_frame(const PackedFloat32Array &pcm) const
{
	if (pcm.size() != VIZEMES_SOURCE_FILTER_SAMPLES) return Dictionary();
	return frame_dictionary(pcm.ptr());
}

Dictionary SourceFilterFrontend::analyze_utterance(const PackedFloat32Array &pcm) const
{
	Dictionary result;
	if (pcm.size() < VIZEMES_SOURCE_FILTER_SAMPLES) return result;
	int frames = 1 + (pcm.size() - VIZEMES_SOURCE_FILTER_SAMPLES) / HOP_SAMPLES;
	PackedFloat32Array reflection;
	PackedFloat32Array envelope;
	PackedFloat32Array scalars;
	reflection.resize(frames * VIZEMES_SOURCE_FILTER_LPC_ORDER);
	envelope.resize(frames * VIZEMES_SOURCE_FILTER_ENVELOPE_BINS);
	/* rms, prediction gain, periodicity, pitch, confidence, valid, HNR, tilt */
	scalars.resize(frames * 8);

	for (int index = 0; index < frames; index++) {
		VizemesSourceFilterFrame frame;
		const float *window = pcm.ptr() + index * HOP_SAMPLES;
		if (vizemes_source_filter_process(window, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0) {
			return Dictionary();
		}
		for (int i = 0; i < VIZEMES_SOURCE_FILTER_LPC_ORDER; i++) {
			reflection[index * VIZEMES_SOURCE_FILTER_LPC_ORDER + i] = frame.reflection[i];
		}
		for (int i = 0; i < VIZEMES_SOURCE_FILTER_ENVELOPE_BINS; i++) {
			envelope[index * VIZEMES_SOURCE_FILTER_ENVELOPE_BINS + i] = frame.envelope_db[i];
		}
		int offset = index * 8;
		scalars[offset] = frame.rms_dbfs;
		scalars[offset + 1] = frame.prediction_gain_db;
		scalars[offset + 2] = frame.periodicity;
		scalars[offset + 3] = frame.pitch_hz;
		scalars[offset + 4] = frame.pitch_confidence;
		scalars[offset + 5] = frame.pitch_valid ? 1.0f : 0.0f;
		scalars[offset + 6] = frame.hnr_db;
		scalars[offset + 7] = frame.residual_tilt_db_octave;
	}

	result["frames"] = frames;
	result["frame_center_offset_samples"] = VIZEMES_SOURCE_FILTER_SAMPLES / 2;
	result["reflection"] = reflection;
	result["envelope_db"] = envelope;
	result["scalar_names"] = PackedStringArray({
			"rms_dbfs", "prediction_gain_db", "periodicity", "pitch_hz",
			"pitch_confidence", "pitch_valid", "hnr_db", "residual_tilt_db_octave"});
	result["scalars"] = scalars;
	return result;
}

int SourceFilterFrontend::get_sample_rate() const
{
	return VIZEMES_SOURCE_FILTER_SAMPLE_RATE;
}

int SourceFilterFrontend::get_window_samples() const
{
	return VIZEMES_SOURCE_FILTER_SAMPLES;
}

int SourceFilterFrontend::get_hop_samples() const
{
	return HOP_SAMPLES;
}

int SourceFilterFrontend::get_lpc_order() const
{
	return VIZEMES_SOURCE_FILTER_LPC_ORDER;
}
