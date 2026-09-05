/** Stateless 25 ms PCM to speech source/filter measurements. */
#ifndef VIZEMES_SOURCE_FILTER_H
#define VIZEMES_SOURCE_FILTER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VIZEMES_SOURCE_FILTER_SAMPLE_RATE 16000
#define VIZEMES_SOURCE_FILTER_SAMPLES 400
#define VIZEMES_SOURCE_FILTER_LPC_ORDER 16
#define VIZEMES_SOURCE_FILTER_ENVELOPE_BINS 80

typedef struct VizemesSourceFilterFrame {
	/* Stable, bounded representation of the vocal-tract all-pole filter. */
	float reflection[VIZEMES_SOURCE_FILTER_LPC_ORDER];
	/* A dense view of that filter for the previewer, normalized to 0 dB peak. */
	float envelope_db[VIZEMES_SOURCE_FILTER_ENVELOPE_BINS];

	float rms_dbfs;
	float prediction_gain_db;
	float periodicity;
	float pitch_hz;
	float pitch_confidence;
	float hnr_db;
	float residual_tilt_db_octave;
	int pitch_valid;
} VizemesSourceFilterFrame;

/**
 * Analyze one mono 16 kHz PCM window. `sample_count` must be exactly 400.
 * The window represents 25 ms and is normally advanced by a separate 160
 * sample (10 ms) caller. Returns 0 on success and -1 for invalid arguments.
 */
int vizemes_source_filter_process(const float *pcm, size_t sample_count,
		VizemesSourceFilterFrame *out);

#ifdef __cplusplus
}
#endif

#endif
