/**
 * Pluggable waveform → feature frontend for viseme inference.
 *
 * Mel (vendored OpenLipSync C) is the first implementation, not a lock-in.
 * LPC or other frontends should implement the same ops so training and Godot
 * stay on one shared C path per experiment.
 */
#ifndef VIZEMES_FEATURE_FRONTEND_H
#define VIZEMES_FEATURE_FRONTEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VizemesFrontendConfig {
	int sample_rate;
	int hop_length_samples;
	int window_length_samples;
	int n_features; /* e.g. n_mels, or LPC order+1 */
	/* Mel-specific (ignored by other frontends): */
	int n_fft;
	float fmin;
	float fmax;
	float top_db;
} VizemesFrontendConfig;

typedef struct VizemesFrontendOps {
	const char *name;
	int (*init)(const VizemesFrontendConfig *cfg);
	void (*free)(void);
	/* One hop frame → n_features floats. Returns 0 ok, -1 error. */
	int (*process_frame)(const float *audio_window, float *feat_out);
	/* Full waveform → (num_frames * n_features) row-major. Returns frames or -1. */
	int (*process_waveform)(const float *samples, int num_samples, float *feat_out,
				 int max_frames);
} VizemesFrontendOps;

/** First implementation: log-mel (see frontends/mel/). */
const VizemesFrontendOps *vizemes_frontend_mel(void);

#ifdef __cplusplus
}
#endif

#endif /* VIZEMES_FEATURE_FRONTEND_H */
