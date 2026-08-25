/**
 * Tiny sidecar reader for export/.../model.json (no cJSON dependency).
 * Enough fields for smoke MLP + mel frontend init.
 */
#ifndef VIZEMES_SIDECAR_JSON_H
#define VIZEMES_SIDECAR_JSON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VizemesModelMeta {
	int context_frames;
	int n_mels;
	int input_features;
	int n_visemes;
	int sample_rate;
	int hop_length_samples;
	int window_length_samples;
	int n_fft;
	float fmin;
	float fmax;
} VizemesModelMeta;

/** Parse model.json. Returns 0 ok, -1 on I/O or missing required keys. */
int vizemes_sidecar_load(const char *path, VizemesModelMeta *out);

#ifdef __cplusplus
}
#endif

#endif /* VIZEMES_SIDECAR_JSON_H */
