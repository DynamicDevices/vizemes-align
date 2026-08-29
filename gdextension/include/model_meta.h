/**
 * Flat model.meta — emitted from model.json by tools/emit_model_meta.py (stdlib json).
 * C only reads key=value lines (no JSON parser in-tree).
 */
#ifndef VIZEMES_MODEL_META_H
#define VIZEMES_MODEL_META_H

#ifdef __cplusplus
extern "C" {
#endif

#define VIZEMES_META_MAX_VISEMES 32

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
	char viseme_names[VIZEMES_META_MAX_VISEMES][32];
} VizemesModelMeta;

/** Load flat .meta (key=value). Returns 0 ok, -1 on I/O or missing required keys. */
int vizemes_meta_load(const char *path, VizemesModelMeta *out);

#ifdef __cplusplus
}
#endif

#endif /* VIZEMES_MODEL_META_H */
