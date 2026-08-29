/**
 * Shared C runtime: frontend features + ONNX → viseme weights.
 *
 * Godot side (goatchurchprime/lipsync shape): feed float weights into
 * VisemeSystem.set_visemes(vv). OVR/ReadyPlayerMe order is typically:
 *   sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, I/ih, O/oh, U/ou [, LA]
 * Smoke MLP outputs 15 classes (no LA); map names via model.meta viseme_N= lines.
 *
 * GDExtension binding comes next; this header is the non-Godot SoT.
 */
#ifndef VIZEMES_VISEME_RUNTIME_H
#define VIZEMES_VISEME_RUNTIME_H

#include "feature_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VizemesRuntime VizemesRuntime;

/**
 * Load flat model.meta (from emit_model_meta.py) + ONNX.
 * frontend may be vizemes_frontend_mel() or later LPC.
 * Returns NULL on failure.
 */
VizemesRuntime *vizemes_runtime_create(const char *model_meta_path,
				       const char *model_onnx_path,
				       const VizemesFrontendOps *frontend);

void vizemes_runtime_destroy(VizemesRuntime *rt);

/** Push mono PCM @ model sample_rate; emit up to max_frames of n_visemes softmax weights. */
int vizemes_runtime_push_pcm(VizemesRuntime *rt, const float *pcm, int n_samples,
			     float *viseme_out, int max_frames, int *n_visemes_out);

/**
 * Run one precomputed flat mel context (length = input_features from model.meta).
 * Writes n_visemes softmax weights. Used by host smoke (skips PCM→mel).
 */
int vizemes_runtime_run_context(VizemesRuntime *rt, const float *flat_ctx, float *viseme_out);

int vizemes_runtime_n_visemes(const VizemesRuntime *rt);
int vizemes_runtime_input_features(const VizemesRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif /* VIZEMES_VISEME_RUNTIME_H */
