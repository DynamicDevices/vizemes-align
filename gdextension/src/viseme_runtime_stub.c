/**
 * Stub runtime — API shape only until ONNX Runtime is linked (make without ORT_ROOT).
 */
#include "viseme_runtime.h"

#include <stddef.h>
#include <stdlib.h>

struct VizemesRuntime {
	const VizemesFrontendOps *frontend;
};

VizemesRuntime *vizemes_runtime_create(const char *model_json_path,
				       const char *model_onnx_path,
				       const VizemesFrontendOps *frontend)
{
	(void)model_json_path;
	(void)model_onnx_path;
	if (!frontend) {
		return NULL;
	}
	VizemesRuntime *rt = (VizemesRuntime *)calloc(1, sizeof(*rt));
	if (!rt) {
		return NULL;
	}
	rt->frontend = frontend;
	return rt;
}

void vizemes_runtime_destroy(VizemesRuntime *rt)
{
	if (!rt) {
		return;
	}
	if (rt->frontend && rt->frontend->free) {
		rt->frontend->free();
	}
	free(rt);
}

int vizemes_runtime_push_pcm(VizemesRuntime *rt, const float *pcm, int n_samples,
			     float *viseme_out, int max_frames, int *n_visemes_out)
{
	(void)rt;
	(void)pcm;
	(void)n_samples;
	(void)viseme_out;
	(void)max_frames;
	if (n_visemes_out) {
		*n_visemes_out = 0;
	}
	return -1;
}

int vizemes_runtime_run_context(VizemesRuntime *rt, const float *flat_ctx, float *viseme_out)
{
	(void)rt;
	(void)flat_ctx;
	(void)viseme_out;
	return -1;
}

int vizemes_runtime_n_visemes(const VizemesRuntime *rt)
{
	(void)rt;
	return 0;
}

int vizemes_runtime_input_features(const VizemesRuntime *rt)
{
	(void)rt;
	return 0;
}
