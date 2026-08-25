/**
 * Host smoke: load smoke ONNX + one flat mel-context row → print argmax + weights.
 *
 *   make ORT_ROOT=/path/to/onnxruntime smoke
 */
#include "feature_frontend.h"
#include "viseme_runtime.h"

#include <stdio.h>
#include <stdlib.h>

static int load_f32(const char *path, float **out, int expect_n)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		perror(path);
		return -1;
	}
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return -1;
	}
	long bytes = ftell(f);
	rewind(f);
	if (bytes <= 0 || (bytes % (long)sizeof(float)) != 0) {
		fclose(f);
		return -1;
	}
	int n = (int)(bytes / (long)sizeof(float));
	if (expect_n > 0 && n != expect_n) {
		fprintf(stderr, "BAD %s: got %d floats, expected %d\n", path, n, expect_n);
		fclose(f);
		return -1;
	}
	float *buf = (float *)malloc((size_t)n * sizeof(float));
	if (!buf || fread(buf, sizeof(float), (size_t)n, f) != (size_t)n) {
		free(buf);
		fclose(f);
		return -1;
	}
	fclose(f);
	*out = buf;
	return n;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr,
			"usage: %s model.json model.onnx context.f32 [expect_class_id]\n",
			argv[0]);
		return 2;
	}
	const char *json_path = argv[1];
	const char *onnx_path = argv[2];
	const char *f32_path = argv[3];
	int expect = argc >= 5 ? atoi(argv[4]) : -1;

	VizemesRuntime *rt =
		vizemes_runtime_create(json_path, onnx_path, vizemes_frontend_mel());
	if (!rt) {
		fprintf(stderr, "vizemes_runtime_create failed\n");
		return 1;
	}

	int nfeat = vizemes_runtime_input_features(rt);
	int nv = vizemes_runtime_n_visemes(rt);
	float *ctx = NULL;
	if (load_f32(f32_path, &ctx, nfeat) < 0) {
		vizemes_runtime_destroy(rt);
		return 1;
	}

	float *w = (float *)calloc((size_t)nv, sizeof(float));
	if (!w || vizemes_runtime_run_context(rt, ctx, w) != 0) {
		fprintf(stderr, "run_context failed\n");
		free(w);
		free(ctx);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	int argmax = 0;
	for (int i = 1; i < nv; i++) {
		if (w[i] > w[argmax]) {
			argmax = i;
		}
	}

	printf("GDEXT_ORT_SMOKE_OK n_visemes=%d input_features=%d argmax=%d weight=%.4f\n", nv,
	       nfeat, argmax, w[argmax]);
	for (int i = 0; i < nv; i++) {
		printf("  [%2d] %.4f%s\n", i, w[i], i == argmax ? " *" : "");
	}

	int rc = 0;
	if (expect >= 0 && argmax != expect) {
		fprintf(stderr, "expect class %d got %d\n", expect, argmax);
		rc = 1;
	}

	free(w);
	free(ctx);
	vizemes_runtime_destroy(rt);
	return rc;
}
