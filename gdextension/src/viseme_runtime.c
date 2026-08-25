/**
 * ONNX Runtime–backed viseme runtime (C API).
 * Build with ORT_ROOT pointing at an onnxruntime prefix (nix or release tarball).
 */
#include "viseme_runtime.h"
#include "sidecar_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

struct VizemesRuntime {
	const VizemesFrontendOps *frontend;
	VizemesModelMeta meta;
	const OrtApi *ort;
	OrtEnv *env;
	OrtSessionOptions *opts;
	OrtSession *session;
	OrtAllocator *allocator;
	char *input_name;
	char *output_name;
	float *mel_ring; /* context_frames * n_mels, oldest→newest */
	int mel_filled;  /* frames currently valid in ring (≤ context) */
	float *pcm_pending;
	int pcm_pending_n;
	int pcm_pending_cap;
};

static void softmax(const float *logits, float *out, int n)
{
	float m = logits[0];
	for (int i = 1; i < n; i++) {
		if (logits[i] > m) {
			m = logits[i];
		}
	}
	float sum = 0.f;
	for (int i = 0; i < n; i++) {
		out[i] = expf(logits[i] - m);
		sum += out[i];
	}
	if (sum <= 0.f) {
		sum = 1.f;
	}
	for (int i = 0; i < n; i++) {
		out[i] /= sum;
	}
}

static int ort_ok(const OrtApi *ort, OrtStatus *st, const char *what)
{
	if (!st) {
		return 0;
	}
	const char *msg = ort->GetErrorMessage(st);
	fprintf(stderr, "ORT %s: %s\n", what, msg ? msg : "(null)");
	ort->ReleaseStatus(st);
	return -1;
}

static int run_flat(VizemesRuntime *rt, const float *flat, float *viseme_out)
{
	const OrtApi *ort = rt->ort;
	OrtMemoryInfo *mem = NULL;
	OrtValue *in_tensor = NULL;
	OrtValue *out_tensor = NULL;
	int rc = -1;

	int64_t shape[2] = {1, (int64_t)rt->meta.input_features};
	if (ort_ok(ort,
		   ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
		   "CreateCpuMemoryInfo")) {
		goto done;
	}
	if (ort_ok(ort,
		   ort->CreateTensorWithDataAsOrtValue(
			   mem, (void *)flat, (size_t)rt->meta.input_features * sizeof(float),
			   shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor),
		   "CreateTensor")) {
		goto done;
	}

	const char *in_names[] = {rt->input_name};
	const char *out_names[] = {rt->output_name};
	if (ort_ok(ort,
		   ort->Run(rt->session, NULL, in_names, (const OrtValue *const *)&in_tensor, 1,
			    out_names, 1, &out_tensor),
		   "Run")) {
		goto done;
	}

	float *logits = NULL;
	if (ort_ok(ort, ort->GetTensorMutableData(out_tensor, (void **)&logits), "GetTensorData")) {
		goto done;
	}
	softmax(logits, viseme_out, rt->meta.n_visemes);
	rc = 0;
done:
	if (out_tensor) {
		ort->ReleaseValue(out_tensor);
	}
	if (in_tensor) {
		ort->ReleaseValue(in_tensor);
	}
	if (mem) {
		ort->ReleaseMemoryInfo(mem);
	}
	return rc;
}

VizemesRuntime *vizemes_runtime_create(const char *model_json_path,
				       const char *model_onnx_path,
				       const VizemesFrontendOps *frontend)
{
	if (!model_json_path || !model_onnx_path || !frontend) {
		return NULL;
	}

	VizemesModelMeta meta;
	if (vizemes_sidecar_load(model_json_path, &meta) != 0) {
		fprintf(stderr, "sidecar load failed: %s\n", model_json_path);
		return NULL;
	}

	VizemesFrontendConfig cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.sample_rate = meta.sample_rate;
	cfg.hop_length_samples = meta.hop_length_samples;
	cfg.window_length_samples = meta.window_length_samples;
	cfg.n_features = meta.n_mels;
	cfg.n_fft = meta.n_fft;
	cfg.fmin = meta.fmin;
	cfg.fmax = meta.fmax;
	cfg.top_db = 80.f;
	if (frontend->init && frontend->init(&cfg) != 0) {
		fprintf(stderr, "frontend init failed\n");
		return NULL;
	}

	const OrtApiBase *base = OrtGetApiBase();
	if (!base) {
		fprintf(stderr, "OrtGetApiBase failed\n");
		if (frontend->free) {
			frontend->free();
		}
		return NULL;
	}
	const OrtApi *ort = base->GetApi(ORT_API_VERSION);
	if (!ort) {
		fprintf(stderr, "ORT GetApi failed (API version mismatch?)\n");
		if (frontend->free) {
			frontend->free();
		}
		return NULL;
	}

	VizemesRuntime *rt = (VizemesRuntime *)calloc(1, sizeof(*rt));
	if (!rt) {
		if (frontend->free) {
			frontend->free();
		}
		return NULL;
	}
	rt->frontend = frontend;
	rt->meta = meta;
	rt->ort = ort;

	if (ort_ok(ort, ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vizemes", &rt->env),
		   "CreateEnv") ||
	    ort_ok(ort, ort->CreateSessionOptions(&rt->opts), "CreateSessionOptions") ||
	    ort_ok(ort,
		   ort->CreateSession(rt->env, model_onnx_path, rt->opts, &rt->session),
		   "CreateSession") ||
	    ort_ok(ort, ort->GetAllocatorWithDefaultOptions(&rt->allocator), "GetAllocator")) {
		vizemes_runtime_destroy(rt);
		return NULL;
	}

	if (ort_ok(ort,
		   ort->SessionGetInputName(rt->session, 0, rt->allocator, &rt->input_name),
		   "GetInputName") ||
	    ort_ok(ort,
		   ort->SessionGetOutputName(rt->session, 0, rt->allocator, &rt->output_name),
		   "GetOutputName")) {
		vizemes_runtime_destroy(rt);
		return NULL;
	}

	rt->mel_ring = (float *)calloc((size_t)meta.context_frames * (size_t)meta.n_mels,
				       sizeof(float));
	rt->pcm_pending_cap = meta.window_length_samples * 4;
	rt->pcm_pending = (float *)calloc((size_t)rt->pcm_pending_cap, sizeof(float));
	if (!rt->mel_ring || !rt->pcm_pending) {
		vizemes_runtime_destroy(rt);
		return NULL;
	}
	return rt;
}

void vizemes_runtime_destroy(VizemesRuntime *rt)
{
	if (!rt) {
		return;
	}
	const OrtApi *ort = rt->ort;
	if (ort && rt->allocator) {
		if (rt->input_name) {
			OrtStatus *st = ort->AllocatorFree(rt->allocator, rt->input_name);
			if (st) {
				ort->ReleaseStatus(st);
			}
			rt->input_name = NULL;
		}
		if (rt->output_name) {
			OrtStatus *st = ort->AllocatorFree(rt->allocator, rt->output_name);
			if (st) {
				ort->ReleaseStatus(st);
			}
			rt->output_name = NULL;
		}
	}
	if (ort) {
		if (rt->session) {
			ort->ReleaseSession(rt->session);
		}
		if (rt->opts) {
			ort->ReleaseSessionOptions(rt->opts);
		}
		if (rt->env) {
			ort->ReleaseEnv(rt->env);
		}
	}
	if (rt->frontend && rt->frontend->free) {
		rt->frontend->free();
	}
	free(rt->mel_ring);
	free(rt->pcm_pending);
	free(rt);
}

int vizemes_runtime_run_context(VizemesRuntime *rt, const float *flat_ctx, float *viseme_out)
{
	if (!rt || !flat_ctx || !viseme_out) {
		return -1;
	}
	return run_flat(rt, flat_ctx, viseme_out);
}

int vizemes_runtime_n_visemes(const VizemesRuntime *rt)
{
	return rt ? rt->meta.n_visemes : 0;
}

int vizemes_runtime_input_features(const VizemesRuntime *rt)
{
	return rt ? rt->meta.input_features : 0;
}

static void ring_push(VizemesRuntime *rt, const float *mel_frame)
{
	const int ctx = rt->meta.context_frames;
	const int nm = rt->meta.n_mels;
	if (rt->mel_filled < ctx) {
		memcpy(rt->mel_ring + (size_t)rt->mel_filled * (size_t)nm, mel_frame,
		       (size_t)nm * sizeof(float));
		rt->mel_filled++;
		return;
	}
	memmove(rt->mel_ring, rt->mel_ring + nm, (size_t)(ctx - 1) * (size_t)nm * sizeof(float));
	memcpy(rt->mel_ring + (size_t)(ctx - 1) * (size_t)nm, mel_frame,
	       (size_t)nm * sizeof(float));
}

int vizemes_runtime_push_pcm(VizemesRuntime *rt, const float *pcm, int n_samples,
			     float *viseme_out, int max_frames, int *n_visemes_out)
{
	if (n_visemes_out) {
		*n_visemes_out = rt ? rt->meta.n_visemes : 0;
	}
	if (!rt || !pcm || n_samples < 0 || !viseme_out || max_frames <= 0) {
		return -1;
	}

	const int win = rt->meta.window_length_samples;
	const int hop = rt->meta.hop_length_samples;
	int emitted = 0;

	/* Append into pending buffer */
	if (rt->pcm_pending_n + n_samples > rt->pcm_pending_cap) {
		int need = rt->pcm_pending_n + n_samples;
		float *nb = (float *)realloc(rt->pcm_pending, (size_t)need * sizeof(float));
		if (!nb) {
			return -1;
		}
		rt->pcm_pending = nb;
		rt->pcm_pending_cap = need;
	}
	memcpy(rt->pcm_pending + rt->pcm_pending_n, pcm, (size_t)n_samples * sizeof(float));
	rt->pcm_pending_n += n_samples;

	float *frame = (float *)malloc((size_t)rt->meta.n_mels * sizeof(float));
	float *flat = (float *)malloc((size_t)rt->meta.input_features * sizeof(float));
	if (!frame || !flat) {
		free(frame);
		free(flat);
		return -1;
	}

	while (rt->pcm_pending_n >= win && emitted < max_frames) {
		if (rt->frontend->process_frame(rt->pcm_pending, frame) != 0) {
			free(frame);
			free(flat);
			return -1;
		}
		ring_push(rt, frame);
		/* Drop hop samples (keep overlap) */
		int drop = hop;
		if (drop > rt->pcm_pending_n) {
			drop = rt->pcm_pending_n;
		}
		memmove(rt->pcm_pending, rt->pcm_pending + drop,
			(size_t)(rt->pcm_pending_n - drop) * sizeof(float));
		rt->pcm_pending_n -= drop;

		if (rt->mel_filled < rt->meta.context_frames) {
			continue; /* pad with zeros already in ring calloc */
		}
		/* Prefer full context; if still filling, left-pad zeros (calloc) */
		const int ctx = rt->meta.context_frames;
		const int nm = rt->meta.n_mels;
		memset(flat, 0, (size_t)rt->meta.input_features * sizeof(float));
		int use = rt->mel_filled < ctx ? rt->mel_filled : ctx;
		memcpy(flat + (size_t)(ctx - use) * (size_t)nm, rt->mel_ring,
		       (size_t)use * (size_t)nm * sizeof(float));
		if (run_flat(rt, flat, viseme_out + (size_t)emitted * (size_t)rt->meta.n_visemes) !=
		    0) {
			free(frame);
			free(flat);
			return -1;
		}
		emitted++;
	}

	free(frame);
	free(flat);
	return emitted;
}
