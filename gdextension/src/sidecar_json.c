#include "sidecar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int find_int(const char *json, const char *key, int *out)
{
	char pat[128];
	snprintf(pat, sizeof(pat), "\"%s\"", key);
	const char *p = strstr(json, pat);
	if (!p) {
		return -1;
	}
	p = strchr(p + strlen(pat), ':');
	if (!p) {
		return -1;
	}
	p++;
	while (*p == ' ' || *p == '\t') {
		p++;
	}
	char *end = NULL;
	long v = strtol(p, &end, 10);
	if (end == p) {
		return -1;
	}
	*out = (int)v;
	return 0;
}

static int find_float(const char *json, const char *key, float *out)
{
	char pat[128];
	snprintf(pat, sizeof(pat), "\"%s\"", key);
	const char *p = strstr(json, pat);
	if (!p) {
		return -1;
	}
	p = strchr(p + strlen(pat), ':');
	if (!p) {
		return -1;
	}
	p++;
	while (*p == ' ' || *p == '\t') {
		p++;
	}
	char *end = NULL;
	float v = strtof(p, &end);
	if (end == p) {
		return -1;
	}
	*out = v;
	return 0;
}

int vizemes_sidecar_load(const char *path, VizemesModelMeta *out)
{
	if (!path || !out) {
		return -1;
	}
	memset(out, 0, sizeof(*out));
	FILE *f = fopen(path, "rb");
	if (!f) {
		return -1;
	}
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return -1;
	}
	long n = ftell(f);
	if (n <= 0 || n > 1 << 20) {
		fclose(f);
		return -1;
	}
	rewind(f);
	char *buf = (char *)malloc((size_t)n + 1);
	if (!buf) {
		fclose(f);
		return -1;
	}
	if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
		free(buf);
		fclose(f);
		return -1;
	}
	buf[n] = '\0';
	fclose(f);

	int rc = 0;
	if (find_int(buf, "context_frames", &out->context_frames) ||
	    find_int(buf, "n_mels", &out->n_mels) ||
	    find_int(buf, "input_features", &out->input_features) ||
	    find_int(buf, "n_visemes", &out->n_visemes) ||
	    find_int(buf, "sample_rate", &out->sample_rate) ||
	    find_int(buf, "hop_length_samples", &out->hop_length_samples) ||
	    find_int(buf, "window_length_samples", &out->window_length_samples)) {
		rc = -1;
	}
	/* n_fft / fmin / fmax optional with defaults matching smoke model */
	if (find_int(buf, "n_fft", &out->n_fft) != 0) {
		out->n_fft = 1024;
	}
	if (find_float(buf, "fmin", &out->fmin) != 0) {
		out->fmin = 50.f;
	}
	if (find_float(buf, "fmax", &out->fmax) != 0) {
		out->fmax = 8000.f;
	}
	free(buf);
	if (rc != 0) {
		return -1;
	}
	if (out->context_frames <= 0 || out->n_mels <= 0 || out->n_visemes <= 0 ||
	    out->input_features != out->context_frames * out->n_mels ||
	    out->sample_rate <= 0 || out->hop_length_samples <= 0 ||
	    out->window_length_samples <= 0) {
		return -1;
	}
	return 0;
}
