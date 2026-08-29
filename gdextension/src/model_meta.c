#include "model_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trim(char *s)
{
	char *e = s + strlen(s);
	while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) {
		*--e = '\0';
	}
	char *p = s;
	while (*p == ' ' || *p == '\t') {
		p++;
	}
	if (p != s) {
		memmove(s, p, strlen(p) + 1);
	}
}

int vizemes_meta_load(const char *path, VizemesModelMeta *out)
{
	if (!path || !out) {
		return -1;
	}
	memset(out, 0, sizeof(*out));
	out->n_fft = 1024;
	out->fmin = 50.f;
	out->fmax = 8000.f;

	FILE *f = fopen(path, "r");
	if (!f) {
		return -1;
	}
	char line[256];
	while (fgets(line, sizeof(line), f)) {
		trim(line);
		if (line[0] == '\0' || line[0] == '#') {
			continue;
		}
		char *eq = strchr(line, '=');
		if (!eq) {
			continue;
		}
		*eq = '\0';
		char *key = line;
		char *val = eq + 1;
		trim(key);
		trim(val);
		if (strcmp(key, "context_frames") == 0) {
			out->context_frames = atoi(val);
		} else if (strcmp(key, "n_mels") == 0) {
			out->n_mels = atoi(val);
		} else if (strcmp(key, "input_features") == 0) {
			out->input_features = atoi(val);
		} else if (strcmp(key, "n_visemes") == 0) {
			out->n_visemes = atoi(val);
		} else if (strcmp(key, "sample_rate") == 0) {
			out->sample_rate = atoi(val);
		} else if (strcmp(key, "hop_length_samples") == 0) {
			out->hop_length_samples = atoi(val);
		} else if (strcmp(key, "window_length_samples") == 0) {
			out->window_length_samples = atoi(val);
		} else if (strcmp(key, "n_fft") == 0) {
			out->n_fft = atoi(val);
		} else if (strcmp(key, "fmin") == 0) {
			out->fmin = strtof(val, NULL);
		} else if (strcmp(key, "fmax") == 0) {
			out->fmax = strtof(val, NULL);
		} else if (strncmp(key, "viseme_", 7) == 0) {
			int id = atoi(key + 7);
			if (id >= 0 && id < VIZEMES_META_MAX_VISEMES) {
				snprintf(out->viseme_names[id], sizeof(out->viseme_names[id]), "%s", val);
				if (id + 1 > out->n_visemes) {
					/* n_visemes key may appear later; track high-water */
				}
			}
		}
	}
	fclose(f);

	if (out->context_frames <= 0 || out->n_mels <= 0 || out->n_visemes <= 0 ||
			out->input_features != out->context_frames * out->n_mels ||
			out->sample_rate <= 0 || out->hop_length_samples <= 0 ||
			out->window_length_samples <= 0) {
		return -1;
	}
	return 0;
}
