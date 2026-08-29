/**
 * Host CSV smoke: load every demo_inputs.csv row through C ORT runtime.
 * Table shape matches scripts/sanity_check_onnx.py (expect vs predict).
 *
 *   make ORT_ROOT=... smoke-csv
 */
#include "feature_frontend.h"
#include "model_meta.h"
#include "viseme_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE (1600 * 24 + 256)
#define MAX_VISEMES VIZEMES_META_MAX_VISEMES

typedef struct {
	char names[MAX_VISEMES][32];
	int n;
} NameTable;

static int load_names_from_meta(const char *meta_path, NameTable *nt)
{
	VizemesModelMeta meta;
	if (vizemes_meta_load(meta_path, &meta) != 0) {
		return -1;
	}
	memset(nt, 0, sizeof(*nt));
	nt->n = meta.n_visemes;
	for (int i = 0; i < meta.n_visemes && i < MAX_VISEMES; i++) {
		snprintf(nt->names[i], sizeof(nt->names[i]), "%s", meta.viseme_names[i]);
	}
	return nt->n > 0 ? 0 : -1;
}

static int split_csv_line(char *line, char **fields, int max_fields)
{
	int n = 0;
	char *p = line;
	while (n < max_fields) {
		fields[n++] = p;
		char *comma = strchr(p, ',');
		if (!comma) {
			break;
		}
		*comma = '\0';
		p = comma + 1;
	}
	/* strip trailing CR/LF on last field */
	if (n > 0) {
		char *last = fields[n - 1];
		size_t L = strlen(last);
		while (L > 0 && (last[L - 1] == '\n' || last[L - 1] == '\r')) {
			last[--L] = '\0';
		}
	}
	return n;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s model.meta model.onnx demo_inputs.csv\n", argv[0]);
		return 2;
	}
	const char *meta_path = argv[1];
	const char *onnx_path = argv[2];
	const char *csv_path = argv[3];

	NameTable names;
	if (load_names_from_meta(meta_path, &names) != 0) {
		fprintf(stderr, "failed to load visemes from %s\n", meta_path);
		return 1;
	}

	VizemesRuntime *rt =
		vizemes_runtime_create(meta_path, onnx_path, vizemes_frontend_mel());
	if (!rt) {
		fprintf(stderr, "vizemes_runtime_create failed\n");
		return 1;
	}

	int nfeat = vizemes_runtime_input_features(rt);
	int nv = vizemes_runtime_n_visemes(rt);
	if (nv > names.n) {
		fprintf(stderr, "runtime n_visemes=%d > meta names=%d\n", nv, names.n);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	FILE *csv = fopen(csv_path, "r");
	if (!csv) {
		perror(csv_path);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	char *line = (char *)malloc(MAX_LINE);
	float *ctx = (float *)malloc((size_t)nfeat * sizeof(float));
	float *w = (float *)calloc((size_t)nv, sizeof(float));
	if (!line || !ctx || !w) {
		fprintf(stderr, "oom\n");
		free(line);
		free(ctx);
		free(w);
		fclose(csv);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	if (!fgets(line, MAX_LINE, csv)) {
		fprintf(stderr, "empty csv\n");
		free(line);
		free(ctx);
		free(w);
		fclose(csv);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	/* Header: probe_id,expect_id,expect_name,f0,... */
	char *hdr_fields[2048];
	int nh = split_csv_line(line, hdr_fields, 2048);
	if (nh < 3 + nfeat) {
		fprintf(stderr, "csv header fields=%d need >= %d\n", nh, 3 + nfeat);
		free(line);
		free(ctx);
		free(w);
		fclose(csv);
		vizemes_runtime_destroy(rt);
		return 1;
	}

	printf("%5s  %-8s  %-8s  %7s  hit\n", "probe", "expect", "predict", "P(exp)");
	int hits = 0;
	int n = 0;

	while (fgets(line, MAX_LINE, csv)) {
		if (line[0] == '\0' || line[0] == '\n') {
			continue;
		}
		char *fields[2048];
		int nf = split_csv_line(line, fields, 2048);
		if (nf < 3 + nfeat) {
			fprintf(stderr, "short row fields=%d\n", nf);
			continue;
		}
		int probe_id = atoi(fields[0]);
		int expect_id = atoi(fields[1]);
		const char *expect_name = fields[2];
		for (int i = 0; i < nfeat; i++) {
			ctx[i] = strtof(fields[3 + i], NULL);
		}
		if (vizemes_runtime_run_context(rt, ctx, w) != 0) {
			fprintf(stderr, "run_context failed probe=%d\n", probe_id);
			free(line);
			free(ctx);
			free(w);
			fclose(csv);
			vizemes_runtime_destroy(rt);
			return 1;
		}
		int pred = 0;
		for (int i = 1; i < nv; i++) {
			if (w[i] > w[pred]) {
				pred = i;
			}
		}
		int hit = (pred == expect_id);
		hits += hit;
		n++;
		const char *pred_name =
			(pred >= 0 && pred < names.n) ? names.names[pred] : "?";
		float p_exp = (expect_id >= 0 && expect_id < nv) ? w[expect_id] : 0.f;
		printf("%5d  %-8s  %-8s  %7.3f  %s\n", probe_id, expect_name, pred_name, p_exp,
		       hit ? "Y" : ".");
	}

	printf("hit_rate=%d/%d\n", hits, n);
	printf("GDEXT_ORT_CSV_SMOKE_OK rows=%d\n", n);

	free(line);
	free(ctx);
	free(w);
	fclose(csv);
	vizemes_runtime_destroy(rt);
	return n > 0 ? 0 : 1;
}
