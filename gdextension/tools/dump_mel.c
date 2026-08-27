/**
 * Dump C mel frames for parity check vs torchaudio (scripts/compare_mel_parity.py).
 *
 *   make dump-mel
 *   ./build/dump_mel ../export/ci-smoke/model.json ../export/ci-smoke/ci-fixture.wav > /tmp/c-mel.txt
 */
#include "mel_spectrogram.h"
#include "sidecar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_wav_mono_f32(const char *path, float **out, int *out_n)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		perror(path);
		return -1;
	}
	unsigned char hdr[12];
	if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0) {
		fprintf(stderr, "not RIFF: %s\n", path);
		fclose(f);
		return -1;
	}
	int channels = 1, bits = 16, rate = 16000;
	long data_off = 0, data_sz = 0;
	while (fread(hdr, 1, 8, f) == 8) {
		unsigned sz = (unsigned)hdr[4] | ((unsigned)hdr[5] << 8) | ((unsigned)hdr[6] << 16) |
			      ((unsigned)hdr[7] << 24);
		if (memcmp(hdr, "fmt ", 4) == 0) {
			unsigned short af, ch, ba;
			unsigned ar, br;
			unsigned short bits_raw;
			if (fread(&af, 2, 1, f) != 1)
				break;
			if (fread(&ch, 2, 1, f) != 1)
				break;
			if (fread(&ar, 4, 1, f) != 1)
				break;
			if (fread(&br, 4, 1, f) != 1)
				break;
			if (fread(&ba, 2, 1, f) != 1)
				break;
			if (fread(&bits_raw, 2, 1, f) != 1)
				break;
			channels = (int)ch;
			rate = (int)ar;
			bits = (int)bits_raw;
			if (sz > 16)
				fseek(f, (long)sz - 16, SEEK_CUR);
		} else if (memcmp(hdr, "data", 4) == 0) {
			data_off = ftell(f);
			data_sz = (long)sz;
			break;
		} else {
			fseek(f, (long)sz, SEEK_CUR);
		}
	}
	if (data_off <= 0 || channels != 1 || bits != 16) {
		fprintf(stderr, "bad wav fmt sr=%d ch=%d bits=%d: %s\n", rate, channels, bits, path);
		fclose(f);
		return -1;
	}
	int ns = (int)(data_sz / 2);
	float *pcm = (float *)malloc((size_t)ns * sizeof(float));
	if (!pcm) {
		fclose(f);
		return -1;
	}
	fseek(f, data_off, SEEK_SET);
	for (int i = 0; i < ns; i++) {
		int lo = fgetc(f), hi = fgetc(f);
		if (lo < 0 || hi < 0) {
			free(pcm);
			fclose(f);
			return -1;
		}
		int s = lo | (hi << 8);
		if (s >= 32768)
			s -= 65536;
		pcm[i] = (float)s / 32768.f;
	}
	fclose(f);
	*out = pcm;
	*out_n = ns;
	return 0;
}

int main(int argc, char **argv)
{
	int power_only = 0;
	if (argc >= 2 && strcmp(argv[1], "--power") == 0) {
		power_only = 1;
		argc--;
		argv++;
	}
	if (argc != 3) {
		fprintf(stderr, "usage: %s [--power] model.json audio.wav\n", argv[0]);
		return 2;
	}

	VizemesModelMeta meta;
	if (vizemes_sidecar_load(argv[1], &meta) != 0) {
		fprintf(stderr, "sidecar load failed: %s\n", argv[1]);
		return 1;
	}

	MelSpectrogramConfig mc;
	memset(&mc, 0, sizeof(mc));
	mc.sample_rate = meta.sample_rate;
	mc.hop_length_samples = meta.hop_length_samples;
	mc.window_length_samples = meta.window_length_samples;
	mc.n_fft = meta.n_fft > 0 ? meta.n_fft : 1024;
	mc.n_mels = meta.n_mels;
	mc.fmin = meta.fmin;
	mc.fmax = meta.fmax;
	mc.top_db = 80.f;
	if (mel_spectrogram_init(&mc) != 0) {
		fprintf(stderr, "mel_spectrogram_init failed\n");
		return 1;
	}

	float *pcm = NULL;
	int ns = 0;
	if (load_wav_mono_f32(argv[2], &pcm, &ns) != 0) {
		mel_spectrogram_free();
		return 1;
	}

	size_t max_frames = (size_t)ns / (size_t)mc.hop_length_samples + 4;
	float *mel = (float *)calloc(max_frames * (size_t)mc.n_mels, sizeof(float));
	if (!mel) {
		free(pcm);
		mel_spectrogram_free();
		return 1;
	}
	size_t nframes = 0;
	if (power_only) {
		if (mel_spectrogram_process_power(pcm, (size_t)ns, mel, &nframes) < 0) {
			fprintf(stderr, "mel_spectrogram_process_power failed\n");
			free(mel);
			free(pcm);
			mel_spectrogram_free();
			return 1;
		}
	} else if (mel_spectrogram_process(pcm, (size_t)ns, mel, &nframes) < 0) {
		fprintf(stderr, "mel_spectrogram_process failed\n");
		free(mel);
		free(pcm);
		mel_spectrogram_free();
		return 1;
	}

	printf("frames %zu n_mels %d\n", nframes, mc.n_mels);
	for (size_t t = 0; t < nframes; t++) {
		for (int m = 0; m < mc.n_mels; m++) {
			printf("%.8g%c", mel[t * (size_t)mc.n_mels + (size_t)m], m + 1 == mc.n_mels ? '\n' : ' ');
		}
	}

	free(mel);
	free(pcm);
	mel_spectrogram_free();
	return 0;
}
