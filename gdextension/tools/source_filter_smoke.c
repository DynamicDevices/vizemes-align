#include "source_filter.h"

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned int random_state = 1;

static float noise_sample(void)
{
	random_state = random_state * 1664525u + 1013904223u;
	return ((random_state >> 8) / 8388608.0f - 1.0f) * 0.2f;
}

int main(void)
{
	float pcm[VIZEMES_SOURCE_FILTER_SAMPLES] = {0};
	VizemesSourceFilterFrame frame;
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) {
		pcm[i] = 0.4f * sinf(2.0f * (float)M_PI * 200.0f * i / 16000.0f);
	}
	if (vizemes_source_filter_process(pcm, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0 ||
			!frame.pitch_valid || fabsf(frame.pitch_hz - 200.0f) > 4.0f ||
			frame.periodicity < 0.9f) {
		fprintf(stderr, "sine failed: pitch=%g periodicity=%g valid=%d\n",
				frame.pitch_hz, frame.periodicity, frame.pitch_valid);
		return 1;
	}
	float sine_periodicity = frame.periodicity;

	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) pcm[i] = noise_sample();
	if (vizemes_source_filter_process(pcm, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0 ||
			frame.periodicity >= sine_periodicity) {
		fprintf(stderr, "noise failed: periodicity=%g\n", frame.periodicity);
		return 1;
	}
	float noise_periodicity = frame.periodicity;

	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) pcm[i] = 0.0f;
	if (vizemes_source_filter_process(pcm, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0 ||
			frame.pitch_valid || frame.pitch_hz != 0.0f) {
		fprintf(stderr, "silence failed: pitch=%g valid=%d\n", frame.pitch_hz, frame.pitch_valid);
		return 1;
	}
	printf("SOURCE_FILTER_SMOKE_OK sine_pitch=200 sine_periodicity=%.3f noise_periodicity=%.3f\n",
			sine_periodicity, noise_periodicity);
	return 0;
}
