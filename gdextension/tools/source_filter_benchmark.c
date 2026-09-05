#include "source_filter.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
	float pcm[VIZEMES_SOURCE_FILTER_SAMPLES];
	VizemesSourceFilterFrame frame;
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) {
		pcm[i] = 0.25f * sinf(2.0f * (float)M_PI * 137.0f * i /
				VIZEMES_SOURCE_FILTER_SAMPLE_RATE);
	}
	const int iterations = 10000;
	clock_t start = clock();
	for (int i = 0; i < iterations; i++) {
		if (vizemes_source_filter_process(pcm, VIZEMES_SOURCE_FILTER_SAMPLES, &frame) != 0) return 1;
		pcm[i % VIZEMES_SOURCE_FILTER_SAMPLES] += 1e-7f;
	}
	double microseconds = 1000000.0 * (clock() - start) / CLOCKS_PER_SEC / iterations;
	printf("SOURCE_FILTER_BENCHMARK frames=%d us_per_frame=%.2f realtime_cpu_at_10ms=%.3f%%\n",
			iterations, microseconds, microseconds / 100.0);
	return 0;
}
