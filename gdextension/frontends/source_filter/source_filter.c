#include "source_filter.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PREEMPHASIS 0.97f
#define QUIET_DBFS -60.0f
#define PITCH_MIN_HZ 70
#define PITCH_MAX_HZ 400

static float clampf(float value, float low, float high)
{
	if (value < low) return low;
	if (value > high) return high;
	return value;
}

static float normalized_correlation(const float *pcm, int lag)
{
	float dot = 0.0f;
	float energy_a = 0.0f;
	float energy_b = 0.0f;
	for (int i = lag; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) {
		float a = pcm[i];
		float b = pcm[i - lag];
		dot += a * b;
		energy_a += a * a;
		energy_b += b * b;
	}
	float denominator = sqrtf(energy_a * energy_b);
	return denominator > 1e-12f ? dot / denominator : 0.0f;
}

static float find_periodicity(const float *pcm, int *best_lag)
{
	float mean = 0.0f;
	float centered[VIZEMES_SOURCE_FILTER_SAMPLES];
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) mean += pcm[i];
	mean /= VIZEMES_SOURCE_FILTER_SAMPLES;
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) centered[i] = pcm[i] - mean;

	float best = 0.0f;
	*best_lag = 0;
	int first_lag = VIZEMES_SOURCE_FILTER_SAMPLE_RATE / PITCH_MAX_HZ;
	int last_lag = VIZEMES_SOURCE_FILTER_SAMPLE_RATE / PITCH_MIN_HZ;
	for (int lag = first_lag; lag <= last_lag; lag++) {
		float correlation = normalized_correlation(centered, lag);
		if (correlation > best) {
			best = correlation;
			*best_lag = lag;
		}
	}
	return clampf(best, 0.0f, 1.0f);
}

static float levinson(const float *autocorrelation, float *lpc, float *reflection)
{
	float previous[VIZEMES_SOURCE_FILTER_LPC_ORDER + 1];
	memset(lpc, 0, sizeof(float) * (VIZEMES_SOURCE_FILTER_LPC_ORDER + 1));
	lpc[0] = 1.0f;
	float error = fmaxf(autocorrelation[0], 1e-12f);

	for (int order = 1; order <= VIZEMES_SOURCE_FILTER_LPC_ORDER; order++) {
		float numerator = autocorrelation[order];
		for (int j = 1; j < order; j++) numerator += lpc[j] * autocorrelation[order - j];
		float coefficient = clampf(-numerator / error, -0.999f, 0.999f);
		memcpy(previous, lpc, sizeof(previous));
		lpc[order] = coefficient;
		for (int j = 1; j < order; j++) lpc[j] = previous[j] + coefficient * previous[order - j];
		reflection[order - 1] = coefficient;
		error = fmaxf(error * (1.0f - coefficient * coefficient), 1e-12f);
	}
	return error;
}

static void make_envelope(const float *lpc, float error, float *envelope, float *tilt)
{
	float peak = -1e30f;
	for (int bin = 0; bin < VIZEMES_SOURCE_FILTER_ENVELOPE_BINS; bin++) {
		float frequency = 4000.0f * bin / (VIZEMES_SOURCE_FILTER_ENVELOPE_BINS - 1);
		float angle = -2.0f * (float)M_PI * frequency / VIZEMES_SOURCE_FILTER_SAMPLE_RATE;
		float real = 1.0f;
		float imaginary = 0.0f;
		for (int order = 1; order <= VIZEMES_SOURCE_FILTER_LPC_ORDER; order++) {
			real += lpc[order] * cosf(angle * order);
			imaginary += lpc[order] * sinf(angle * order);
		}
		envelope[bin] = 10.0f * log10f(error / fmaxf(real * real + imaginary * imaginary, 1e-12f));
		peak = fmaxf(peak, envelope[bin]);
	}
	for (int bin = 0; bin < VIZEMES_SOURCE_FILTER_ENVELOPE_BINS; bin++) {
		envelope[bin] = fmaxf(envelope[bin] - peak, -50.0f);
	}

	/* Least-squares slope against log2 frequency, excluding near-DC bins. */
	float mean_x = 0.0f;
	float mean_y = 0.0f;
	int count = VIZEMES_SOURCE_FILTER_ENVELOPE_BINS - 4;
	for (int bin = 4; bin < VIZEMES_SOURCE_FILTER_ENVELOPE_BINS; bin++) {
		mean_x += log2f((4000.0f * bin / (VIZEMES_SOURCE_FILTER_ENVELOPE_BINS - 1)) / 200.0f);
		mean_y += envelope[bin];
	}
	mean_x /= count;
	mean_y /= count;
	float numerator = 0.0f;
	float denominator = 0.0f;
	for (int bin = 4; bin < VIZEMES_SOURCE_FILTER_ENVELOPE_BINS; bin++) {
		float x = log2f((4000.0f * bin / (VIZEMES_SOURCE_FILTER_ENVELOPE_BINS - 1)) / 200.0f) - mean_x;
		numerator += x * (envelope[bin] - mean_y);
		denominator += x * x;
	}
	*tilt = denominator > 0.0f ? numerator / denominator : 0.0f;
}

int vizemes_source_filter_process(const float *pcm, size_t sample_count,
		VizemesSourceFilterFrame *out)
{
	if (!pcm || !out || sample_count != VIZEMES_SOURCE_FILTER_SAMPLES) return -1;
	memset(out, 0, sizeof(*out));

	float energy = 0.0f;
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) energy += pcm[i] * pcm[i];
	float rms = sqrtf(energy / VIZEMES_SOURCE_FILTER_SAMPLES);
	out->rms_dbfs = 20.0f * log10f(fmaxf(rms, 1e-8f));

	int pitch_lag = 0;
	out->periodicity = find_periodicity(pcm, &pitch_lag);
	out->pitch_confidence = out->rms_dbfs > QUIET_DBFS
			? clampf((out->periodicity - 0.2f) / 0.65f, 0.0f, 1.0f)
			: 0.0f;
	out->pitch_valid = pitch_lag > 0 && out->pitch_confidence >= 0.2f;
	out->pitch_hz = out->pitch_valid
			? (float)VIZEMES_SOURCE_FILTER_SAMPLE_RATE / pitch_lag
			: 0.0f;
	out->hnr_db = out->pitch_confidence > 0.0f
			? 10.0f * log10f(fmaxf(out->periodicity, 1e-4f) /
					fmaxf(1.0f - out->periodicity, 1e-4f))
			: -10.0f;

	float windowed[VIZEMES_SOURCE_FILTER_SAMPLES];
	for (int i = 0; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) {
		float previous = i > 0 ? pcm[i - 1] : pcm[0];
		float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i /
				(VIZEMES_SOURCE_FILTER_SAMPLES - 1));
		windowed[i] = (pcm[i] - PREEMPHASIS * previous) * hann;
	}
	float autocorrelation[VIZEMES_SOURCE_FILTER_LPC_ORDER + 1];
	for (int lag = 0; lag <= VIZEMES_SOURCE_FILTER_LPC_ORDER; lag++) {
		autocorrelation[lag] = 0.0f;
		for (int i = lag; i < VIZEMES_SOURCE_FILTER_SAMPLES; i++) {
			autocorrelation[lag] += windowed[i] * windowed[i - lag];
		}
	}
	float lpc[VIZEMES_SOURCE_FILTER_LPC_ORDER + 1];
	float residual = levinson(autocorrelation, lpc, out->reflection);
	out->prediction_gain_db = 10.0f * log10f(fmaxf(autocorrelation[0], 1e-12f) / residual);
	make_envelope(lpc, residual, out->envelope_db, &out->residual_tilt_db_octave);
	return 0;
}
