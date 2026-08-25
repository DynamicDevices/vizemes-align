#include "feature_frontend.h"
#include "mel_spectrogram.h"

#include <string.h>

static int mel_init(const VizemesFrontendConfig *cfg)
{
	MelSpectrogramConfig mc;
	memset(&mc, 0, sizeof(mc));
	mc.sample_rate = cfg->sample_rate;
	mc.hop_length_samples = cfg->hop_length_samples;
	mc.window_length_samples = cfg->window_length_samples;
	mc.n_fft = cfg->n_fft > 0 ? cfg->n_fft : 1024;
	mc.n_mels = cfg->n_features;
	mc.fmin = cfg->fmin;
	mc.fmax = cfg->fmax;
	mc.top_db = cfg->top_db > 0.f ? cfg->top_db : 80.f;
	return mel_spectrogram_init(&mc);
}

static void mel_free(void)
{
	mel_spectrogram_free();
}

static int mel_process_frame(const float *audio_window, float *feat_out)
{
	return mel_spectrogram_process_frame(audio_window, feat_out);
}

static int mel_process_waveform(const float *samples, int num_samples, float *feat_out,
				int max_frames)
{
	size_t nframes = 0;
	int rc = mel_spectrogram_process(samples, (size_t)num_samples, feat_out, &nframes);
	if (rc < 0) {
		return -1;
	}
	if (max_frames > 0 && (int)nframes > max_frames) {
		return max_frames;
	}
	return (int)nframes;
}

static const VizemesFrontendOps MEL_OPS = {
	.name = "mel",
	.init = mel_init,
	.free = mel_free,
	.process_frame = mel_process_frame,
	.process_waveform = mel_process_waveform,
};

const VizemesFrontendOps *vizemes_frontend_mel(void)
{
	return &MEL_OPS;
}
