#!/usr/bin/env python3
"""Training features from the production C speech source/filter extractor."""
from __future__ import annotations

import ctypes
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gdextension/frontends/source_filter/source_filter.c"
INCLUDE = SOURCE.parent
LIBRARY = ROOT / "gdextension/build/libvizemes_source_filter_host.so"
SAMPLE_RATE = 16_000
WINDOW = 400
HOP = 160
LPC_ORDER = 16
ENVELOPE_BINS = 80


class _Frame(ctypes.Structure):
    _fields_ = [
        ("reflection", ctypes.c_float * LPC_ORDER),
        ("envelope_db", ctypes.c_float * ENVELOPE_BINS),
        ("rms_dbfs", ctypes.c_float),
        ("prediction_gain_db", ctypes.c_float),
        ("periodicity", ctypes.c_float),
        ("pitch_hz", ctypes.c_float),
        ("pitch_confidence", ctypes.c_float),
        ("hnr_db", ctypes.c_float),
        ("residual_tilt_db_octave", ctypes.c_float),
        ("pitch_valid", ctypes.c_int),
    ]


def _library() -> ctypes.CDLL:
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cc", "-O3", "-std=c11", "-shared", "-fPIC", f"-I{INCLUDE}",
             str(SOURCE), "-lm", "-o", str(LIBRARY)],
            check=True,
        )
    library = ctypes.CDLL(str(LIBRARY))
    library.vizemes_source_filter_process.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(_Frame)
    ]
    library.vizemes_source_filter_process.restype = ctypes.c_int
    return library


def _pcm16_mono(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wav:
        contract = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        if contract != (1, 2, SAMPLE_RATE):
            raise ValueError(f"expected mono PCM16 16 kHz WAV, got {contract}: {wav_path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def source_filter_features_c(wav_path: Path, include_source: bool = True) -> np.ndarray:
    """Return reflection coefficients, optionally followed by eight source/reliability values."""
    pcm = _pcm16_mono(wav_path)
    frames = 1 + (len(pcm) - WINDOW) // HOP
    if frames <= 0:
        return np.empty((0, LPC_ORDER + (8 if include_source else 0)), np.float32)
    output = np.empty((frames, LPC_ORDER + (8 if include_source else 0)), np.float32)
    library = _library()
    frame = _Frame()
    for index in range(frames):
        window = np.ascontiguousarray(pcm[index * HOP:index * HOP + WINDOW])
        pointer = window.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if library.vizemes_source_filter_process(pointer, WINDOW, ctypes.byref(frame)) != 0:
            raise RuntimeError(f"source/filter extraction failed at frame {index}")
        output[index, :LPC_ORDER] = frame.reflection
        if include_source:
            pitch_log = np.log(max(frame.pitch_hz, 70.0) / 70.0) / np.log(400.0 / 70.0)
            output[index, LPC_ORDER:] = (
                np.clip((frame.rms_dbfs + 60.0) / 60.0, 0.0, 1.0),
                np.clip(frame.prediction_gain_db / 30.0, 0.0, 1.0),
                frame.periodicity,
                np.clip(pitch_log, 0.0, 1.0),
                frame.pitch_confidence,
                float(frame.pitch_valid),
                np.clip((frame.hnr_db + 5.0) / 35.0, 0.0, 1.0),
                np.clip((frame.residual_tilt_db_octave + 18.0) / 36.0, 0.0, 1.0),
            )
    return output


if __name__ == "__main__":
    import sys
    values = source_filter_features_c(Path(sys.argv[1]))
    print(f"SOURCE_FILTER_C_OK frames={values.shape[0]} features={values.shape[1]}")
