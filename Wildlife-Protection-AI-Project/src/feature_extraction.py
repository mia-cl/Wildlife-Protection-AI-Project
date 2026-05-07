"""Audio feature extraction utilities for wildlife sound classification."""

from __future__ import annotations

import numpy as np
import librosa

from config import SAMPLE_RATE, DURATION, N_MFCC


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load an audio file with a fixed sample rate and duration."""
    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True,
    )
    return audio, sample_rate


def extract_features(file_path: str) -> np.ndarray:
    """Extract numerical features from an audio file.

    Features include MFCCs, chroma, spectral centroid, spectral rolloff,
    zero crossing rate and root mean square energy.
    """
    audio, sample_rate = load_audio(file_path)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {file_path}")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_rolloff, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.mean(rms, axis=1),
    ])

    return feature_vector
