"""
feature_extraction.py
---------------------
Audio feature extraction: Mel spectrograms + handcrafted stats.
Shared by both genre and emotion pipelines.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MELS = 128
SEGMENT_DURATION = 3  # seconds
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION


def load_audio(file_path):
    """Load audio file and return signal + sample rate."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return signal, sr


def extract_mel(signal, sr):
    """
    Extract a single normalized Mel spectrogram from the first 3-second segment.
    Returns shape: (1, 128, T, 1) ready for CNN input.
    """
    segment = signal[:SAMPLES_PER_SEGMENT]

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    std = np.std(mel)
    if std != 0:
        mel = (mel - np.mean(mel)) / std

    return mel[np.newaxis, ..., np.newaxis]


def extract_mel_segments(signal, sr, max_segments=5):
    """
    Split the audio into non-overlapping 3-second segments and return
    a list of normalised Mel spectrograms, each shaped (1, 128, T, 1).

    At most *max_segments* are returned.  If the audio is shorter than
    one full segment the single (zero-padded) segment is still returned.
    """
    total_samples = len(signal)
    segments = []

    for i in range(max_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        if start >= total_samples:
            break
        chunk = signal[start:end]
        # Zero-pad if the last chunk is shorter than 3 s
        if len(chunk) < SAMPLES_PER_SEGMENT:
            chunk = np.pad(chunk, (0, SAMPLES_PER_SEGMENT - len(chunk)))

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS)
        mel = librosa.power_to_db(mel, ref=np.max)

        std = np.std(mel)
        if std != 0:
            mel = (mel - np.mean(mel)) / std

        segments.append(mel[np.newaxis, ..., np.newaxis])

    return segments if segments else [extract_mel(signal, sr)]


def extract_handcrafted(signal, sr):
    """
    Extract handcrafted features from the first 3-second segment.
    Returns shape: (1, 4) — [tempo, spectral_centroid, rms, zcr]
    """
    segment = signal[:SAMPLES_PER_SEGMENT]

    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    tempo = float(np.squeeze(tempo))

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=segment)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(segment)))

    return np.array([[tempo, centroid, rms, zcr]], dtype=np.float32)


def extract_handcrafted_segments(signal, sr, max_segments=5):
    """
    Extract handcrafted feature vectors for each non-overlapping 3-s segment.

    Returns a list of arrays, each shaped (1, 4).
    """
    total_samples = len(signal)
    stats_list = []

    for i in range(max_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        if start >= total_samples:
            break
        chunk = signal[start:end]
        if len(chunk) < SAMPLES_PER_SEGMENT:
            chunk = np.pad(chunk, (0, SAMPLES_PER_SEGMENT - len(chunk)))

        tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr)
        tempo = float(np.squeeze(tempo))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
        rms = float(np.mean(librosa.feature.rms(y=chunk)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))

        stats_list.append(np.array([[tempo, centroid, rms, zcr]], dtype=np.float32))

    return stats_list if stats_list else [extract_handcrafted(signal, sr)]
