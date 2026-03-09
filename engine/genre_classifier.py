"""genre_classifier.py
--------------------
Loads the FMA genre CNN model and provides genre prediction.
Softmax outputs are treated as fuzzy membership degrees.
Labels are loaded from models/genre_labels.json.
"""

import json
import os
from collections import deque
import numpy as np
import librosa
import tensorflow as tf

# ── Audio / spectrogram constants (must match training) ─────────
SR = 22_050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_BINS = 431        # 10-s segment → ceil(220500/512)+1
DURATION = 30                 # seconds of audio to use
N_SEGMENTS = 3                # segments per track


def _load_labels(json_path: str) -> list[str]:
    """Read genre_labels.json → ordered list of genre strings."""
    with open(json_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


class GenreClassifier:
    """FMA-medium genre classifier (10-class CNN on log-mel spectrograms)."""

    def __init__(self, model_path=None, labels_path=None, *,
                 model=None, labels=None, temperature=1.0):
        """
        Args:
            model_path:  Path to .keras file (loads from disk).
            labels_path: Path to genre_labels.json.
            model:       Pre-loaded tf.keras.Model (preferred — avoids reload).
            labels:      Pre-loaded list[str] of genre labels.
            temperature: Softmax temperature scaling (default 1.0 = no change).
        Supply *one* of model_path / model.  ``model`` takes priority.
        """
        # ── Model ──────────────────────────────────────────────────
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Provide model_path or model")

        # ── Labels ─────────────────────────────────────────────────
        if labels is not None:
            self.labels = labels
        elif labels_path is not None:
            self.labels = _load_labels(labels_path)
        else:
            # Default: try models/genre_labels.json relative to this file
            default = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir, "models", "genre_labels.json",
            )
            self.labels = _load_labels(os.path.normpath(default))

        self.temperature = temperature

        # Rolling buffer for live-mic temporal smoothing (last 5 predictions)
        self._live_prob_history: deque[np.ndarray] = deque(maxlen=5)

    # ── High-level: audio file → result ─────────────────────────

    def predict_file(self, audio_path: str) -> dict:
        """End-to-end prediction from an audio file path.

        Loads audio, extracts 3 mel segments, averages softmax,
        and returns the standard result dict with top-2 genres.
        """
        mel_segments = self._audio_to_mel_segments(audio_path)
        return self.predict_averaged(mel_segments)

    # ── Segment-level API (used by recommender / desktop app) ───

    def predict(self, mel):
        """Predict genre from a single mel spectrogram.

        Args:
            mel: numpy array of shape (1, 128, 431, 1)

        Returns:
            dict with top_genre, confidence, fuzzy_memberships
        """
        probs = self.model.predict(mel, verbose=0)[0]
        probs = self._apply_temperature(probs)
        return self._build_result(probs)

    def predict_averaged(self, mel_segments):
        """Average softmax vectors across multiple segments.

        Args:
            mel_segments: list of numpy arrays, each (1, 128, 431, 1)

        Returns:
            Same dict as predict(), with mean probabilities.
        """
        all_probs = []
        for mel in mel_segments:
            probs = self.model.predict(mel, verbose=0)[0]
            all_probs.append(probs)

        mean_probs = np.mean(all_probs, axis=0)
        mean_probs = self._apply_temperature(mean_probs)
        return self._build_result(mean_probs)

    def predict_averaged_smoothed(self, mel_segments):
        """Like predict_averaged but with temporal smoothing.

        Stores the current probability vector in a rolling buffer
        (last 5 predictions) and returns a result built from the
        mean across the buffer.  This prevents genre predictions
        from jumping rapidly between classes during live microphone
        input.
        """
        all_probs = []
        for mel in mel_segments:
            probs = self.model.predict(mel, verbose=0)[0]
            all_probs.append(probs)

        current_probs = np.mean(all_probs, axis=0)
        self._live_prob_history.append(current_probs)

        smoothed = np.mean(list(self._live_prob_history), axis=0)
        smoothed = self._apply_temperature(smoothed)
        return self._build_result(smoothed)

    def clear_live_history(self):
        """Reset the temporal smoothing buffer (call on live-mic stop)."""
        self._live_prob_history.clear()

    # ── Audio → mel segments ────────────────────────────────────

    @staticmethod
    def _audio_to_mel_segments(audio_path: str) -> list[np.ndarray]:
        """Load audio and return list of mel segment arrays (1,128,431,1).

        Adaptive segmentation based on audio length:
          ≥ 30 s : 3 × 10 s segments (original behaviour)
          10–30 s: as many 10 s segments as fit
          < 10 s : single segment using entire audio (padded to 431 bins)
        """
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        return GenreClassifier._signal_to_mel_segments(y)

    @staticmethod
    def _signal_to_mel_segments(y: np.ndarray) -> list[np.ndarray]:
        """Convert a raw signal array to mel segment batches (1,128,431,1).

        Same adaptive logic as _audio_to_mel_segments but works on a
        pre-loaded signal, useful when the recommender already has it.
        """
        seg_duration = DURATION // N_SEGMENTS          # 10 s per segment
        seg_samples = SR * seg_duration
        audio_len = len(y)

        if audio_len >= SR * DURATION:
            # Full 30 s track — use 3 × 10 s
            y = y[: SR * DURATION]
            n_segs = N_SEGMENTS
        elif audio_len >= seg_samples:
            # Between 10 s and 30 s — as many 10 s segments as available
            n_segs = audio_len // seg_samples
        else:
            # Short clip (< 10 s) — single segment from the whole clip
            n_segs = 1

        segments = []
        for i in range(n_segs):
            start = i * seg_samples
            seg = y[start: start + seg_samples]

            # Pad short segments so the mel shape is consistent
            if len(seg) < seg_samples:
                seg = np.pad(seg, (0, seg_samples - len(seg)))

            mel = librosa.feature.melspectrogram(
                y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Normalise (zero mean, unit std)
            mean, std = mel_db.mean(), mel_db.std()
            mel_db = (mel_db - mean) / std if std > 0 else mel_db - mean

            # Pad or trim time axis to TARGET_TIME_BINS
            if mel_db.shape[1] < TARGET_TIME_BINS:
                pad_w = TARGET_TIME_BINS - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_w)))
            else:
                mel_db = mel_db[:, :TARGET_TIME_BINS]

            segments.append(mel_db[np.newaxis, ..., np.newaxis])  # (1,128,431,1)

        return segments

    # ── Temperature scaling ─────────────────────────────────────

    def _apply_temperature(self, probs):
        """Re-scale probabilities via softmax(log(p) / T)."""
        if self.temperature == 1.0:
            return probs
        log_probs = np.log(np.clip(probs, 1e-12, None))
        scaled = log_probs / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()

    # ── Result builder ──────────────────────────────────────────

    def _build_result(self, probs, hybrid_threshold=0.10):
        """Convert probability vector into the standard result dict.

        Returns top-2 predicted genres.  If the gap between them is
        less than *hybrid_threshold* the label is ``Hybrid: A / B``.
        """
        fuzzy = {label: float(p) for label, p in zip(self.labels, probs)}

        sorted_idx = np.argsort(probs)[::-1]
        top_idx = int(sorted_idx[0])
        second_idx = int(sorted_idx[1])

        gap = float(probs[top_idx] - probs[second_idx])

        if gap < hybrid_threshold:
            top_genre = f"Hybrid: {self.labels[top_idx]} / {self.labels[second_idx]}"
        else:
            top_genre = self.labels[top_idx]

        return {
            "top_genre": top_genre,
            "confidence": float(probs[top_idx]),
            "top_2": [
                {"genre": self.labels[top_idx], "probability": float(probs[top_idx])},
                {"genre": self.labels[second_idx], "probability": float(probs[second_idx])},
            ],
            "fuzzy_memberships": fuzzy,
        }
