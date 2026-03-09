"""
emotion_regressor.py
---------------------
Loads the hybrid emotion model (CNN + handcrafted features).
Predicts continuous Valence and Arousal values and maps to mood quadrant.
"""

import numpy as np
import librosa
import tensorflow as tf

# ── Audio constants (must match emotion training pipeline) ──────
SR = 22_050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_DURATION = 3          # seconds per segment
SAMPLES_PER_SEGMENT = SR * SEGMENT_DURATION
MAX_SEGMENTS = 5


class EmotionRegressor:
    """DEAM emotion regressor — predicts Valence & Arousal (1–9 scale)."""

    def __init__(self, model_path=None, *, model=None):
        """
        Args:
            model_path: Path to .keras file (loads from disk).
            model:      Pre-loaded tf.keras.Model (preferred — avoids reload).
        Supply *one* of the two.  ``model`` takes priority.
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Provide model_path or model")

    # ── High-level: audio file → result ─────────────────────────

    def predict_file(self, audio_path: str) -> dict:
        """End-to-end prediction from an audio file path.

        Loads audio, extracts mel + handcrafted segments,
        averages V/A across segments, and returns the result dict.
        """
        mel_segments, stats_segments = self._audio_to_segments(audio_path)
        return self.predict_averaged(mel_segments, stats_segments)

    # ── Segment-level API (used by recommender / desktop app) ───

    def predict(self, mel, stats):
        """
        Predict valence and arousal from mel spectrogram + handcrafted stats.

        Args:
            mel:   numpy array of shape (1, 128, T, 1)
            stats: numpy array of shape (1, 4)

        Returns:
            dict with valence, arousal, mood, mood_label
        """
        prediction = self.model.predict([mel, stats], verbose=0)[0]
        valence = float(prediction[0])
        arousal = float(prediction[1])
        rms_mean = float(stats[0, 2])  # stats layout: [tempo, centroid, rms, zcr]
        return self._build_result(valence, arousal, rms_mean)

    def predict_averaged(self, mel_segments, stats_segments):
        """
        Average valence/arousal predictions across multiple segments.

        Args:
            mel_segments:   list of (1, 128, T, 1) arrays
            stats_segments: list of (1, 4) arrays  (same length)

        Returns:
            Same dict as predict(), with mean V/A and mood from the mean.
        """
        vals, aros, rms_vals = [], [], []
        for mel, stats in zip(mel_segments, stats_segments):
            pred = self.model.predict([mel, stats], verbose=0)[0]
            vals.append(float(pred[0]))
            aros.append(float(pred[1]))
            rms_vals.append(float(stats[0, 2]))

        valence = float(np.mean(vals))
        arousal = float(np.mean(aros))
        rms_mean = float(np.mean(rms_vals))
        return self._build_result(valence, arousal, rms_mean)

    # ── Audio → segments ────────────────────────────────────────

    @staticmethod
    def _audio_to_segments(audio_path: str):
        """Load audio and return (mel_segments, stats_segments).

        mel_segments:  list of (1, 128, T, 1) arrays
        stats_segments: list of (1, 4) arrays  — [tempo, centroid, rms, zcr]
        """
        y, _ = librosa.load(audio_path, sr=SR, mono=True)

        mel_segments = []
        stats_segments = []

        for i in range(MAX_SEGMENTS):
            start = i * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            if start >= len(y):
                break
            chunk = y[start:end]
            if len(chunk) < SAMPLES_PER_SEGMENT:
                chunk = np.pad(chunk, (0, SAMPLES_PER_SEGMENT - len(chunk)))

            # ── Mel spectrogram ──
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=SR, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mean, std = mel_db.mean(), mel_db.std()
            mel_db = (mel_db - mean) / std if std > 0 else mel_db - mean
            mel_segments.append(mel_db[np.newaxis, ..., np.newaxis])

            # ── Handcrafted stats ──
            tempo, _ = librosa.beat.beat_track(y=chunk, sr=SR)
            tempo = float(np.squeeze(tempo))
            centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=chunk, sr=SR)))
            rms = float(np.mean(librosa.feature.rms(y=chunk)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
            stats_segments.append(
                np.array([[tempo, centroid, rms, zcr]], dtype=np.float32))

        # Fallback: if audio is too short, ensure at least one segment
        if not mel_segments:
            chunk = np.pad(y, (0, max(0, SAMPLES_PER_SEGMENT - len(y))))
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=SR, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mean, std = mel_db.mean(), mel_db.std()
            mel_db = (mel_db - mean) / std if std > 0 else mel_db - mean
            mel_segments.append(mel_db[np.newaxis, ..., np.newaxis])

            tempo, _ = librosa.beat.beat_track(y=chunk, sr=SR)
            tempo = float(np.squeeze(tempo))
            centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=chunk, sr=SR)))
            rms = float(np.mean(librosa.feature.rms(y=chunk)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
            stats_segments.append(
                np.array([[tempo, centroid, rms, zcr]], dtype=np.float32))

        return mel_segments, stats_segments

    # ── Result builder ──────────────────────────────────────────

    @staticmethod
    def _spread_va(valence, arousal):
        """Linearly scale V/A away from centre to use the full 1–9 range.

        The DEAM model tends to predict values clustered near 4.5–5.5.
        This stretches them outward so the V-A plot is more dynamic.
        """
        valence = (valence - 4.5) * 2.0 + 5.0
        arousal = (arousal - 4.5) * 2.0 + 5.0
        valence = float(np.clip(valence, 1.0, 9.0))
        arousal = float(np.clip(arousal, 1.0, 9.0))
        return valence, arousal

    @staticmethod
    def _build_result(valence, arousal, rms_mean=0.0):
        """Build the standard result dict from V/A values.

        If *rms_mean* is provided the raw arousal is boosted by
        ``2.0 * rms_mean`` before the spread transform, helping
        energetic music (rock, electronic) land in the upper
        quadrants of the V-A space.
        """
        # Energy-based arousal boost
        arousal = arousal + 2.0 * rms_mean
        arousal = float(np.clip(arousal, 1.0, 9.0))

        valence, arousal = EmotionRegressor._spread_va(valence, arousal)
        mood = EmotionRegressor._mood_quadrant(valence, arousal)
        return {
            "valence": valence,
            "arousal": arousal,
            "mood": mood,
            "mood_label": mood,     # backward-compat alias
        }

    # ── Mood quadrant mapping ───────────────────────────────────

    @staticmethod
    def _mood_quadrant(valence, arousal, midpoint=5.0, neutral_radius=1.0):
        """
        Map valence/arousal to Russell's circumplex quadrant.

        A neutral buffer zone of *neutral_radius* (default 1.0) is applied
        first: if both dimensions fall within that distance of the midpoint
        the mood is 'Neutral / Balanced'.  This prevents small V/A
        variations near centre from being classified as Sad/Calm or
        Angry/Intense.  Only clearly deviating values reach a quadrant.
        """
        if (abs(valence - midpoint) < neutral_radius
                and abs(arousal - midpoint) < neutral_radius):
            return "Neutral / Balanced"

        if valence >= midpoint and arousal >= midpoint:
            return "Happy / Energetic"
        elif valence >= midpoint and arousal < midpoint:
            return "Happy / Calm"
        elif valence < midpoint and arousal >= midpoint:
            return "Angry / Intense"
        else:
            return "Sad / Calm"
