"""
genre_classifier.py
--------------------
Loads the genre CNN model and provides genre prediction.
Softmax outputs are treated as fuzzy membership degrees.
"""

import os
import numpy as np
import tensorflow as tf

# Genre labels (must match training label order)
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class GenreClassifier:
    def __init__(self, model_path=None, *, model=None, temperature=1.0):
        """
        Args:
            model_path:  Path to .keras file (loads from disk).
            model:       Pre-loaded tf.keras.Model (preferred — avoids reload).
            temperature: Softmax temperature scaling (default 1.0 = no change).
                         Values < 1 sharpen the distribution (more confident),
                         values > 1 soften it (more uniform).
        Supply *one* of model_path / model.  ``model`` takes priority.
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Provide model_path or model")

        self.temperature = temperature

    def predict(self, mel):
        """
        Predict genre from a single Mel spectrogram.

        Args:
            mel: numpy array of shape (1, 128, T, 1)

        Returns:
            dict with:
                - "top_genre": str
                - "confidence": float
                - "fuzzy_memberships": dict[str, float]  (all genres)
        """
        probs = self.model.predict(mel, verbose=0)[0]
        probs = self._apply_temperature(probs)
        return self._build_result(probs)

    def predict_averaged(self, mel_segments):
        """
        Predict genre by averaging softmax vectors across multiple segments.

        Args:
            mel_segments: list of numpy arrays, each (1, 128, T, 1)

        Returns:
            Same dict format as predict(), but probabilities are the
            mean across all segments — giving a more robust estimate.
        """
        all_probs = []
        for mel in mel_segments:
            probs = self.model.predict(mel, verbose=0)[0]
            all_probs.append(probs)

        mean_probs = np.mean(all_probs, axis=0)
        mean_probs = self._apply_temperature(mean_probs)
        return self._build_result(mean_probs)

    # ── Temperature scaling ─────────────────────────────────────

    def _apply_temperature(self, probs):
        """Re-scale probabilities via  softmax(log(p) / T).

        At T=1.0 this is an identity (within float precision).
        T<1 sharpens, T>1 softens the distribution.
        """
        if self.temperature == 1.0:
            return probs
        # Clip to avoid log(0)
        log_probs = np.log(np.clip(probs, 1e-12, None))
        scaled = log_probs / self.temperature
        # Numerically stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()

    @staticmethod
    def _build_result(probs, hybrid_threshold=0.10):
        """Convert a probability vector into the standard result dict.

        If the gap between the top two genres is less than *hybrid_threshold*,
        the label becomes ``"Hybrid: genre1 / genre2"`` to signal ambiguity.
        The full fuzzy membership vector is always returned unchanged.
        """
        fuzzy = {label: float(prob) for label, prob in zip(GENRE_LABELS, probs)}

        sorted_idx = np.argsort(probs)[::-1]  # descending
        top_idx = int(sorted_idx[0])
        second_idx = int(sorted_idx[1])

        gap = float(probs[top_idx] - probs[second_idx])

        if gap < hybrid_threshold:
            top_genre = f"Hybrid: {GENRE_LABELS[top_idx]} / {GENRE_LABELS[second_idx]}"
        else:
            top_genre = GENRE_LABELS[top_idx]

        return {
            "top_genre": top_genre,
            "confidence": float(probs[top_idx]),
            "fuzzy_memberships": fuzzy,
        }
