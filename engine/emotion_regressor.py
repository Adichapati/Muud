"""
emotion_regressor.py
---------------------
Loads the hybrid emotion model (CNN + handcrafted features).
Predicts continuous Valence and Arousal values.
"""

import numpy as np
import tensorflow as tf


class EmotionRegressor:
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

    def predict(self, mel, stats):
        """
        Predict valence and arousal from mel spectrogram + handcrafted stats.

        Args:
            mel:   numpy array of shape (1, 128, T, 1)
            stats: numpy array of shape (1, 4)

        Returns:
            dict with:
                - "valence": float (1–9 scale, DEAM)
                - "arousal": float (1–9 scale, DEAM)
                - "mood_label": str  (human-readable quadrant)
        """
        prediction = self.model.predict([mel, stats], verbose=0)[0]
        valence = float(prediction[0])
        arousal = float(prediction[1])

        mood = self._mood_quadrant(valence, arousal)

        return {
            "valence": valence,
            "arousal": arousal,
            "mood_label": mood,
        }

    def predict_averaged(self, mel_segments, stats_segments):
        """
        Average valence/arousal predictions across multiple segments.

        Args:
            mel_segments:   list of (1, 128, T, 1) arrays
            stats_segments: list of (1, 4) arrays  (same length)

        Returns:
            Same dict as predict(), with mean V/A and mood from the mean.
        """
        vals, aros = [], []
        for mel, stats in zip(mel_segments, stats_segments):
            pred = self.model.predict([mel, stats], verbose=0)[0]
            vals.append(float(pred[0]))
            aros.append(float(pred[1]))

        valence = float(np.mean(vals))
        arousal = float(np.mean(aros))
        mood = self._mood_quadrant(valence, arousal)

        return {
            "valence": valence,
            "arousal": arousal,
            "mood_label": mood,
        }

    @staticmethod
    def _mood_quadrant(valence, arousal, midpoint=5.0, neutral_radius=0.5):
        """
        Map valence/arousal to Russell's circumplex quadrant.

        A neutral buffer zone is applied first: if both dimensions fall
        within *neutral_radius* of the midpoint the mood is 'Neutral / Balanced'.
        Otherwise the standard four-quadrant mapping is used.
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
