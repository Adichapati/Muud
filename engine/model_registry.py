"""
model_registry.py
-----------------
Singleton that loads all ML models exactly once at application startup.
Provides pre-loaded model references so no component ever re-reads from disk.
"""

import os
import numpy as np
import tensorflow as tf


class ModelRegistry:
    """Thread-safe singleton holding all Keras models."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_root: str | None = None):
        if self._initialized:
            return
        if project_root is None:
            raise ValueError("project_root is required on first initialisation")

        models_dir = os.path.join(project_root, "models")

        print("[ModelRegistry] Loading genre model …")
        self.genre_model: tf.keras.Model = tf.keras.models.load_model(
            os.path.join(models_dir, "genre_fma_cnn.keras")
        )

        print("[ModelRegistry] Loading emotion model …")
        self.emotion_model: tf.keras.Model = tf.keras.models.load_model(
            os.path.join(models_dir, "emotion_hybrid_model.keras")
        )

        self._initialized = True
        print("[ModelRegistry] All models loaded ✓")

    # ── Warm-up ─────────────────────────────────────────────────

    def warmup(self):
        """
        Run a dummy forward pass through each model so TensorFlow
        compiles the graph before the user clicks anything.
        """
        print("[ModelRegistry] Warming up models …")

        # Genre model: expects (1, 128, 431, 1) — FMA mel spectrogram shape
        dummy_genre_mel = np.zeros((1, 128, 431, 1), dtype=np.float32)
        self.genre_model.predict(dummy_genre_mel, verbose=0)

        # Emotion model: expects (1, 128, 130, 1) mel + (1, 4) stats
        dummy_emo_mel = np.zeros((1, 128, 130, 1), dtype=np.float32)
        dummy_stats = np.zeros((1, 4), dtype=np.float32)
        self.emotion_model.predict([dummy_emo_mel, dummy_stats], verbose=0)

        print("[ModelRegistry] Warm-up complete ✓")
