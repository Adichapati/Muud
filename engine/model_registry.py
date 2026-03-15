"""
model_registry.py
-----------------
Singleton that loads all ML models exactly once at application startup.
Provides pre-loaded model references so no component ever re-reads from disk.
"""

import os
import shutil
import zipfile
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
        self.genre_model: tf.keras.Model = self._load_model(
            os.path.join(models_dir, "best_genre_cnn_trans.keras")
        )

        print("[ModelRegistry] Loading emotion model …")
        self.emotion_model: tf.keras.Model = self._load_model(
            os.path.join(models_dir, "emotion_hybrid_model.keras")
        )

        self._initialized = True
        print("[ModelRegistry] All models loaded ✓")

    @staticmethod
    def _load_model(path: str) -> tf.keras.Model:
        """Load a Keras model, handling both .keras (ZIP) and legacy HDF5 formats.

        Keras 3.x expects .keras files to be ZIP archives.  Models saved
        with Keras 2.x using HDF5 but given a .keras extension will fail
        the default loader, so we detect the format and fall back.
        """
        if zipfile.is_zipfile(path):
            # Native Keras 3 .keras format (ZIP)
            print(f"  → loading as .keras (ZIP): {os.path.basename(path)}")
            return tf.keras.models.load_model(path)
        else:
            # Legacy HDF5 format saved with .keras extension.
            # Keras 3.x enforces extension, so copy to a .h5 temp path.
            h5_path = path.rsplit(".", 1)[0] + ".h5"
            print(f"  → loading as legacy HDF5: {os.path.basename(path)}")
            try:
                shutil.copy2(path, h5_path)
                return tf.keras.models.load_model(h5_path, compile=False)
            finally:
                # Clean up the temporary .h5 copy
                if os.path.exists(h5_path):
                    try:
                        os.remove(h5_path)
                    except OSError:
                        pass

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
