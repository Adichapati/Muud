"""
test_emotion.py — Quick inference sanity check for the DEAM emotion regressor.
Run from project root:  python -m inference.test_emotion
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from engine.emotion_regressor import EmotionRegressor

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "emotion_hybrid_model.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("Loading emotion model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    reg = EmotionRegressor(model=model)
    print("Model loaded.\n")

    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not files:
        print("No audio files found in test_audio/")
        return

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        print(f"── {fname} ──")
        result = reg.predict_file(path)
        print(f"  Valence : {result['valence']:.3f}")
        print(f"  Arousal : {result['arousal']:.3f}")
        print(f"  Mood    : {result['mood']}")
        print()


if __name__ == "__main__":
    main()
