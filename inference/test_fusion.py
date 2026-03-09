"""
test_fusion.py — End-to-end fusion test: genre + emotion → fused mood score.
Run from project root:  python -m inference.test_fusion
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from engine.genre_classifier import GenreClassifier
from engine.emotion_regressor import EmotionRegressor
from engine.fusion import FuzzyFusion

GENRE_MODEL = os.path.join(PROJECT_ROOT, "models", "genre_fma_cnn.keras")
EMOTION_MODEL = os.path.join(PROJECT_ROOT, "models", "emotion_hybrid_model.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("Loading models …")
    genre_clf = GenreClassifier(model=tf.keras.models.load_model(GENRE_MODEL))
    emotion_reg = EmotionRegressor(model=tf.keras.models.load_model(EMOTION_MODEL))
    fusion = FuzzyFusion(w_genre=0.6, w_emotion=0.4)
    print("Ready.\n")

    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        g = genre_clf.predict_file(path)
        e = emotion_reg.predict_file(path)
        f = fusion.fuse(g, e)

        print(f"── {fname} ──")
        print(f"  Genre       : {f['top_genre']} ({f['genre_confidence']:.3f})")
        print(f"  V/A         : {f['valence']:.2f} / {f['arousal']:.2f}")
        print(f"  Mood        : {f['mood_label']}")
        print(f"  Genre score : {f['genre_score']:.4f}")
        print(f"  Emotion score: {f['emotion_score']:.4f}")
        print(f"  Fused score : {f['mood_score']:.4f}")
        print()


if __name__ == "__main__":
    main()
