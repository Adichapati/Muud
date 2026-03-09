"""
test_genre.py — Quick inference sanity check for the FMA genre classifier.
Run from project root:  python -m inference.test_genre
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from engine.genre_classifier import GenreClassifier

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "genre_fma_cnn.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("Loading genre model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    clf = GenreClassifier(model=model)
    print("Model loaded.\n")

    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not files:
        print("No audio files found in test_audio/")
        return

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        print(f"── {fname} ──")
        result = clf.predict_file(path)
        print(f"  Top genre : {result['top_genre']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Top 2     : {result['top_2'][0]['genre']} ({result['top_2'][0]['probability']:.3f})"
              f"  |  {result['top_2'][1]['genre']} ({result['top_2'][1]['probability']:.3f})")
        # Show top-5 fuzzy memberships
        sorted_fuzzy = sorted(result["fuzzy_memberships"].items(),
                              key=lambda x: x[1], reverse=True)[:5]
        fuzzy_str = "  ".join(f"{g}: {p:.3f}" for g, p in sorted_fuzzy)
        print(f"  Fuzzy top5: {fuzzy_str}")
        print()


if __name__ == "__main__":
    main()
