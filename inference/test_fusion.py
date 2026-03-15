"""
test_fusion.py — End-to-end fusion verification: genre + emotion → fused mood score.
Run from project root:  python -m inference.test_fusion
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from engine.genre_classifier import GenreClassifier
from engine.emotion_regressor import EmotionRegressor
from engine.fusion import FuzzyFusion

GENRE_MODEL = os.path.join(PROJECT_ROOT, "models", "best_genre_cnn_trans.keras")
EMOTION_MODEL = os.path.join(PROJECT_ROOT, "models", "emotion_hybrid_model.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("=" * 60)
    print("FUSION PIPELINE — INFERENCE VERIFICATION")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────
    print(f"\nLoading genre model : {os.path.basename(GENRE_MODEL)}")
    genre_model = tf.keras.models.load_model(GENRE_MODEL)
    genre_clf = GenreClassifier(model=genre_model)

    print(f"Loading emotion model: {os.path.basename(EMOTION_MODEL)}")
    emotion_model = tf.keras.models.load_model(EMOTION_MODEL)
    emotion_reg = EmotionRegressor(model=emotion_model)

    fusion = FuzzyFusion(w_genre=0.6, w_emotion=0.4)

    print(f"\n[DEBUG] Genre model  input: {genre_model.input_shape}")
    print(f"[DEBUG] Genre model output: {genre_model.output_shape}")
    if isinstance(emotion_model.input_shape, list):
        for i, s in enumerate(emotion_model.input_shape):
            print(f"[DEBUG] Emotion input {i}   : {s}")
    else:
        print(f"[DEBUG] Emotion input     : {emotion_model.input_shape}")
    print(f"[DEBUG] Emotion output    : {emotion_model.output_shape}")
    print(f"[DEBUG] Fusion weights    : genre={fusion.w_genre}, emotion={fusion.w_emotion}")
    print()

    # ── Run on test audio ────────────────────────────────────────
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not files:
        print("No audio files found in test_audio/")
        return

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        print(f"── {fname} ──")

        # Genre prediction with debug
        mel_segments = GenreClassifier._audio_to_mel_segments(path)
        print(f"  [DEBUG] Genre segments   : {len(mel_segments)} × {mel_segments[0].shape}")

        g = genre_clf.predict_file(path)
        all_probs = np.array(list(g["fuzzy_memberships"].values()))
        print(f"  [DEBUG] Genre probs      : [{', '.join(f'{p:.3f}' for p in all_probs)}]")
        print(f"  Genre           : {g['top_genre']} ({g['confidence']:.3f})")

        # Emotion prediction
        e = emotion_reg.predict_file(path)
        print(f"  Valence / Arousal: {e['valence']:.2f} / {e['arousal']:.2f}")
        print(f"  Mood             : {e['mood']}")

        # Fusion
        f = fusion.fuse(g, e)
        print(f"  [DEBUG] Genre score      : {f['genre_score']:.4f}")
        print(f"  [DEBUG] Emotion score    : {f['emotion_score']:.4f}")
        print(f"  ► Fused mood score: {f['mood_score']:.4f}")
        print(f"  ► Mood label      : {f['mood_label']}")
        print()

    print("=" * 60)
    print("FUSION VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
