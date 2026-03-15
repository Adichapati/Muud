"""
test_emotion.py — Inference verification for the DEAM emotion regressor.
Run from project root:  python -m inference.test_emotion
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from engine.emotion_regressor import EmotionRegressor
from engine.feature_extraction import load_audio, extract_mel, extract_handcrafted

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "emotion_hybrid_model.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("=" * 60)
    print("EMOTION REGRESSOR — INFERENCE VERIFICATION")
    print("=" * 60)

    # ── Load model and inspect I/O ───────────────────────────────
    print(f"\nLoading model: {os.path.basename(MODEL_PATH)}")
    model = tf.keras.models.load_model(MODEL_PATH)
    reg = EmotionRegressor(model=model)

    # Emotion model has two inputs: mel + stats
    if isinstance(model.input_shape, list):
        for i, shape in enumerate(model.input_shape):
            print(f"[DEBUG] Model input {i} shape : {shape}")
    else:
        print(f"[DEBUG] Model input shape : {model.input_shape}")
    print(f"[DEBUG] Model output shape: {model.output_shape}")

    # ── Dummy forward pass ───────────────────────────────────────
    dummy_mel = np.zeros((1, 128, 130, 1), dtype=np.float32)
    dummy_stats = np.zeros((1, 4), dtype=np.float32)
    dummy_out = model.predict([dummy_mel, dummy_stats], verbose=0)
    print(f"[DEBUG] Dummy output shape: {dummy_out.shape}")
    print(f"[DEBUG] Dummy V/A values  : {dummy_out[0]}\n")

    # ── Run on test audio ────────────────────────────────────────
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not files:
        print("No audio files found in test_audio/")
        return

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        print(f"── {fname} ──")

        # Extract features manually for debug
        signal, sr = load_audio(path)
        mel = extract_mel(signal, sr)
        stats = extract_handcrafted(signal, sr)
        print(f"  [DEBUG] Mel shape    : {mel.shape}")
        print(f"  [DEBUG] Stats shape  : {stats.shape}")
        print(f"  [DEBUG] Stats values : tempo={stats[0,0]:.1f}  centroid={stats[0,1]:.1f}  "
              f"rms={stats[0,2]:.4f}  zcr={stats[0,3]:.4f}")

        # Raw model output (before post-processing)
        raw_out = model.predict([mel, stats], verbose=0)[0]
        print(f"  [DEBUG] Raw V/A      : valence={raw_out[0]:.3f}  arousal={raw_out[1]:.3f}")

        # Full prediction via regressor API (with post-processing)
        result = reg.predict_file(path)
        print(f"  Valence (final)  : {result['valence']:.3f}")
        print(f"  Arousal (final)  : {result['arousal']:.3f}")
        print(f"  Mood             : {result['mood']}")
        print()

    print("=" * 60)
    print("EMOTION VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
