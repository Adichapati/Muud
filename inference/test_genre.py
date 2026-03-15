"""
test_genre.py — Inference verification for the CNN + Transformer genre classifier.
Run from project root:  python -m inference.test_genre
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from engine.genre_classifier import GenreClassifier

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_genre_cnn_trans.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("=" * 60)
    print("GENRE CLASSIFIER — INFERENCE VERIFICATION")
    print("=" * 60)

    # ── Load model and inspect I/O shapes ────────────────────────
    print(f"\nLoading model: {os.path.basename(MODEL_PATH)}")
    model = tf.keras.models.load_model(MODEL_PATH)
    clf = GenreClassifier(model=model)

    print(f"\n[DEBUG] Model input shape : {model.input_shape}")
    print(f"[DEBUG] Model output shape: {model.output_shape}")

    expected_input = (None, 128, 431, 1)
    expected_output = (None, 10)
    assert model.input_shape == expected_input, \
        f"INPUT SHAPE MISMATCH: got {model.input_shape}, expected {expected_input}"
    assert model.output_shape == expected_output, \
        f"OUTPUT SHAPE MISMATCH: got {model.output_shape}, expected {expected_output}"
    print("[DEBUG] ✓ Input/output shapes match expected (batch,128,431,1) → (batch,10)")

    # ── Dummy forward pass ───────────────────────────────────────
    dummy = np.zeros((1, 128, 431, 1), dtype=np.float32)
    dummy_out = model.predict(dummy, verbose=0)
    print(f"[DEBUG] Dummy prediction shape: {dummy_out.shape}")
    print(f"[DEBUG] Dummy prediction sum  : {dummy_out.sum():.4f} (should be ~1.0)\n")

    # ── Run on test audio ────────────────────────────────────────
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not files:
        print("No audio files found in test_audio/")
        return

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        print(f"── {fname} ──")

        # Extract mel segments manually for debug info
        mel_segments = GenreClassifier._audio_to_mel_segments(path)
        print(f"  [DEBUG] Segments extracted : {len(mel_segments)}")
        for i, seg in enumerate(mel_segments):
            print(f"  [DEBUG] Segment {i} shape   : {seg.shape}")

        # Raw model probabilities for first segment
        raw_probs = model.predict(mel_segments[0], verbose=0)[0]
        print(f"  [DEBUG] Raw probs (seg 0)  : "
              f"[{', '.join(f'{p:.4f}' for p in raw_probs)}]")
        print(f"  [DEBUG] Probs sum          : {raw_probs.sum():.4f}")

        # Full prediction via classifier API
        result = clf.predict_averaged(mel_segments)
        print(f"  Predicted genre : {result['top_genre']}")
        print(f"  Confidence      : {result['confidence']:.4f}")
        print(f"  Top 2           : {result['top_2'][0]['genre']} ({result['top_2'][0]['probability']:.3f})"
              f"  |  {result['top_2'][1]['genre']} ({result['top_2'][1]['probability']:.3f})")

        # Show top-5 fuzzy memberships
        sorted_fuzzy = sorted(result["fuzzy_memberships"].items(),
                              key=lambda x: x[1], reverse=True)[:5]
        fuzzy_str = "  ".join(f"{g}: {p:.3f}" for g, p in sorted_fuzzy)
        print(f"  Fuzzy top 5     : {fuzzy_str}")
        print()

    print("=" * 60)
    print("GENRE VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
