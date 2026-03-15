"""
test_recommend.py — End-to-end recommendation verification.
Run from project root:  python -m inference.test_recommend
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from engine.model_registry import ModelRegistry
from engine.recommender import MusicRecommender

TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("=" * 60)
    print("RECOMMENDATION PIPELINE — INFERENCE VERIFICATION")
    print("=" * 60)

    # ── Load models via registry ─────────────────────────────────
    print("\nLoading models via ModelRegistry …")
    registry = ModelRegistry(PROJECT_ROOT)
    registry.warmup()

    genre_model = registry.genre_model
    print(f"\n[DEBUG] Genre model loaded  : {type(genre_model).__name__}")
    print(f"[DEBUG] Genre input shape   : {genre_model.input_shape}")
    print(f"[DEBUG] Genre output shape  : {genre_model.output_shape}")

    emotion_model = registry.emotion_model
    print(f"[DEBUG] Emotion model loaded: {type(emotion_model).__name__}")
    if isinstance(emotion_model.input_shape, list):
        for i, s in enumerate(emotion_model.input_shape):
            print(f"[DEBUG] Emotion input {i}    : {s}")
    else:
        print(f"[DEBUG] Emotion input shape : {emotion_model.input_shape}")
    print(f"[DEBUG] Emotion output shape: {emotion_model.output_shape}")

    # Validate genre model shape
    expected_in = (None, 128, 431, 1)
    expected_out = (None, 10)
    assert genre_model.input_shape == expected_in, \
        f"Genre input mismatch: {genre_model.input_shape} != {expected_in}"
    assert genre_model.output_shape == expected_out, \
        f"Genre output mismatch: {genre_model.output_shape} != {expected_out}"
    print("[DEBUG] ✓ Genre model I/O shapes verified")

    recommender = MusicRecommender(PROJECT_ROOT)
    print()

    # ── Find test files ──────────────────────────────────────────
    all_files = sorted(f for f in os.listdir(TEST_DIR)
                       if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")))

    if not all_files:
        print("No audio files found in test_audio/")
        return

    # Use up to 3 files for testing
    test_files = all_files[:3]

    for fname in test_files:
        path = os.path.join(TEST_DIR, fname)
        if not os.path.isfile(path):
            print(f"  SKIP: {fname} not found\n")
            continue

        print(f"══ {fname} ══")
        result = recommender.recommend(path, top_n=5)
        q = result["query"]
        g = q["genre"]
        e = q["emotion"]

        # Genre debug
        all_probs = np.array(list(g["fuzzy_memberships"].values()))
        print(f"  [DEBUG] Genre probs       : [{', '.join(f'{p:.3f}' for p in all_probs)}]")
        print(f"  Predicted genre   : {g['top_genre']} ({g['confidence']:.3f})")
        print(f"  Top 2             : {g['top_2'][0]['genre']} ({g['top_2'][0]['probability']:.3f})"
              f"  |  {g['top_2'][1]['genre']} ({g['top_2'][1]['probability']:.3f})")

        # Emotion debug
        print(f"  Valence / Arousal : {e['valence']:.2f} / {e['arousal']:.2f}")
        print(f"  Mood              : {e['mood']}")

        # Fused score if available
        if "fusion" in q:
            f = q["fusion"]
            print(f"  [DEBUG] Genre score       : {f.get('genre_score', 'N/A')}")
            print(f"  [DEBUG] Emotion score     : {f.get('emotion_score', 'N/A')}")
            print(f"  [DEBUG] Fused score       : {f.get('mood_score', 'N/A')}")

        # Recommendations
        print()
        print(f"  Top 5 Recommendations:")
        print(f"  {'#':>2}  {'Score':>6}  {'Genre':<22}  {'Song'}")
        print(f"  {'─'*2}  {'─'*6}  {'─'*22}  {'─'*30}")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  {i:>2}  {rec['score']:>6.4f}  {rec['genre']:<22}  "
                  f"{rec['title']} — {rec['artist']}")
        print()

    print("=" * 60)
    print("RECOMMENDATION VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
