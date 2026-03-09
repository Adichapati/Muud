"""
test_recommend.py — End-to-end recommendation test.
Run from project root:  python -m inference.test_recommend
"""

import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.model_registry import ModelRegistry
from engine.recommender import MusicRecommender

TEST_DIR = os.path.join(PROJECT_ROOT, "test_audio")


def main():
    print("Loading models …")
    registry = ModelRegistry(PROJECT_ROOT)
    registry.warmup()
    print("Models ready.\n")

    recommender = MusicRecommender(PROJECT_ROOT)
    print()

    # Pick 3 representative files
    test_files = ["classical.00005.wav", "rock.00030.wav", "blues_test.wav"]

    for fname in test_files:
        path = os.path.join(TEST_DIR, fname)
        if not os.path.isfile(path):
            print(f"  SKIP: {fname} not found\n")
            continue

        result = recommender.recommend(path, top_n=5)
        q = result["query"]
        g = q["genre"]
        e = q["emotion"]

        print(f"══ {fname} ══")
        print(f"  Predicted genre : {g['top_genre']} ({g['confidence']:.3f})")
        print(f"  Emotion V/A     : {e['valence']:.2f} / {e['arousal']:.2f}")
        print(f"  Mood            : {e['mood_label']}")
        print()
        print(f"  Top 5 Recommendations:")
        print(f"  {'#':>2}  {'Score':>6}  {'Genre':<22}  {'Song'}")
        print(f"  {'─'*2}  {'─'*6}  {'─'*22}  {'─'*30}")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  {i:>2}  {rec['score']:>6.4f}  {rec['genre']:<22}  "
                  f"{rec['title']} — {rec['artist']}")
        print()


if __name__ == "__main__":
    main()
