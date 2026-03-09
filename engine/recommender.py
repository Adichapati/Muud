"""
recommender.py
--------------
High-level orchestration: analyze audio, query CSV song database,
rank by genre + emotion similarity, return recommendations.
"""

import csv
import os
import numpy as np

from engine.feature_extraction import (
    load_audio, extract_mel_segments,
    extract_handcrafted_segments,
)
from engine.genre_classifier import GenreClassifier
from engine.emotion_regressor import EmotionRegressor
from engine.model_registry import ModelRegistry
from engine.fusion import FuzzyFusion, fused_score


# ── Genre similarity helpers ───────────────────────────────────
# Genres considered "similar" for partial-match scoring (0.5).
# ── Genre similarity matrix ────────────────────────────────────
# Pairwise scores for related genres (symmetric — stored once per pair).
# Unlisted pairs default to 0.0.

_GENRE_SIM_MATRIX = {
    ("Classical",    "Instrumental"):        0.8,
    ("Classical",    "Old-Time / Historic"):  0.6,
    ("Electronic",   "Experimental"):        0.7,
    ("Electronic",   "Pop"):                 0.5,
    ("Electronic",   "Hip-Hop"):             0.4,
    ("Experimental", "Instrumental"):        0.6,
    ("Folk",         "Old-Time / Historic"):  0.8,
    ("Folk",         "International"):       0.7,
    ("Folk",         "Rock"):                0.4,
    ("Hip-Hop",      "Pop"):                 0.6,
    ("International","Old-Time / Historic"):  0.6,
    ("Rock",         "Pop"):                 0.7,
    ("Rock",         "Electronic"):          0.4,
}

# Build a fast lookup (both orderings) at import time
_SIM_LOOKUP: dict[tuple[str, str], float] = {}
for (a, b), score in _GENRE_SIM_MATRIX.items():
    _SIM_LOOKUP[(a, b)] = score
    _SIM_LOOKUP[(b, a)] = score


def _genre_similarity(predicted_genre: str, candidate_genre: str) -> float:
    """Return similarity in [0, 1] between predicted and candidate genre.

    - Exact match → 1.0
    - Matrix match → value from _GENRE_SIM_MATRIX (e.g. 0.8, 0.7 …)
    - No match → 0.0

    ``predicted_genre`` may be a hybrid label like
    "Hybrid: Instrumental / Experimental"  — the best match across
    both parts is returned.
    """
    cand = candidate_genre.strip()

    # Extract genre names from possible hybrid label
    if predicted_genre.startswith("Hybrid:"):
        parts = [g.strip() for g in predicted_genre.replace("Hybrid:", "").split("/")]
    else:
        parts = [predicted_genre.strip()]

    best = 0.0
    for p in parts:
        if p == cand:
            return 1.0                          # exact match — can't do better
        score = _SIM_LOOKUP.get((p, cand), 0.0)
        if score > best:
            best = score

    return best


def _emotion_similarity(q_val, q_aro, c_val, c_aro, max_dist=None):
    """Inverse-Euclidean emotion similarity in [0, 1].

    max_dist defaults to the DEAM diagonal √(8² + 8²) ≈ 11.31.
    """
    if max_dist is None:
        max_dist = np.sqrt(8**2 + 8**2)
    dist = np.sqrt((q_val - c_val) ** 2 + (q_aro - c_aro) ** 2)
    return float(1.0 - min(dist / max_dist, 1.0))


# ── CSV loader ─────────────────────────────────────────────────

def load_song_csv(csv_path: str) -> list[dict]:
    """Load song database from CSV (columns: song, artist, genre, valence, arousal)."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "title":   row["song"].strip(),
                "artist":  row["artist"].strip(),
                "genre":   row["genre"].strip(),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
            })
    return songs


class MusicRecommender:
    """
    Full inference pipeline:
      1. Extract features from input audio
      2. Predict genre (fuzzy memberships) + emotion (valence, arousal)
      3. Fuse scores, rank CSV song database
      4. Return top-N recommendations
    """

    def __init__(self, project_root, song_db=None, genre_segments=5):
        self.project_root = project_root
        self.genre_segments = genre_segments

        # Obtain the singleton model registry (must already be initialised)
        registry = ModelRegistry(project_root)

        # Build classifiers from pre-loaded models — no disk I/O here
        self.genre_clf = GenreClassifier(model=registry.genre_model)
        self.emotion_reg = EmotionRegressor(model=registry.emotion_model)
        self.fusion = FuzzyFusion(w_genre=0.6, w_emotion=0.4)

        # Analysis cache: avoids re-extracting features + re-predicting
        self._analysis_cache = {}   # file_path → analysis dict

        # Song database — load from CSV
        if song_db is not None:
            self.song_db = song_db
        else:
            csv_path = os.path.join(project_root, "data", "song_db.csv")
            if os.path.isfile(csv_path):
                self.song_db = load_song_csv(csv_path)
                print(f"[Recommender] Loaded {len(self.song_db)} songs from song_db.csv")
            else:
                self.song_db = []
                print("[Recommender] WARNING: data/song_db.csv not found — empty database")

    # ── Core Analysis ───────────────────────────────────────────

    def analyze(self, file_path):
        """
        Analyze a single audio file.  Results are cached.

        Genre uses the classifier's own 10-s segmentation (431 time bins).
        Emotion uses 3-s segments with handcrafted stats.

        Returns:
            dict with "genre", "emotion", and "fusion" sub-dicts.
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in self._analysis_cache:
            return self._analysis_cache[abs_path]

        # Genre: use the classifier's own segmentation (handles short clips)
        genre_result = self.genre_clf.predict_file(file_path)

        # Emotion: 3-s segments with handcrafted stats (matches emotion model)
        signal, sr = load_audio(file_path)
        mel_segments = extract_mel_segments(signal, sr,
                                            max_segments=self.genre_segments)
        stats_segments = extract_handcrafted_segments(signal, sr,
                                                      max_segments=self.genre_segments)
        n = min(len(mel_segments), len(stats_segments))
        emotion_result = self.emotion_reg.predict_averaged(
            mel_segments[:n], stats_segments[:n],
        )

        # Fusion
        fusion_result = self.fusion.fuse(genre_result, emotion_result)

        result = {
            "file": os.path.basename(file_path),
            "genre": genre_result,
            "emotion": emotion_result,
            "fusion": fusion_result,
        }
        self._analysis_cache[abs_path] = result
        return result

    def invalidate_cache(self, file_path=None):
        """Drop cached analysis for *file_path*, or the entire cache."""
        if file_path is None:
            self._analysis_cache.clear()
        else:
            self._analysis_cache.pop(os.path.abspath(file_path), None)

    def analyze_signal(self, signal, sr=22050, label="live"):
        """Analyze a pre-loaded audio signal (no file I/O).

        Used by the live-mic feature.  Not cached.

        Returns the same dict structure as analyze().
        """
        from engine.genre_classifier import GenreClassifier as _GC

        # Genre — use the classifier's adaptive segmentation on the signal
        mel_segs = _GC._signal_to_mel_segments(signal)
        genre_result = self.genre_clf.predict_averaged_smoothed(mel_segs)

        # Emotion — 3-s segments
        mel_segments = extract_mel_segments(signal, sr,
                                            max_segments=self.genre_segments)
        stats_segments = extract_handcrafted_segments(signal, sr,
                                                      max_segments=self.genre_segments)
        n = min(len(mel_segments), len(stats_segments))
        emotion_result = self.emotion_reg.predict_averaged(
            mel_segments[:n], stats_segments[:n],
        )

        fusion_result = self.fusion.fuse(genre_result, emotion_result)

        return {
            "file": label,
            "genre": genre_result,
            "emotion": emotion_result,
            "fusion": fusion_result,
        }

    # ── Recommendation ──────────────────────────────────────────

    def recommend(self, file_path, top_n=5, w_genre=0.7, w_emotion=0.3):
        """
        Analyze input audio and return top-N song recommendations.

        Scoring per candidate:
            genre_sim    = 1.0 (exact match), 0.5 (similar), 0.0 (other)
            emotion_sim  = 1 − (euclidean_dist / max_dist)
            score        = w_genre * genre_sim + w_emotion * emotion_sim

        Returns:
            { "query": analysis_dict,
              "recommendations": [ {title, artist, genre, valence, arousal, score}, … ] }
        """
        analysis = self.analyze(file_path)

        q_genre = analysis["genre"]["top_genre"]
        q_val = analysis["emotion"]["valence"]
        q_aro = analysis["emotion"]["arousal"]

        scored = []
        for song in self.song_db:
            g_sim = _genre_similarity(q_genre, song["genre"])
            e_sim = _emotion_similarity(q_val, q_aro,
                                        song["valence"], song["arousal"])
            score = w_genre * g_sim + w_emotion * e_sim
            scored.append({**song, "score": round(score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": analysis,
            "recommendations": scored[:top_n],
        }

    # ── Database Management ─────────────────────────────────────

    def load_song_db(self, csv_path):
        """Reload song database from a CSV file."""
        self.song_db = load_song_csv(csv_path)

    def reload_default_db(self):
        """Reload the default data/song_db.csv."""
        csv_path = os.path.join(self.project_root, "data", "song_db.csv")
        if os.path.isfile(csv_path):
            self.song_db = load_song_csv(csv_path)

