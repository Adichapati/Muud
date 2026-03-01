"""
recommender.py
--------------
High-level orchestration: analyze audio, query song database,
rank by fused score, return recommendations.
"""

import os
import json
import numpy as np

from engine.feature_extraction import (
    load_audio, extract_mel, extract_mel_segments,
    extract_handcrafted, extract_handcrafted_segments,
)
from engine.genre_classifier import GenreClassifier
from engine.emotion_regressor import EmotionRegressor
from engine.model_registry import ModelRegistry
from engine.fusion import fused_score


# ── Default song database ──────────────────────────────────────
# Each entry: { "title", "artist", "genre", "valence", "arousal" }
# This is a starter set — expand by pre-analyzing your music library.

DEFAULT_SONG_DB = [
    {"title": "Comfortably Numb",   "artist": "Pink Floyd",       "genre": "rock",      "valence": 3.5, "arousal": 3.0},
    {"title": "Back in Black",      "artist": "AC/DC",            "genre": "rock",      "valence": 6.5, "arousal": 7.5},
    {"title": "Nuvole Bianche",     "artist": "Ludovico Einaudi", "genre": "classical", "valence": 4.0, "arousal": 2.5},
    {"title": "Spring - Vivaldi",   "artist": "Vivaldi",          "genre": "classical", "valence": 7.0, "arousal": 6.5},
    {"title": "So What",            "artist": "Miles Davis",      "genre": "jazz",      "valence": 5.5, "arousal": 3.5},
    {"title": "Take Five",          "artist": "Dave Brubeck",     "genre": "jazz",      "valence": 6.0, "arousal": 4.5},
    {"title": "The Thrill Is Gone", "artist": "B.B. King",        "genre": "blues",     "valence": 3.0, "arousal": 3.5},
    {"title": "Cross Road Blues",   "artist": "Robert Johnson",   "genre": "blues",     "valence": 3.5, "arousal": 5.0},
    {"title": "Blinding Lights",    "artist": "The Weeknd",       "genre": "pop",       "valence": 6.5, "arousal": 7.0},
    {"title": "Someone Like You",   "artist": "Adele",            "genre": "pop",       "valence": 3.0, "arousal": 2.5},
    {"title": "Stayin Alive",       "artist": "Bee Gees",         "genre": "disco",     "valence": 7.5, "arousal": 7.5},
    {"title": "Rapper's Delight",   "artist": "Sugarhill Gang",   "genre": "hiphop",    "valence": 7.0, "arousal": 6.5},
    {"title": "Lose Yourself",      "artist": "Eminem",           "genre": "hiphop",    "valence": 4.5, "arousal": 8.0},
    {"title": "No Woman No Cry",    "artist": "Bob Marley",       "genre": "reggae",    "valence": 5.5, "arousal": 3.0},
    {"title": "Master of Puppets",  "artist": "Metallica",        "genre": "metal",     "valence": 4.0, "arousal": 8.5},
    {"title": "Ace of Spades",      "artist": "Motorhead",        "genre": "metal",     "valence": 5.5, "arousal": 8.5},
    {"title": "Ring of Fire",       "artist": "Johnny Cash",      "genre": "country",   "valence": 6.0, "arousal": 5.0},
    {"title": "Jolene",             "artist": "Dolly Parton",     "genre": "country",   "valence": 4.0, "arousal": 4.5},
]


class MusicRecommender:
    """
    Full inference pipeline:
      1. Extract features from input audio
      2. Predict genre (fuzzy memberships) + emotion (valence, arousal)
      3. Rank song database by fused similarity score
      4. Return top-N recommendations
    """

    def __init__(self, project_root, song_db=None, genre_segments=5):
        self.project_root = project_root
        self.genre_segments = genre_segments   # max segments for genre averaging

        # Obtain the singleton model registry (must already be initialised)
        registry = ModelRegistry(project_root)

        # Build classifiers from pre-loaded models — no disk I/O here
        self.genre_clf = GenreClassifier(model=registry.genre_model)
        self.emotion_reg = EmotionRegressor(model=registry.emotion_model)

        # Analysis cache: avoids re-extracting features + re-predicting
        # when the user clicks RECOMMEND after ANALYZE on the same file.
        self._analysis_cache = {}   # file_path → analysis dict

        # Song database
        self.song_db = song_db or DEFAULT_SONG_DB

    # ── Core Analysis ───────────────────────────────────────────

    def analyze(self, file_path):
        """
        Analyze a single audio file.  Results are cached so repeated
        calls on the same path skip feature extraction and inference.

        Returns:
            dict with "genre" and "emotion" sub-dicts.
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in self._analysis_cache:
            return self._analysis_cache[abs_path]

        signal, sr = load_audio(file_path)

        # Genre: average predictions across multiple 3-s segments
        mel_segments = extract_mel_segments(signal, sr,
                                            max_segments=self.genre_segments)
        genre_result = self.genre_clf.predict_averaged(mel_segments)

        # Emotion: average V/A across the same segments
        stats_segments = extract_handcrafted_segments(signal, sr,
                                                      max_segments=self.genre_segments)
        # Align lengths (mel and stats may differ by one if audio length is borderline)
        n = min(len(mel_segments), len(stats_segments))
        emotion_result = self.emotion_reg.predict_averaged(
            mel_segments[:n], stats_segments[:n],
        )

        result = {
            "file": os.path.basename(file_path),
            "genre": genre_result,
            "emotion": emotion_result,
        }
        self._analysis_cache[abs_path] = result
        return result

    def invalidate_cache(self, file_path=None):
        """Drop cached analysis for *file_path*, or the entire cache."""
        if file_path is None:
            self._analysis_cache.clear()
        else:
            self._analysis_cache.pop(os.path.abspath(file_path), None)

    # ── Recommendation ──────────────────────────────────────────

    def recommend(self, file_path, top_n=5, w_genre=0.4, w_emotion=0.6):
        """
        Analyze input audio and return top-N song recommendations
        ranked by weighted fuzzy fusion score.
        """
        analysis = self.analyze(file_path)

        scored = []
        for song in self.song_db:
            score = fused_score(analysis, song, w_genre=w_genre, w_emotion=w_emotion)
            scored.append({**song, "score": round(score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": analysis,
            "recommendations": scored[:top_n],
        }

    # ── Database Management ─────────────────────────────────────

    def load_song_db(self, json_path):
        """Load song database from a JSON file."""
        with open(json_path, "r") as f:
            self.song_db = json.load(f)

    def save_song_db(self, json_path):
        """Save current song database to a JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.song_db, f, indent=2)
