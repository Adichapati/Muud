"""
recommender.py
--------------
High-level orchestration: analyze audio, query song database,
rank by genre + emotion similarity, return recommendations.

Primary source: Spotify API (via SpotifyClient).
Fallback:       static data/song_db.csv.
"""

import csv
import logging
import os
import tempfile
from collections import OrderedDict
import numpy as np

from engine.feature_extraction import (
    load_audio, extract_mel, extract_handcrafted,
    extract_mel_segments, extract_handcrafted_segments,
)
from engine.genre_classifier import GenreClassifier
from engine.emotion_regressor import EmotionRegressor
from engine.model_registry import ModelRegistry
from engine.fusion import FuzzyFusion, fused_score
from engine.spotify_client import SpotifyClient

logger = logging.getLogger(__name__)


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

        # Spotify recommendation cache: avoids repeated API calls + preview downloads
        # Key: (genre, rounded_valence, rounded_arousal)
        # Value: list of scored track dicts
        self._spotify_cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._spotify_cache_limit = 50

        # ── Spotify client (primary track source) ────────────────
        try:
            self.spotify = SpotifyClient()
            print("[Recommender] Spotify client initialised ✓")
        except Exception as exc:
            self.spotify = None
            print(f"[Recommender] Spotify unavailable ({exc}) — CSV-only mode")

        # ── CSV fallback database ────────────────────────────────
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

    def invalidate_spotify_cache(self):
        """Clear the Spotify recommendation cache."""
        self._spotify_cache.clear()
        logger.info("Spotify recommendation cache cleared")

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

    def recommend(self, file_path, top_n=10, w_genre=0.7, w_emotion=0.3):
        """
        Analyze input audio and return top-N song recommendations.

        Primary source: Spotify (fetches 50 tracks by predicted genre).
        Fallback:       static song_db.csv.

        Scoring per candidate:
            genre_sim    = 1.0 (exact match), matrix value, or 0.0
            emotion_sim  = 1 − (euclidean_dist / max_dist)
            score        = w_genre * genre_sim + w_emotion * emotion_sim

        Returns:
            { "query": analysis_dict, "source": "spotify" | "csv",
              "recommendations": [ {title, artist, genre, …, score}, … ] }
        """
        analysis = self.analyze(file_path)

        q_genre = analysis["genre"]["top_genre"]
        q_val = analysis["emotion"]["valence"]
        q_aro = analysis["emotion"]["arousal"]

        # ── Build cache key (genre + rounded V/A) ───────────────
        search_genre = q_genre
        if search_genre.startswith("Hybrid:"):
            search_genre = search_genre.replace("Hybrid:", "").split("/")[0].strip()
        cache_key = (search_genre.lower(), round(q_val), round(q_aro))

        # ── Check Spotify cache ──────────────────────────────────
        if cache_key in self._spotify_cache:
            logger.info("Spotify cache HIT for %s", cache_key)
            self._spotify_cache.move_to_end(cache_key)   # refresh LRU
            scored = self._spotify_cache[cache_key]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return {
                "query": analysis,
                "source": "spotify_cache",
                "recommendations": scored[:top_n],
            }

        # ── Try Spotify API ────────────────────────────────────
        spotify_tracks = []
        if self.spotify is not None:
            try:
                spotify_tracks = self.spotify.search_tracks_by_genre(
                    search_genre, limit=50,
                )
                logger.info("Spotify returned %d tracks for '%s'",
                            len(spotify_tracks), search_genre)
            except Exception as exc:
                logger.warning("Spotify search failed: %s — falling back to CSV", exc)
                spotify_tracks = []

        if spotify_tracks:
            scored = self._score_spotify_tracks(
                spotify_tracks, q_genre, q_val, q_aro, w_genre, w_emotion,
            )
            # Store in cache (evict oldest if full)
            self._spotify_cache[cache_key] = scored
            if len(self._spotify_cache) > self._spotify_cache_limit:
                self._spotify_cache.popitem(last=False)   # remove oldest
            source = "spotify"
        else:
            # ── CSV fallback ─────────────────────────────────────
            scored = self._score_csv_tracks(
                q_genre, q_val, q_aro, w_genre, w_emotion,
            )
            source = "csv"

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": analysis,
            "source": source,
            "recommendations": scored[:top_n],
        }

    # ── Scoring helpers ──────────────────────────────────────────

    def _score_csv_tracks(self, q_genre, q_val, q_aro, w_genre, w_emotion):
        """Score candidates from the static song_db.csv."""
        scored = []
        for song in self.song_db:
            g_sim = _genre_similarity(q_genre, song["genre"])
            e_sim = _emotion_similarity(q_val, q_aro,
                                        song["valence"], song["arousal"])
            score = w_genre * g_sim + w_emotion * e_sim
            scored.append({**song, "score": round(score, 4)})
        return scored

    def _score_spotify_tracks(self, tracks, q_genre, q_val, q_aro,
                              w_genre, w_emotion):
        """Score candidates fetched from Spotify.

        Downloads each track's 30-second preview audio, runs it through
        the EmotionRegressor to get real valence/arousal, then scores
        using the standard genre + emotion similarity formula.

        Tracks without a preview_url fall back to neutral midpoint (5, 5).
        Popularity is used as a small tie-breaker.
        """
        NEUTRAL_V, NEUTRAL_A = 5.0, 5.0
        scored = []

        for i, t in enumerate(tracks):
            preview_url = t.get("preview_url")
            track_v, track_a = NEUTRAL_V, NEUTRAL_A
            analyzed = False

            # ── Download preview and predict emotion ─────────────
            if preview_url:
                try:
                    track_v, track_a = self._analyze_preview(preview_url)
                    analyzed = True
                except Exception as exc:
                    logger.debug("Preview analysis failed for '%s': %s",
                                 t.get("song", "?"), exc)

            g_sim = _genre_similarity(q_genre, q_genre)   # same genre → 1.0
            e_sim = _emotion_similarity(q_val, q_aro, track_v, track_a)

            # Small popularity bonus (0–0.05 range) to break ties
            pop_bonus = (t.get("popularity", 0) / 100) * 0.05

            score = w_genre * g_sim + w_emotion * e_sim + pop_bonus

            scored.append({
                "title":       t["song"],
                "artist":      t["artist"],
                "genre":       q_genre,
                "valence":     track_v,
                "arousal":     track_a,
                "score":       round(score, 4),
                "preview_url": preview_url,
                "album_art":   t.get("album_art"),
                "spotify_url": t.get("spotify_url"),
                "emotion_analyzed": analyzed,
            })

            if (i + 1) % 10 == 0:
                logger.info("  Scored %d/%d Spotify tracks", i + 1, len(tracks))

        return scored

    def _analyze_preview(self, preview_url: str) -> tuple[float, float]:
        """Download a Spotify preview MP3 and return (valence, arousal).

        The audio is saved to a temporary file, loaded via librosa,
        and run through the EmotionRegressor.  The temp file is
        deleted immediately after analysis.
        """
        import urllib.request

        # Download to a temp file (librosa needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve(preview_url, tmp_path)

        try:
            signal, sr = load_audio(tmp_path)
            mel_segs = extract_mel_segments(signal, sr, max_segments=3)
            stats_segs = extract_handcrafted_segments(signal, sr, max_segments=3)
            n = min(len(mel_segs), len(stats_segs))

            result = self.emotion_reg.predict_averaged(
                mel_segs[:n], stats_segs[:n],
            )
            return result["valence"], result["arousal"]
        finally:
            # Always clean up the temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Database Management ─────────────────────────────────────

    def load_song_db(self, csv_path):
        """Reload song database from a CSV file."""
        self.song_db = load_song_csv(csv_path)

    def reload_default_db(self):
        """Reload the default data/song_db.csv."""
        csv_path = os.path.join(self.project_root, "data", "song_db.csv")
        if os.path.isfile(csv_path):
            self.song_db = load_song_csv(csv_path)

