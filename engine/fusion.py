"""
fusion.py
---------
Weighted fuzzy fusion of genre membership + emotion similarity.

Provides:
    • FuzzyFusion class — combines genre + emotion into a final mood score
    • Standalone helpers used by recommender & UI:
        emotion_similarity, genre_similarity, fused_score
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  FuzzyFusion — main fusion engine
# ═══════════════════════════════════════════════════════════════

class FuzzyFusion:
    """Combine genre confidence and emotion V/A into a single mood score.

    Parameters:
        w_genre   – weight for genre confidence   (default 0.6)
        w_emotion – weight for emotion component   (default 0.4)
        va_scale  – max value on the V-A scale     (DEAM = 9.0)
    """

    def __init__(self, w_genre=0.6, w_emotion=0.4, va_scale=9.0):
        assert abs(w_genre + w_emotion - 1.0) < 1e-6, "Weights must sum to 1"
        self.w_genre = w_genre
        self.w_emotion = w_emotion
        self.va_scale = va_scale

    # ── Public API ──────────────────────────────────────────────

    def fuse(self, genre_result: dict, emotion_result: dict) -> dict:
        """
        Fuse genre and emotion analysis into a final mood assessment.

        Args:
            genre_result:   dict from GenreClassifier
                            (must contain 'confidence', 'top_genre',
                             'fuzzy_memberships')
            emotion_result: dict from EmotionRegressor
                            (must contain 'valence', 'arousal',
                             'mood' or 'mood_label')

        Returns:
            {
                "mood_label":        str   – final mood quadrant,
                "mood_score":        float – weighted fusion score [0, 1],
                "genre_score":       float – genre confidence [0, 1],
                "emotion_score":     float – normalised (V+A)/2  [0, 1],
                "top_genre":         str,
                "genre_confidence":  float,
                "valence":           float,
                "arousal":           float,
                "fuzzy_memberships": dict,
            }
        """
        genre_score = float(genre_result["confidence"])

        valence = float(emotion_result["valence"])
        arousal = float(emotion_result["arousal"])
        emotion_score = self._normalize_va(valence, arousal)

        final_score = self.w_genre * genre_score + self.w_emotion * emotion_score

        mood_label = emotion_result.get("mood") or emotion_result.get("mood_label", "Unknown")

        return {
            "mood_label":        mood_label,
            "mood_score":        round(final_score, 4),
            "genre_score":       round(genre_score, 4),
            "emotion_score":     round(emotion_score, 4),
            "top_genre":         genre_result["top_genre"],
            "genre_confidence":  round(genre_score, 4),
            "valence":           round(valence, 4),
            "arousal":           round(arousal, 4),
            "fuzzy_memberships": genre_result.get("fuzzy_memberships", {}),
        }

    # ── Helpers ─────────────────────────────────────────────────

    def _normalize_va(self, valence, arousal):
        """Compute emotion score as 1 − normalised distance from centre (5, 5).

        Songs near the neutral centre get a low score (emotion is
        uninformative); songs far from centre get a high score,
        reflecting strong emotional character.

        Returns a value in [0, 1].
        """
        midpoint = self.va_scale / 2.0          # 4.5 on a 0-9 scale, 5.0 conceptually
        # Max possible distance: corner to centre on 1–9 scale → √(4²+4²) ≈ 5.66
        max_dist = np.sqrt(2) * (self.va_scale - 1.0) / 2.0
        dist = np.sqrt((valence - midpoint) ** 2 + (arousal - midpoint) ** 2)
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
#  Standalone helpers (backward-compatible — used by recommender
#  and desktop_app)
# ═══════════════════════════════════════════════════════════════


def emotion_similarity(query_va, candidate_va):
    """
    Compute similarity between two (valence, arousal) points.
    Uses inverse Euclidean distance mapped to [0, 1].

    Max possible distance on DEAM 1-9 scale ≈ 11.31 (diagonal).
    """
    dist = np.sqrt(
        (query_va[0] - candidate_va[0]) ** 2 +
        (query_va[1] - candidate_va[1]) ** 2
    )
    max_dist = np.sqrt(8**2 + 8**2)  # ~11.31
    return 1.0 - (dist / max_dist)


def genre_similarity(query_memberships, candidate_genre):
    """
    Fuzzy genre similarity: returns the query's membership degree
    for the candidate's genre label.
    """
    return query_memberships.get(candidate_genre, 0.0)


def fused_score(query_result, candidate, w_genre=0.4, w_emotion=0.6,
                neutral_radius=0.5, midpoint=5.0):
    """
    Compute weighted fusion score between a query analysis and a
    candidate song entry from the recommendation database.

    Adaptive weighting:
        If the query's emotion sits near the neutral centre
        (Euclidean distance to midpoint < *neutral_radius*), emotion
        carries little discriminative power, so the weights shift to
        favour genre (0.75 / 0.25).  Otherwise the defaults apply.

    Args:
        query_result: dict from analyze_audio() with:
            - genre.fuzzy_memberships
            - emotion.valence, emotion.arousal
        candidate: dict with:
            - "genre": str
            - "valence": float
            - "arousal": float
        w_genre:  default genre weight  (used when emotion is informative)
        w_emotion: default emotion weight
        neutral_radius: distance threshold for neutral-zone detection
        midpoint: V-A scale centre (DEAM = 5.0)

    Returns:
        float score in [0, 1]  (higher = more similar)
    """
    q_val = query_result["emotion"]["valence"]
    q_aro = query_result["emotion"]["arousal"]

    # Distance of the query emotion from the neutral centre
    centre_dist = np.sqrt((q_val - midpoint) ** 2 + (q_aro - midpoint) ** 2)

    if centre_dist < neutral_radius:
        # Emotion is near-neutral → lean on genre instead
        wg, we = 0.75, 0.25
    else:
        wg, we = w_genre, w_emotion

    g_sim = genre_similarity(
        query_result["genre"]["fuzzy_memberships"],
        candidate["genre"]
    )

    e_sim = emotion_similarity(
        (q_val, q_aro),
        (candidate["valence"], candidate["arousal"])
    )

    return wg * g_sim + we * e_sim
