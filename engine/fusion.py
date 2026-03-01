"""
fusion.py
---------
Weighted fuzzy fusion of genre membership + emotion similarity.
Produces a single relevance score for ranking/recommendation.
"""

import numpy as np


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
