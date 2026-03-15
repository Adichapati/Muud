"""
evaluate_genre_distribution.py
-------------------------------
Diagnostic script to detect genre prediction bias in the MUUD genre classifier.

Randomly samples 200 tracks from the FMA-medium dataset, runs genre predictions,
and reports frequency/percentage distributions with a bar chart.

Run from project root:  python -m inference.evaluate_genre_distribution

Flags a ⚠ WARNING if any single genre exceeds 30% of predictions.
"""

import os, sys, json, random, time
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from engine.genre_classifier import GenreClassifier

# ── Paths ────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_genre_cnn_trans.keras")
LABELS_PATH   = os.path.join(PROJECT_ROOT, "models", "genre_labels.json")
CSV_PATH      = os.path.join(PROJECT_ROOT, "data", "FMA",
                             "fma_medium_genre_clean.csv")
AUDIO_ROOT    = os.path.join(PROJECT_ROOT, "data", "FMA",
                             "fma_medium", "fma_medium")
CHART_OUT     = os.path.join(PROJECT_ROOT, "inference",
                             "genre_distribution.png")

# ── Config ───────────────────────────────────────────────────────
SAMPLE_SIZE   = 200
BIAS_THRESHOLD = 0.30          # warn if any genre > 30 %
RANDOM_SEED   = 42


def load_labels(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


def main():
    print("=" * 64)
    print("  GENRE PREDICTION BIAS EVALUATION")
    print("=" * 64)

    # ── Load label map ───────────────────────────────────────────
    labels = load_labels(LABELS_PATH)
    print(f"\nGenre labels ({len(labels)}): {', '.join(labels)}")

    # ── Load model ───────────────────────────────────────────────
    print(f"\nLoading model: {os.path.basename(MODEL_PATH)} …")
    model = tf.keras.models.load_model(MODEL_PATH)
    clf   = GenreClassifier(model=model, labels=labels)
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # ── Load and sample CSV ──────────────────────────────────────
    print(f"\nLoading FMA metadata: {os.path.basename(CSV_PATH)}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total rows: {len(df)}")

    # Resolve full audio paths and filter to files that actually exist
    df["full_path"] = df["file_path"].apply(
        lambda p: os.path.normpath(os.path.join(AUDIO_ROOT, p))
    )
    df_exists = df[df["full_path"].apply(os.path.isfile)].copy()
    print(f"  Files found on disk: {len(df_exists)}")

    if len(df_exists) < SAMPLE_SIZE:
        print(f"\n⚠  Only {len(df_exists)} files available — using all of them.")
        sample = df_exists
    else:
        sample = df_exists.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    print(f"  Sample size: {len(sample)}")

    # ── True label distribution in the sample ────────────────────
    true_counts = Counter(sample["genre_label"])
    print(f"\n  True genre distribution in sample:")
    for genre in labels:
        cnt = true_counts.get(genre, 0)
        print(f"    {genre:<22} {cnt:>4}")

    # ── Run predictions ──────────────────────────────────────────
    print(f"\nRunning predictions on {len(sample)} tracks …")
    predicted_genres: list[str] = []
    skipped = 0
    t0 = time.time()

    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        audio_path = row["full_path"]
        true_label = row["genre_label"]

        try:
            result = clf.predict_file(audio_path)
            # Extract the primary genre (strip "Hybrid: X / Y" → take first)
            pred = result["top_genre"]
            if pred.startswith("Hybrid:"):
                pred = pred.replace("Hybrid:", "").strip().split("/")[0].strip()
            predicted_genres.append(pred)
        except Exception as exc:
            skipped += 1
            predicted_genres.append("__ERROR__")
            if skipped <= 5:
                print(f"  ✗ [{idx}/{len(sample)}] {os.path.basename(audio_path)}: {exc}")
            continue

        # Progress update every 25 tracks
        if idx % 25 == 0 or idx == len(sample):
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  [{idx:>3}/{len(sample)}]  {rate:.1f} tracks/s  "
                  f"last: {pred:<22}  (true: {true_label})")

    elapsed = time.time() - t0
    valid_preds = [g for g in predicted_genres if g != "__ERROR__"]
    print(f"\nDone in {elapsed:.1f}s  |  {len(valid_preds)} succeeded, {skipped} skipped")

    # ── Frequency table ──────────────────────────────────────────
    pred_counts = Counter(valid_preds)
    total = len(valid_preds)

    print("\n" + "=" * 64)
    print("  PREDICTION DISTRIBUTION")
    print("=" * 64)
    print(f"\n  {'Genre':<22}  {'Count':>5}  {'Pct':>6}  {'Bar'}")
    print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*30}")

    warnings: list[str] = []

    for genre in labels:
        cnt = pred_counts.get(genre, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        flag = " ⚠" if pct > BIAS_THRESHOLD * 100 else ""
        print(f"  {genre:<22}  {cnt:>5}  {pct:>5.1f}%  {bar}{flag}")
        if pct > BIAS_THRESHOLD * 100:
            warnings.append(f"{genre}: {pct:.1f}%")

    # Unknown predictions (genre not in label set)
    unknown = [g for g in valid_preds if g not in labels]
    if unknown:
        unknown_counts = Counter(unknown)
        print(f"\n  Unknown predictions: {dict(unknown_counts)}")

    # ── Accuracy against true labels ─────────────────────────────
    correct = 0
    confusion_pairs: Counter = Counter()
    for (_, row), pred in zip(sample.iterrows(), predicted_genres):
        if pred == "__ERROR__":
            continue
        if pred == row["genre_label"]:
            correct += 1
        else:
            confusion_pairs[(row["genre_label"], pred)] += 1

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n  Sample accuracy: {correct}/{total} = {acc:.1f}%")

    # Top-10 confusion pairs
    if confusion_pairs:
        print(f"\n  Top confusion pairs (true → predicted):")
        for (true_g, pred_g), cnt in confusion_pairs.most_common(10):
            print(f"    {true_g:<22} → {pred_g:<22}  ({cnt})")

    # ── Warnings ─────────────────────────────────────────────────
    print()
    if warnings:
        print("⚠" * 32)
        print("  BIAS WARNING: The following genres exceed "
              f"{BIAS_THRESHOLD*100:.0f}% of predictions:")
        for w in warnings:
            print(f"    ⚠  {w}")
        print("⚠" * 32)
    else:
        print("✓  No genre exceeds the bias threshold "
              f"({BIAS_THRESHOLD*100:.0f}%). Distribution looks balanced.")

    # ── Bar chart ────────────────────────────────────────────────
    print(f"\nSaving bar chart → {CHART_OUT}")
    _plot_distribution(labels, pred_counts, true_counts, total)
    print("Done.\n")


def _plot_distribution(labels, pred_counts, true_counts, total):
    """Save a grouped bar chart comparing predicted vs true distribution."""
    x = np.arange(len(labels))
    width = 0.35

    pred_vals = [pred_counts.get(g, 0) for g in labels]
    true_vals = [true_counts.get(g, 0) for g in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, pred_vals, width, label="Predicted",
                   color="#5b9bd5", edgecolor="white")
    bars2 = ax.bar(x + width/2, true_vals, width, label="True label",
                   color="#ed7d31", edgecolor="white", alpha=0.7)

    # 30 % threshold line
    threshold_line = total * BIAS_THRESHOLD
    ax.axhline(y=threshold_line, color="red", linestyle="--", linewidth=1,
               label=f"Bias threshold ({BIAS_THRESHOLD*100:.0f}%)")

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Genre Prediction Distribution vs True Labels  "
                 f"(n={total})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on predicted bars
    for bar, val in zip(bars1, pred_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(CHART_OUT, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
