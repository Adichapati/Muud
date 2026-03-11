"""
genre_crnn_model.py — CRNN architecture for 10-class music genre classification
================================================================================
Kaggle training script.  Copy-paste into a Kaggle notebook cell.

Input:  Mel spectrogram (128, 431, 1)
Output: 10-class softmax (sparse categorical crossentropy)

Architecture: 4× Conv blocks → Permute+Reshape → Compress → 2× BiLSTM → GlobalAvgPool → Dense head

Optimised for Kaggle (2× T4 GPUs via MirroredStrategy).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

# ── Multi-GPU Strategy (Kaggle 2×T4) ──────────────────────────
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")


def build_genre_crnn(input_shape=(128, 431, 1), num_classes=10):
    """
    Build a CRNN model for genre classification.

    CNN extracts local spectral/temporal features,
    then bidirectional LSTMs capture long-range temporal dependencies.
    """
    inp = Input(shape=input_shape, name="mel_input")

    # ── Conv Block 1 ────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Conv Block 2 ────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Conv Block 3 ────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Conv Block 4 ────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── Reshape CNN output → sequence for RNN ───────────────────
    # Conv2D layout: (batch, freq, time, channels)
    # After 4× MaxPool(2,2):
    #   freq:  128 → 64 → 32 → 16 → 8
    #   time:  431 → 215 → 107 → 53 → 26
    # Current shape: (batch, 8, 26, 128)
    #
    # RNN should process along the TIME axis, so:
    #   Permute → (batch, time=26, freq=8, channels=128)
    #   Reshape → (batch, 26, 8×128=1024)
    x = layers.Permute((2, 1, 3))(x)          # (batch, 26, 8, 128)
    x = layers.Reshape((-1, 8 * 128))(x)      # (batch, 26, 1024)

    # ── Compress feature dim before LSTM (saves RAM) ────────────
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)  # (batch, 26, 128)

    # ── Bidirectional LSTM layers ─────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True)
    )(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True)
    )(x)
    x = layers.Dropout(0.4)(x)

    # ── Global Average Pooling over time ────────────────────────
    x = layers.GlobalAveragePooling1D()(x)

    # ── Dense classification head ───────────────────────────────
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="genre_output")(x)

    # ── Build & compile ─────────────────────────────────────────
    model = Model(inputs=inp, outputs=out, name="genre_crnn")

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ═════════════════════════════════════════════════════════════════
#  Build and print summary
# ═════════════════════════════════════════════════════════════════

print("Building Genre CRNN model …\n")
with strategy.scope():
    model = build_genre_crnn()
model.summary()

# Quick shape verification with a dummy batch
dummy = np.random.randn(2, 128, 431, 1).astype("float32")
preds = model.predict(dummy, verbose=0)
print(f"\nDummy input shape : {dummy.shape}")
print(f"Output shape      : {preds.shape}")
print(f"Sum of softmax    : {preds[0].sum():.6f}")
