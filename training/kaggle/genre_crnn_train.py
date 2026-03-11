"""
genre_crnn_train.py — Training loop for the Genre CRNN model
=============================================================
Kaggle training script.  Copy-paste into a Kaggle notebook cell
AFTER running genre_crnn_model.py (so `model` is already built).

Expects numpy files uploaded as a Kaggle Dataset:
  - X_train.npy  (N_train, 128, 431, 1)  float32
  - y_train.npy  (N_train,)              int
  - X_val.npy    (N_val, 128, 431, 1)    float32
  - y_val.npy    (N_val,)                int

Uses tf.data to stream batches from memory-mapped files,
so only ~1 batch is in RAM at a time.
Checkpoint saved to /kaggle/working/ (download from notebook output).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Mixed Precision (faster on GPU with float16 compute) ───────
# NOTE: Disabled — caused NaN loss with softmax + MirroredStrategy.
# The 2×T4 setup is fast enough without it.
# tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ── Config ──────────────────────────────────────────────────────

EPOCHS = 60
BATCH_SIZE = 64  # 2×T4 GPUs → 32 per GPU
DATA_DIR = "/kaggle/input/genre-mudd"  # ← Kaggle dataset name
CHECKPOINT_PATH = "/kaggle/working/best_genre_crnn.keras"  # saved to Kaggle output

# ── Memory-mapped data loading (doesn't copy into RAM) ─────────

import os

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"), mmap_mode="r")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), mmap_mode="r")
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"),   mmap_mode="r")
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"),    mmap_mode="r")

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")

# ── SpecAugment (training data only) ────────────────────────────

FREQ_MASK_PARAM = 15   # max mel bins to mask
TIME_MASK_PARAM = 30   # max time frames to mask

def spec_augment(mel, label):
    """Apply random frequency and time masking to a mel spectrogram."""
    freq_bins = tf.shape(mel)[0]   # 128
    time_bins = tf.shape(mel)[1]   # 431

    # Frequency masking
    f = tf.random.uniform([], 0, FREQ_MASK_PARAM, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, freq_bins - f, dtype=tf.int32)
    freq_mask = tf.concat([
        tf.ones([f0, time_bins, 1]),
        tf.zeros([f, time_bins, 1]),
        tf.ones([freq_bins - f0 - f, time_bins, 1]),
    ], axis=0)

    # Time masking
    t = tf.random.uniform([], 0, TIME_MASK_PARAM, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, time_bins - t, dtype=tf.int32)
    time_mask = tf.concat([
        tf.ones([freq_bins, t0, 1]),
        tf.zeros([freq_bins, t, 1]),
        tf.ones([freq_bins, time_bins - t0 - t, 1]),
    ], axis=1)

    mel = mel * freq_mask * time_mask
    return mel, label

# ── tf.data pipeline (streams batches, never loads full dataset) ─

def make_dataset(X, y, batch_size, shuffle_size=0, augment=False):
    """Create a tf.data.Dataset that streams from memory-mapped arrays."""
    n = len(y)

    def generator_fn():
        idx = np.arange(n)
        if shuffle_size:
            np.random.shuffle(idx)
        for i in idx:
            yield X[i].astype(np.float32), int(y[i])

    ds = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle_size:
        ds = ds.shuffle(buffer_size=min(shuffle_size, 4096))
    if augment:
        ds = ds.map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    return ds

train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle_size=len(y_train), augment=True)
val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, shuffle_size=0, augment=False)
steps_per_epoch = len(y_train) // BATCH_SIZE
validation_steps = len(y_val) // BATCH_SIZE

# ── Callbacks ───────────────────────────────────────────────────

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

callbacks = [early_stop, reduce_lr, checkpoint]

# ── Resume from checkpoint if it exists ─────────────────────────

initial_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n✅ Found checkpoint: {CHECKPOINT_PATH}")
    model.load_weights(CHECKPOINT_PATH)
    initial_epoch = int(input("Enter last completed epoch number (0 if fresh): "))
    print(f"Resuming from epoch {initial_epoch}…\n")
else:
    print("\nNo checkpoint found — starting fresh.\n")

# ── Training ────────────────────────────────────────────────────

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
)

# ── Print final metrics ────────────────────────────────────────

best_idx = history.history["val_loss"].index(min(history.history["val_loss"]))
best_abs_epoch = initial_epoch + best_idx + 1
print(f"\nBest epoch: {best_abs_epoch}")
print(f"  Train acc : {history.history['accuracy'][best_idx]:.4f}")
print(f"  Val acc   : {history.history['val_accuracy'][best_idx]:.4f}")
print(f"  Val loss  : {history.history['val_loss'][best_idx]:.4f}")
