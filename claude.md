# Muud - Project Context for AI Assistant

## Last Updated: March 15, 2026

---

## Project Overview
**Muud** is a hybrid soft-computing music intelligence system that:
1. Takes audio files (or live mic input) as input
2. Extracts audio features (Mel spectrograms + handcrafted stats via librosa)
3. Classifies by **genre** (10-class FMA CNN + Transformer) and predicts **emotion** (valence/arousal regression)
4. Fuses genre + emotion via weighted fuzzy scoring
5. Recommends songs natively via the **Spotify API** (ranked by fused similarity), falling back to a CSV database natively.
6. UI: **CustomTkinter** glassmorphism desktop app featuring a dynamic hero carousel with circular album art, live spectrogram, V-A plot, and explainability mechanics.

---

## Tech Stack
- **Language:** Python 3
- **ML Framework:** TensorFlow / Keras (`.keras` format)
- **Audio Processing:** librosa
- **UI:** CustomTkinter (glassmorphism), Pillow, Tkinter
- **Live Audio:** sounddevice
- **API:** spotipy, python-dotenv
- **Dependencies:** tensorflow>=2.15, numpy>=1.24, scikit-learn>=1.3, librosa>=0.10, sounddevice>=0.4, scipy>=1.11, pandas>=2.0, matplotlib>=3.7, seaborn>=0.13, customtkinter>=5.2, spotipy>=2.23, Pillow>=10.0, python-dotenv>=1.0

---

## Project Structure
```
Muud/
├── main.py                      # Entry point — loads models, launches Tkinter UI
├── engine/
│   ├── __init__.py
│   ├── feature_extraction.py    # Shared audio feature extraction (Mel + handcrafted)
│   ├── genre_classifier.py      # GenreClassifier — 10-class FMA CNN + fuzzy memberships
│   ├── emotion_regressor.py     # EmotionRegressor — Hybrid CNN+Dense → V/A (DEAM)
│   ├── fusion.py                # FuzzyFusion + standalone similarity helpers
│   ├── recommender.py           # MusicRecommender — orchestrates analysis + CSV ranking
│   └── model_registry.py        # Singleton model loader with warm-up
├── models/
│   ├── best_genre_cnn_trans.keras # Trained genre CNN+Transformer (FMA-medium, 10 classes)
│   ├── best_genre_crnn.keras    # Previous genre CRNN (superseded by CNN+Transformer)
│   ├── emotion_hybrid_model.keras  # Trained emotion hybrid model (DEAM)
│   ├── genre_labels.json        # {0: "Classical", ..., 9: "Rock"}
│   └── genre_mel_cnn.keras      # Legacy GTZAN model (not used in current app)
├── ui/
│   ├── __init__.py
│   └── desktop_app.py           # MuudApp — CustomTkinter GUI (Glassmorphism + Hero Carousel)
├── inference/
│   ├── test_genre.py            # Sanity check: genre predictions on test_audio/
│   ├── test_emotion.py          # Sanity check: V/A predictions on test_audio/
│   ├── test_fusion.py           # End-to-end genre+emotion+fusion test
│   └── test_recommend.py        # Full recommendation pipeline test
├── training/
│   ├── fma_dataset_inspection.ipynb   # Download & inspect FMA-medium
│   ├── fma_genre_clean.ipynb          # Genre label cleaning → train/val CSVs
│   ├── genre_mel_training.ipynb       # Legacy GTZAN training (superseded)
│   ├── train_hybrid_emotion.ipynb     # DEAM hybrid emotion model training
│   ├── fusion_inference.ipynb         # End-to-end inference demo
│   └── reports/
├── data/
│   ├── song_db.csv              # 60 songs: song, artist, genre, valence, arousal
│   ├── DEAM/                    # DEAM dataset (1802 songs, V/A annotations)
│   │   ├── DEAM_Annotations/    # Per-song and per-rater annotations
│   │   ├── DEAM_audio/MEMD_audio/  # Audio files
│   │   └── features/features/   # Pre-extracted per-song CSVs (10.csv, 101.csv, ...)
│   ├── FMA/                     # FMA-medium dataset
│   │   ├── fma_medium_genre_clean.csv  # Cleaned: track_id, file_path, genre_label, genre_index
│   │   ├── fma_medium_train.csv        # Training split
│   │   ├── fma_medium_val.csv          # Validation split
│   │   ├── features/           # Pre-extracted numpy arrays (X_train, X_val, y_train, y_val)
│   │   ├── fma_medium/         # Raw audio files
│   │   └── fma_metadata/       # FMA metadata files
│   └── GTZAN/                  # GTZAN dataset (10 genres × 100 songs)
├── test_audio/                 # Audio files for inference testing
└── claude.md                   # THIS FILE
```

---

## Data Flow / Architecture

```
Audio Input (file or live mic @ 22,050 Hz)
    │
    ├─► Genre Pipeline (10-s segments, 431 time bins)
    │   └─► GenreClassifier (CNN + Transformer) → softmax → fuzzy memberships
    │
    ├─► Emotion Pipeline (3-s segments, ~130 time bins)
    │   ├─► Mel spectrogram (128, 130, 1)
    │   └─► Handcrafted stats (4,) = [tempo, centroid, rms, zcr]
    │       └─► EmotionRegressor (Hybrid CNN+Dense) → [valence, arousal] (1-9)
    │
    └─► FuzzyFusion (0.6×genre + 0.4×emotion) → mood score
        └─► Recommender ranks song_db.csv by fused similarity → top-N
```

---

## Genre Classifier (`engine/genre_classifier.py`)
- **Model:** `models/best_genre_cnn_trans.keras` *(replaces previous `best_genre_crnn.keras`)*
- **Architecture:** CNN feature extractor → Attention / Transformer block → Dense classification head
- **Input shape:** `(batch, 128, 431, 1)` — 10-second mel spectrogram *(unchanged from previous model)*
- **Output:** 10-class softmax
- **10 Classes:** Classical, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Old-Time / Historic, Pop, Rock
- **Validation accuracy:** ~0.77 *(improved from ~0.69 with previous CRNN)*
- **Segmentation:** Adaptive based on audio length:
  - ≥ 30s → 3 × 10-second segments
  - 10–30s → as many 10s segments as fit
  - < 10s → single segment padded to 431 time bins
- **Mel extraction params:** SR=22050, N_MELS=128, N_FFT=2048, HOP_LENGTH=512, ref=np.max
- **Normalization:** z-score per spectrogram, time-axis padded/trimmed to 431 bins
- **Post-processing:** Temperature scaling, hybrid threshold (gap < 0.10 → "Hybrid: A / B")
- **Live mic:** Rolling 5-prediction buffer for temporal smoothing

---

## Emotion Regressor (`engine/emotion_regressor.py`)
- **Model:** `models/emotion_hybrid_model.keras`
- **Inputs:** Two branches:
  - Mel: `(batch, 128, T, 1)` — 3-second segment (~130 time bins)
  - Stats: `(batch, 4)` — [tempo, spectral_centroid, rms, zcr]
- **Output:** `(batch, 2)` → [valence, arousal] continuous (DEAM 1–9 scale)
- **Warm-up shape:** `(1, 128, 130, 1)` mel + `(1, 4)` stats
- **Post-processing:**
  - RMS-based arousal boost: `arousal += 2.0 * rms_mean`
  - Spread transform: `(v - 4.5) * 2.0 + 5.0` (expands clustered predictions)
  - Clamp to [1, 9]
- **Mood quadrants** (Russell's circumplex, midpoint=5.0, neutral_radius=1.0):
  - Happy / Energetic (V≥5, A≥5)
  - Happy / Calm (V≥5, A<5)
  - Sad / Calm (V<5, A<5)
  - Angry / Intense (V<5, A≥5)
  - Neutral / Balanced (both within ±1.0 of midpoint)

---

## Fusion (`engine/fusion.py`)
- **FuzzyFusion:** `mood_score = 0.6 × genre_confidence + 0.4 × emotion_score`
- **emotion_score** = 1 − (distance_from_centre / max_distance)
- **Recommender scoring:** `w_genre × genre_sim + w_emotion × emotion_sim`
  - Adaptive: near-neutral emotion → shift to 0.75 genre / 0.25 emotion
- **Genre similarity matrix** in recommender.py: pairwise scores (e.g., Classical↔Instrumental=0.8, Folk↔Old-Time=0.8)

---

## Recommender (`engine/recommender.py`)
- Orchestrates: load_audio → feature extraction → genre + emotion → fusion → rank CSV
- **Song DB:** `data/song_db.csv` (60 songs with: song, artist, genre, valence, arousal)
- **Caching:** Results cached by absolute file path
- **Scoring:** `w_genre(0.7) × genre_sim + w_emotion(0.3) × emotion_sim`

---

## Feature Extraction (`engine/feature_extraction.py`)
- **Shared by both pipelines** (but genre_classifier.py has its own internal extraction too)
- `load_audio(path)` → signal, sr at 22050 Hz
- `extract_mel(signal, sr)` → `(1, 128, T, 1)` single 3-s segment
- `extract_mel_segments(signal, sr, max_segments=5)` → list of mel arrays
- `extract_handcrafted(signal, sr)` → `(1, 4)` [tempo, centroid, rms, zcr]
- `extract_handcrafted_segments(signal, sr, max_segments=5)` → list of stat arrays
- **Note:** Uses `librosa.power_to_db(mel)` (no ref param) vs genre_classifier uses `ref=np.max`

---

## Datasets

### FMA-Medium (Genre Training)
- **Total usable tracks (after cleaning):** ~15,561
- **CSV:** `data/FMA/fma_medium_genre_clean.csv` — columns: track_id, file_path (Windows paths), genre_label, genre_index (0-9)
- **Splits:** `fma_medium_train.csv`, `fma_medium_val.csv`
- **Local pre-extracted features:** `data/FMA/features/{X_train,X_val,y_train,y_val}.npy`
- **Kaggle cached spectrograms:** `/kaggle/working/mel_cache/{X_train,X_val,y_train,y_val}.npy` (~2.7 GB for X_train)
- **Audio:** `data/FMA/fma_medium/fma_medium/`

### DEAM (Emotion Training)
- 1,802 songs with per-song valence/arousal annotations (1-9 scale)
- **Annotations:** `data/DEAM/DEAM_Annotations/annotations/`
  - `annotations averaged per song/` — static per-song V/A
  - `annotations per each rater/` — individual rater scores
- **Audio:** `data/DEAM/DEAM_audio/MEMD_audio/`
- **Pre-extracted features:** `data/DEAM/features/features/` (per-song CSVs like 10.csv, 101.csv, ...)
- Training: 3-second segments → ~18,020 samples from 1,802 songs

### GTZAN (Legacy — not used for current model)
- 10 genres × 100 songs (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- Was used for `genre_mel_cnn.keras` (superseded by FMA model)

---

## Model I/O Shapes (Critical for Retraining)

| Model | Input | Output |
|-------|-------|--------|
| Genre CNN+Transformer | `(batch, 128, 431, 1)` mel spectrogram | `(batch, 10)` softmax |
| Emotion Hybrid | `[(batch, 128, T, 1) mel, (batch, 4) stats]` | `(batch, 2)` [V, A] |

### Genre Mel Extraction (in genre_classifier.py)
- 10-second segments, pad/trim to 431 time bins
- `melspectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512)`
- `power_to_db(mel, ref=np.max)`
- z-score normalize

### Emotion Mel Extraction (in emotion_regressor.py)
- 3-second segments (~130 time bins)
- `melspectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512)`
- `power_to_db(mel, ref=np.max)`
- z-score normalize
- Stats: [tempo, centroid, rms, zcr] as (1, 4)

---

## Genre Model Training Details (CNN + Transformer, March 2026)

- **Platform:** Kaggle Notebook with 2× Tesla T4 GPUs
- **Why Kaggle:** Local TensorFlow GPU support was unstable
- **Architecture:** CNN feature extractor → Attention / Transformer block → Dense (10-class softmax)
- **Training data:** Mel spectrograms precomputed from raw FMA audio and cached as `.npy` files
- **Audio processing pipeline:**
  - Sample rate: 22,050 Hz
  - Segment length: 10 seconds
  - Mel spectrogram: n_mels=128, n_fft=2048, hop_length=512
  - Spectrogram → dB conversion → z-score normalization → pad/trim to 431 time bins
- **Training parameters:**
  - Epochs: 40 | Batch size: 8 | Optimizer: Adam (lr=1e-4)
  - Loss: sparse_categorical_crossentropy
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (patience=4, factor=0.5), ModelCheckpoint (save_best_only)
- **Training speed:** ~50 seconds per epoch (with cached spectrograms)
- **Results:**
  - Epoch 1 — train_acc ≈ 0.76, val_acc ≈ 0.76
  - Best — train_acc ≈ 0.79, val_acc ≈ 0.77
- **Output checkpoint:** `best_genre_cnn_trans.keras`
- **Constraint:** I/O shapes preserved (`(batch, 128, 431, 1)` → `(batch, 10)`) so existing inference code works unchanged

---

## Next Steps

1. **Integrate new genre model** into `engine/genre_classifier.py` (load `best_genre_cnn_trans.keras` instead of `best_genre_crnn.keras`)
2. **Verify inference** works with the new model (input shape must remain 128×431×1)
3. **Evaluate Experimental genre bias** — check if the new model reduces over-prediction of Experimental
4. **Improve recommendation scoring** if necessary
5. **Optionally retrain or fine-tune the emotion model** (DEAM)
6. **Add Spotify preview playback** to the UI for recommended tracks

---

## Session Log
### Session 1 — March 10, 2026
- Full codebase scan completed
- Created comprehensive `claude.md` context file
- Documented all engine modules, model architectures, data flow, datasets, I/O shapes
- Ready for Kaggle training script generation

### Session 2 — March 15, 2026
- Genre classifier retrained on Kaggle using FMA-medium dataset
- Architecture changed from CRNN to CNN + Transformer
- Mel spectrograms precomputed and cached as `.npy` files for fast training (~50s/epoch)
- Best validation accuracy improved from ~0.69 to ~0.77
- New model checkpoint: `best_genre_cnn_trans.keras`
- `claude.md` updated with new architecture, training details, results, and next steps
