# MUUD — Hybrid Soft Computing Music Intelligence System

A desktop application that uses **CNN-based genre classification**, **hybrid emotion regression (valence/arousal)**, and **fuzzy fusion scoring** to analyze music and recommend similar songs.

Built with TensorFlow/Keras, librosa, and a retro arcade-themed Tkinter GUI.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Datasets](#datasets)
5. [Training the Models](#training-the-models)
6. [Running the Desktop App](#running-the-desktop-app)
7. [Architecture Overview](#architecture-overview)
8. [Soft Computing Techniques](#soft-computing-techniques)
9. [Development Log](#development-log)
10. [License](#license)

---

## Features

| Feature | Details |
|---------|---------|
| **Genre Classification** | 10-class CNN trained on FMA-medium (~68 % val accuracy). Softmax outputs used as fuzzy membership degrees. Multi-segment averaging + temperature scaling. Hybrid label when top-2 genres are within 0.10. |
| **Emotion Regression** | Hybrid CNN + handcrafted features (tempo, spectral centroid, RMS, ZCR) → continuous Valence & Arousal on the DEAM 1–9 scale. Multi-segment averaging, V/A spread transform, and RMS-based arousal energy boost. |
| **Fuzzy Fusion Scoring** | Weighted fusion: `0.7 × genre_similarity + 0.3 × emotion_similarity`. Graded genre similarity matrix captures inter-genre relationships. |
| **Recommendation Engine** | Ranks a 60-song database by fused similarity score; top-N results displayed in a sortable table. |
| **Live Microphone Mode** | Continuous streaming from the default microphone with a rolling spectrogram display and inference every ~7 s. Full analysis results update live. |
| **5 s Microphone Recording** | Quick "REC 5 s" button — records, auto-analyzes, then cleans up the temp file. |
| **Explainability Panel** | Collapsible panel showing fusion formula, genre membership bar chart, emotion similarity breakdown, and intermediate computation values. |
| **V-A Visualisation** | Embedded matplotlib scatter plot of Russell's circumplex model with quadrant labels. Marker size and glow intensity scale with genre confidence. |
| **Top-3 Genre Probabilities** | Results panel shows the top-3 predicted genres with percentage probabilities before the full bar chart. |
| **Temporal Smoothing** | Live mic genre predictions are smoothed over a rolling buffer of 5 inference windows to reduce label flickering. |
| **Singleton Model Loading** | `ModelRegistry` singleton loads all Keras models once at startup with warm-up passes — no reloading on repeated analyses. Results cached by file path. |
| **Retro Arcade UI** | Dark navy theme, neon accents (cyan / magenta / green / yellow), custom 3D `NeonButton` canvas widgets, pulsing title animation, box-drawing characters. |

---

## Project Structure

```
Muud/
├── main.py                          # Entry point — launches desktop app
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── engine/                          # Core ML + inference logic
│   ├── __init__.py
│   ├── feature_extraction.py        # load_audio, mel spectrogram & handcrafted
│   │                                  feature extraction (segmented)
│   ├── genre_classifier.py          # GenreClassifier — predict, predict_averaged,
│   │                                  predict_averaged_smoothed (live), temperature
│   │                                  scaling, hybrid genre labeling, adaptive
│   │                                  segmentation for short / long clips
│   ├── emotion_regressor.py         # EmotionRegressor — predict, predict_averaged,
│   │                                  V/A spread transform, RMS energy boost,
│   │                                  mood quadrant with ±1.0 neutral zone
│   ├── fusion.py                    # Weighted fuzzy fusion (genre + emotion),
│   │                                  emotion_similarity, genre_similarity helpers
│   ├── model_registry.py            # ModelRegistry singleton — thread-safe model
│   │                                  loading + warmup
│   └── recommender.py               # MusicRecommender — analyze (file), analyze_signal
│                                      (live), recommend, graded genre similarity matrix
│
├── ui/                              # Desktop GUI
│   ├── __init__.py
│   └── desktop_app.py               # MuudApp Tkinter app — retro theme, NeonButton,
│                                      V-A plot, live spectrogram, live mic streaming,
│                                      5 s recording, analysis/recommend/explain panels
│
├── models/                          # Trained model weights (git-ignored except JSON)
│   ├── genre_fma_cnn.keras          # FMA 10-class genre CNN (git-ignored)
│   ├── emotion_hybrid_model.keras   # DEAM hybrid emotion regressor (git-ignored)
│   └── genre_labels.json            # Genre index → name mapping (tracked)
│
├── data/                            # Datasets & song database
│   ├── song_db.csv                  # 60-song database with V/A annotations (tracked)
│   ├── DEAM/                        # DEAM dataset (git-ignored — download separately)
│   ├── GTZAN/                       # GTZAN dataset (git-ignored)
│   └── FMA/                         # FMA-medium dataset (git-ignored)
│
├── training/                        # Jupyter / Colab notebooks
│   ├── genre_mel_training.ipynb     # GTZAN genre CNN training (initial model)
│   ├── train_hybrid_emotion.ipynb   # DEAM hybrid emotion model training
│   ├── fma_genre_clean.ipynb        # FMA-medium genre label cleaning / CSV prep
│   ├── fma_dataset_inspection.ipynb # FMA-medium dataset analysis & genre selection
│   ├── fusion_inference.ipynb       # End-to-end inference pipeline test
│   └── reports/                     # Saved figures from training runs
│
├── inference/                       # Standalone test scripts
│   ├── test_genre.py               # Genre classifier sanity check
│   ├── test_emotion.py             # Emotion regressor sanity check
│   ├── test_fusion.py              # Fusion pipeline test
│   └── test_recommend.py           # Full recommendation pipeline test
│
└── test_audio/                      # Sample audio for quick testing (git-ignored)
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or virtualenv
- A working microphone (optional — for live mic / recording features)

### 1. Create environment

```bash
conda create -n emotioncnn python=3.10
conda activate emotioncnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn
pip install sounddevice scipy
```

### 3. Download datasets

See [Datasets](#datasets) below for download links and placement instructions.
Datasets are only needed if you plan to retrain the models.

### 4. Obtain trained models

**Option A — Train locally / on Colab:**
Run the training notebooks (see [Training the Models](#training-the-models)).

**Option B — Use pre-trained weights:**
Place the `.keras` files in `models/`:
- `models/genre_fma_cnn.keras`
- `models/emotion_hybrid_model.keras`

### 5. Launch the app

```bash
python main.py
```

---

## Datasets

All datasets are **git-ignored** due to their size. Download and place them manually.

### DEAM (MediaEval Database for Emotional Analysis of Music)

- **Used for:** Emotion regression model (Valence / Arousal)
- **Size:** ~12 GB (1 802 songs + annotations)
- **Download:** [DEAM on Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)
- **Placement:** Extract to `data/DEAM/` preserving the original folder structure:
  ```
  data/DEAM/
  ├── DEAM_Annotations/
  │   └── annotations/
  │       ├── annotations averaged per song/
  │       └── annotations per each rater/
  ├── DEAM_audio/
  │   └── MEMD_audio/
  └── features/
      └── features/
  ```

### FMA-Medium (Free Music Archive)

- **Used for:** Genre classification CNN (10-class)
- **Size:** ~22 GB audio + ~350 MB metadata
- **Downloads:**
  - Audio: [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip)
  - Metadata: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)
  - GitHub: [mdeff/fma](https://github.com/mdeff/fma)
- **Placement:** Extract to `data/FMA/fma_medium/` and `data/FMA/fma_metadata/`.

### GTZAN Genre Collection (legacy)

- **Used for:** Initial genre model training (superseded by FMA)
- **Size:** ~1.2 GB (1 000 clips × 30 s, 10 genres)
- **Download:** [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Placement:** Extract to `data/GTZAN/` so that genre folders (`blues/`, `classical/`, etc.) are directly inside.

---

## Training the Models

Run notebooks from an activated `emotioncnn` environment, or adapt them for **Google Colab** with GPU.

### 1. Genre CNN — FMA-medium

**Notebooks:** `training/fma_dataset_inspection.ipynb` → `training/fma_genre_clean.ipynb`

1. `fma_dataset_inspection.ipynb` — downloads & inspects FMA-medium metadata, selects top-10 genres
2. `fma_genre_clean.ipynb` — cleans genre labels, creates train/val CSV splits
3. Train the CNN (Colab recommended for GPU) → saves `models/genre_fma_cnn.keras`

**Genre classes (10):** Classical, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Old-Time / Historic, Pop, Rock

### 2. Hybrid Emotion Model — DEAM

**Notebook:** `training/train_hybrid_emotion.ipynb`

- Loads DEAM audio + per-song V/A annotations
- Splits into 3 s segments; extracts 128-bin Mel spectrogram + 4 handcrafted features (tempo, spectral centroid, RMS, ZCR)
- Trains a hybrid CNN (Mel branch + dense branch) → 2 regression outputs (valence, arousal)
- Saves → `models/emotion_hybrid_model.keras`

### 3. Inference Test

**Notebook:** `training/fusion_inference.ipynb` — end-to-end pipeline test (feature extraction → genre → emotion → fusion → recommendation)

---

## Running the Desktop App

```bash
conda activate emotioncnn
python main.py
```

### What happens on launch

1. `ModelRegistry` singleton loads both Keras models from `models/`
2. Warm-up forward passes compile TF graphs (first launch slightly slower)
3. Retro-themed Tkinter window opens (1350 × 860)

### Controls

| Button | Action |
|--------|--------|
| **BROWSE** | Select a `.wav` / `.mp3` / `.flac` / `.ogg` file |
| **ANALYZE** | Run genre + emotion analysis on the selected file |
| **RECOMMEND** | Get top-5 similar songs from the song database |
| **EXPLAIN** | Toggle explainability panel (fusion breakdown) |
| **REC 5 s** | Record 5 seconds from microphone → auto-analyze |
| **🎤 LIVE MIC** | Toggle continuous microphone streaming with live spectrogram and rolling inference |

### Output Panels

- **Results** — Top-3 genre probabilities, full fuzzy membership bar chart, mood quadrant, valence/arousal scores
- **V-A Plot** — Russell's circumplex scatter; marker size scales with confidence
- **Recommendations** — Sortable table ranked by fused similarity score
- **Explainability** — Intermediate fusion computation values

---

## Architecture Overview

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Audio File  │────▶│ Feature Extract  │────▶│  Genre CNN      │──▶ Fuzzy memberships
│  or Live Mic │     │ (Mel + Stats)    │     │  (10-class FMA) │    (softmax probs)
└──────────────┘     │ × N segments     │     └─────────────────┘
                     │                  │     ┌─────────────────┐
                     │                  │────▶│  Emotion Hybrid │──▶ Valence, Arousal
                     └─────────────────┘     │  (CNN + Dense)  │    (1–9 scale)
                                              └─────────────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │  Fuzzy Fusion   │
                                              │  (weighted sim) │──▶ Ranked recommendations
                                              └─────────────────┘
```

### Input Processing

- **File analysis:** Audio loaded at 22 050 Hz, split into segments (10 s for genre, 3 s for emotion). Predictions averaged across segments.
- **Live mic:** Continuous 22 050 Hz stream, rolling 30 s buffer, inference every ~7 s with temporal smoothing (5-window average).
- **Short clips (< 10 s):** Single padded segment for genre; normal 3 s segments for emotion.

---

## Soft Computing Techniques

This project integrates multiple soft computing paradigms:

| Technique | Where Used | Details |
|-----------|-----------|---------|
| **Fuzzy Logic** | Genre classification | Softmax probabilities treated as fuzzy membership degrees across all 10 genres; hybrid labelling when membership gap < 0.10 |
| **Fuzzy Fusion** | Recommendation scoring | Weighted combination: `0.7 × genre_similarity + 0.3 × emotion_similarity`. Genre similarity uses a graded inter-genre similarity matrix. |
| **Neural Network (CNN)** | Genre classifier | Convolutional Neural Network on 128-bin Mel spectrograms (FMA-medium, 10 classes) |
| **Hybrid Neural Network** | Emotion regressor | CNN branch (Mel spectrogram) + Dense branch (handcrafted features) → multi-output regression |
| **Temperature Scaling** | Genre softmax | `softmax(log(p) / T)` post-hoc calibration to sharpen/soften probability distributions |
| **Temporal Smoothing** | Live mic predictions | Rolling average over 5 inference windows reduces noise in real-time genre predictions |
| **V/A Spread Transform** | Emotion post-processing | Linear transform expands clustered valence/arousal predictions across the full 1–9 scale |
| **Energy-based Arousal Boost** | Emotion post-processing | RMS energy injected directly into arousal prediction to improve sensitivity to dynamic range |

---

## Development Log

| # | What was done |
|---|---------------|
| 1 | Fixed DEAM training pipeline — path resolution, `iterrows()` float cast, librosa tempo array squeeze |
| 2 | Created hybrid emotion training notebook — CNN + handcrafted features → V/A regression (18 020 segments from 1 802 songs) |
| 3 | Built modular `engine/` package — `feature_extraction`, `genre_classifier`, `emotion_regressor`, `fusion`, `recommender` |
| 4 | Created retro arcade Tkinter GUI — dark navy theme, neon accents, custom `NeonButton` canvas widgets |
| 5 | Added embedded V-A scatter plot (matplotlib `FigureCanvasTkAgg`) with quadrant labels and glowing dot |
| 6 | Refactored recommendations into a sortable `ttk.Treeview` table (7 columns, neon-green rank-1 highlight) |
| 7 | Added collapsible explainability panel — fusion formula, genre bars, emotion similarity breakdown |
| 8 | Added microphone recording — 5 s at 22 050 Hz via `sounddevice`, temp WAV, auto-analyze, blinking indicator |
| 9 | Optimised startup — `ModelRegistry` singleton loads models once, warm-up passes, analysis caching |
| 10 | Multi-segment genre averaging — split audio into N segments, average softmax vectors |
| 11 | Improved mood classification — neutral buffer zone (±1.0 from midpoint) |
| 12 | Multi-segment emotion averaging — average V/A across segments for stability |
| 13 | Hybrid genre labelling — `"Hybrid: X / Y"` when top-2 gap < 0.10 |
| 14 | Adaptive fusion weights — graded genre similarity matrix for cross-genre relationships |
| 15 | Temperature scaling — `softmax(log(p)/T)` on genre probabilities |
| 16 | FMA-medium dataset inspection — download, extract, analyse genre distribution |
| 17 | Trained FMA-medium genre CNN (10 classes, ~68 % val accuracy), replacing GTZAN model |
| 18 | Adaptive segmentation — short clips (< 10 s) use a single padded segment; 10-30 s clips use proportional segments |
| 19 | V/A spread transform — linear expansion of clustered predictions across full 1–9 range |
| 20 | RMS energy arousal boost — injects RMS energy into arousal to improve dynamic sensitivity |
| 21 | Temporal smoothing for live mic — rolling 5-window average of genre probabilities |
| 22 | Top-3 genre probabilities in results panel with percentages |
| 23 | Confidence-scaled V-A plot — marker size and glow intensity proportional to genre confidence |
| 24 | Live microphone mode — continuous streaming, rolling spectrogram, ~7 s inference cycle |
| 25 | Live mic → analysis panel — live inference results populate the full analysis text and V-A plot |
| 26 | Codebase cleanup — updated README, .gitignore, requirements.txt; removed unused files |

---

## License

This project is for academic/educational purposes.

- **DEAM:** Creative Commons — [MediaEval](https://cvml.unige.ch/databases/DEAM/)
- **FMA:** Creative Commons — [GitHub](https://github.com/mdeff/fma)
- **GTZAN:** Research use — [Marsyas](http://marsyas.info/downloads/datasets.html)
