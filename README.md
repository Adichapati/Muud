# MUUD — Hybrid Soft Computing Music Intelligence System

A desktop application that uses **CNN-based genre classification**, **emotion regression (valence / arousal)**, and **fuzzy fusion scoring** to analyze music and recommend similar songs.

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
8. [Development Log](#development-log)  

---

## Features

| Feature | Details |
|---------|---------|
| **Genre Classification** | 10-class CNN trained on GTZAN, softmax outputs used as fuzzy membership degrees. Multi-segment averaging + temperature scaling for robustness. Hybrid label when top-2 genres are within 0.10. |
| **Emotion Regression** | Hybrid CNN + handcrafted features (tempo, spectral centroid, RMS, ZCR) → continuous Valence & Arousal (DEAM 1–9 scale). Multi-segment averaging for stability. Neutral buffer zone (±0.5 from midpoint). |
| **Fuzzy Fusion Scoring** | Adaptive weighted fusion: `0.4 × genre_sim + 0.6 × emotion_sim` (shifts to `0.75 / 0.25` when emotion is near-neutral). |
| **Recommendation Engine** | Ranks a song database by fused similarity score; top-N results displayed in a sortable table. |
| **Explainability Panel** | Collapsible panel showing fusion formula, genre membership bar chart, emotion similarity breakdown, and intermediate computation values. |
| **V-A Visualization** | Embedded matplotlib scatter plot of Russell's circumplex (Valence × Arousal) with quadrant labels and glowing dot for current song. |
| **Microphone Recording** | "REC 5s" button records 5 seconds at 22 050 Hz, saves temp WAV, auto-analyzes, then cleans up. Blinking ● indicator during recording. |
| **Singleton Model Loading** | `ModelRegistry` singleton loads all Keras models once at startup with warm-up passes — no reloading on button clicks. Analysis results cached by file path. |
| **Retro Arcade UI** | Dark navy theme, neon accents (cyan / magenta / green / yellow), custom 3D `NeonButton` canvas widgets, box-drawing characters, pixel-style fonts. |

---

## Project Structure

```
Muud/
├── main.py                          # Entry point — launches desktop app
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignores datasets, models, temp files
│
├── engine/                          # Core ML + inference logic
│   ├── __init__.py
│   ├── feature_extraction.py        # load_audio, extract_mel, extract_mel_segments,
│   │                                  extract_handcrafted, extract_handcrafted_segments
│   ├── genre_classifier.py          # GenreClassifier — predict, predict_averaged,
│   │                                  temperature scaling, hybrid genre labeling
│   ├── emotion_regressor.py         # EmotionRegressor — predict, predict_averaged,
│   │                                  mood quadrant with neutral zone
│   ├── fusion.py                    # emotion_similarity, genre_similarity, fused_score
│   │                                  (adaptive weighting for near-neutral emotion)
│   ├── model_registry.py            # ModelRegistry singleton — loads models once, warmup
│   └── recommender.py               # MusicRecommender — analyze, recommend, caching
│
├── ui/                              # Desktop GUI
│   ├── __init__.py
│   └── desktop_app.py               # MuudApp (Tkinter) — retro theme, NeonButton,
│                                      Treeview table, V-A plot, explainability, mic rec
│
├── models/                          # Trained model weights (git-ignored)
│   ├── genre_mel_cnn.keras
│   ├── emotion_hybrid_model.keras
│   └── genre_label_encoder.pkl
│
├── training/                        # Jupyter notebooks
│   ├── genre_mel_training.ipynb     # GTZAN genre CNN training
│   ├── train_hybrid_emotion.ipynb   # DEAM hybrid emotion model training
│   ├── fusion_inference.ipynb       # Inference pipeline test
│   ├── fma_dataset_inspection.ipynb # FMA-medium dataset analysis (for new genre model)
│   └── reports/                     # Saved figures from notebooks
│
├── data/                            # Datasets (git-ignored — download separately)
│   ├── GTZAN/                       # 10 genre folders, 100 × 30 s clips each
│   ├── DEAM/                        # 1802 songs, V-A annotations, features
│   └── FMA/                         # FMA-medium (25 000 tracks)
│
├── test_audio/                      # Sample audio files for quick testing
├── inference/                       # (placeholder for future inference scripts)
└── app/                             # (placeholder for future web app)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv

### 1. Create environment

```bash
conda create -n emotioncnn python=3.10
conda activate emotioncnn
```

### 2. Install dependencies

```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn
pip install sounddevice scipy       # for microphone recording
```

### 3. Download datasets

See the [Datasets](#datasets) section below for download links and placement instructions.

### 4. Train models (or use pre-trained)

Run the training notebooks in order (see [Training the Models](#training-the-models)).

### 5. Launch the app

```bash
cd Muud
python main.py
```

---

## Datasets

All datasets are **git-ignored** due to their size. Download and place them manually.

### GTZAN Genre Collection

- **Used for:** Genre classification CNN  
- **Size:** ~1.2 GB (1 000 clips × 30 s, 10 genres)  
- **Download:** [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- **Placement:** Extract to `data/GTZAN/` so that genre folders (`blues/`, `classical/`, etc.) are directly inside.

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

- **Used for:** Rebuilding genre model with larger/more diverse data  
- **Size:** ~22 GB audio + ~350 MB metadata  
- **Downloads:**
  - Audio: [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip)  
  - Metadata: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)  
  - GitHub: [mdeff/fma](https://github.com/mdeff/fma)  
- **Placement:** Extract to `data/FMA/fma_medium/` and `data/FMA/fma_metadata/`.

---

## Training the Models

Run notebooks from an activated `emotioncnn` environment. Open in Jupyter or VS Code.

### 1. Genre CNN (`training/genre_mel_training.ipynb`)

- Loads GTZAN 30 s clips, splits into 3 s segments
- Extracts 128-bin Mel spectrograms (z-normalized)
- Trains a CNN with softmax output (10 classes)
- Saves → `models/genre_mel_cnn.keras`

### 2. Hybrid Emotion Model (`training/train_hybrid_emotion.ipynb`)

- Loads DEAM audio + per-song V/A annotations
- Splits into 3 s segments; extracts Mel spectrogram + 4 handcrafted features
- Trains a hybrid CNN (Mel branch + dense branch) → 2 outputs (valence, arousal)
- Saves → `models/emotion_hybrid_model.keras`

### 3. FMA Dataset Inspection (`training/fma_dataset_inspection.ipynb`)

- Downloads & extracts FMA-medium audio + metadata
- Loads `tracks.csv` / `genres.csv`, identifies top-level genres
- Counts tracks per genre, selects top 10 for future training
- **Note:** This is analysis only — no training yet.

---

## Running the Desktop App

```bash
conda activate emotioncnn
cd Muud
python main.py
```

### What happens on launch

1. `ModelRegistry` singleton loads both Keras models from `models/`
2. Warm-up forward passes compile TF graphs (first launch slightly slower)
3. Retro-themed Tkinter window opens

### Usage

| Button | Action |
|--------|--------|
| **BROWSE** | Select a `.wav` / `.mp3` / `.flac` / `.ogg` file |
| **ANALYZE** | Run genre + emotion analysis on the selected file |
| **RECOMMEND** | Get top-5 similar songs from the database |
| **EXPLAIN** | Toggle explainability panel (fusion breakdown) |
| **REC 5s** | Record 5 seconds from microphone → auto-analyze |

---

## Architecture Overview

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Audio File  │────▶│ Feature Extract  │────▶│  Genre CNN      │──▶ Fuzzy memberships
│  or Mic Rec  │     │ (Mel + Stats)    │     │  (10-class)     │
└──────────────┘     │ × N segments     │     └─────────────────┘
                     │                  │     ┌─────────────────┐
                     │                  │────▶│  Emotion Hybrid │──▶ Valence, Arousal
                     └─────────────────┘     │  (CNN + Dense)  │
                                              └─────────────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │  Fuzzy Fusion   │
                                              │  (adaptive wt)  │──▶ Ranked recommendations
                                              └─────────────────┘
```

### Key Design Decisions

- **Multi-segment averaging**: Both genre and emotion predictions are averaged across up to 5 non-overlapping 3-second segments for robustness.
- **Temperature scaling**: Optional `softmax(log(p)/T)` on genre probabilities to sharpen or soften the distribution without retraining.
- **Adaptive fusion weights**: When emotion is near-neutral (distance to center < 0.5), genre weight increases to 0.75 since emotion carries less discriminative power.
- **Hybrid genre labeling**: If top-2 genre probabilities differ by < 0.10, the label becomes `"Hybrid: rock / metal"` instead of forcing a single class.
- **Neutral mood zone**: If both valence and arousal are within ±0.5 of the midpoint (5.0), mood is labeled "Neutral / Balanced" instead of forcing a quadrant.

---

## Development Log

| Step | What was done |
|------|---------------|
| 1 | Fixed DEAM training pipeline — path resolution, `iterrows()` float cast, librosa tempo array squeeze |
| 2 | Created hybrid emotion training notebook — CNN + handcrafted features → V/A regression (18 020 segments from 1 802 songs) |
| 3 | Built modular `engine/` package — `feature_extraction`, `genre_classifier`, `emotion_regressor`, `fusion`, `recommender` |
| 4 | Created retro arcade Tkinter GUI — dark navy theme, neon accents, custom `NeonButton` canvas widgets |
| 5 | Added embedded V-A scatter plot (matplotlib `FigureCanvasTkAgg`) with quadrant labels and glowing dot |
| 6 | Refactored recommendations into a sortable `ttk.Treeview` table (7 columns, neon green rank-1 highlight) |
| 7 | Added collapsible explainability panel — fusion formula, genre bars, emotion similarity breakdown |
| 8 | Added microphone recording — 5 s at 22 050 Hz via sounddevice, temp WAV, auto-analyze, blinking indicator |
| 9 | Optimized startup — `ModelRegistry` singleton loads models once, warm-up passes, analysis caching |
| 10 | Multi-segment genre averaging — split audio into N segments, average softmax vectors |
| 11 | Improved mood classification — neutral buffer zone, updated quadrant labels |
| 12 | Multi-segment emotion averaging — average V/A across segments for stability |
| 13 | Hybrid genre labeling — `"Hybrid: X / Y"` when top-2 gap < 0.10 |
| 14 | Adaptive fusion weights — shift to 0.75/0.25 genre/emotion when near-neutral |
| 15 | Temperature scaling — optional `softmax(log(p)/T)` on genre probabilities |
| 16 | FMA-medium dataset inspection notebook — download, extract, analyze genre distribution |

---

## License

This project is for academic/educational purposes.

- **GTZAN:** Research use — [Marsyas](http://marsyas.info/downloads/datasets.html)
- **DEAM:** Creative Commons — [MediaEval](https://cvml.unige.ch/databases/DEAM/)
- **FMA:** Creative Commons — [GitHub](https://github.com/mdeff/fma)
