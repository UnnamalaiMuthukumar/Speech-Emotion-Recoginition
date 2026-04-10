# Speech Emotion Detection using MLP

Detects human emotions from speech audio using a Multi-Layer Perceptron (MLP) classifier. 
Trained on the RAVDESS dataset, achieving **82.37% accuracy** across 4 emotion classes.

## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | **82.37%** |
| Training Samples | 985 |
| Test Samples | 329 |
| Audio Features | 180 (MFCC + Chroma + MEL) |
| Emotions Detected | Angry, Happy, Neutral, Sad |

## Classification Report

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry   | 0.94      | 0.86   | 0.90     |
| Happy   | 0.81      | 0.80   | 0.80     |
| Neutral | 0.70      | 0.73   | 0.71     |
| Sad     | 0.81      | 0.86   | 0.83     |

## How It Works

1. Audio file (.wav) is loaded using `librosa` and `soundfile`
2. Three audio features are extracted:
   - **MFCC** (Mel Frequency Cepstral Coefficients) — 40 features
   - **Chroma** — 12 features
   - **MEL Spectrogram** — 128 features
3. Features are concatenated into a 180-dimensional vector
4. MLP Classifier predicts one of 4 emotions

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | MLP Classifier |
| Librosa | Audio feature extraction |
| SoundFile | Audio file reading |
| NumPy | Numerical operations |
| Google Colab | Training environment |

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- This project uses 4 emotions: angry, happy, neutral, sad

## Model Architecture

```
MLP Classifier
├── Hidden Layer: 300 neurons
├── Learning Rate: adaptive
├── Alpha: 0.01
├── Batch Size: 256
└── Max Iterations: 500
```

## How to Run

```bash
# Install dependencies
pip install librosa soundfile scikit-learn numpy

# Run in Google Colab
# Upload RAVDESS dataset to Google Drive at: /content/drive/My Drive/wav/
# Run all cells in Speechemotion_mlp.ipynb
```

#output
<img width="445" height="148" alt="Screenshot 2026-04-10 144518" src="https://github.com/user-attachments/assets/504287b8-f806-492e-bd7c-f02bbbc3ea0a" />

<img width="718" height="431" alt="Screenshot 2026-04-10 144534" src="https://github.com/user-attachments/assets/f0c53582-f7b4-43d6-9b7d-2f40ab2114f5" />


## Key Findings

- Best performance on **Angry** class (94% precision)
- Model generalizes well across gender (male/female actors)
- MFCC features contribute most to classification accuracy
- Batch size 256 outperformed batch size 200 (82.37% vs 80.24%)

## Author

Unnamalai Muthukumar
BVoc AI & ML | MS-CSDA @ IIT Patna (Pursuing)
[LinkedIn] www.linkedin.com/in/unnamalai-muthukumar | [GitHub] https://github.com/UnnamalaiMuthukumar
