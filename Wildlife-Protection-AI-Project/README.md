# Wildlife Protection AI Project

An AI-based wildlife protection project designed to support early detection of illegal poaching activities in forest environments using acoustic signal analysis and active learning.

## Project Overview

Illegal poaching is difficult to detect in large forest areas because human patrols are limited by distance, time, terrain and cost. This project explores how artificial intelligence can assist wildlife protection teams by analysing environmental audio and identifying suspicious sounds such as gunshots, chainsaws, vehicles or unusual human activity.

The system is designed as a prototype for conservation monitoring. It uses audio feature extraction, machine learning classification and an active learning workflow to improve model performance with limited labelled data.

## Key Features

- Audio-based detection of potential poaching-related sounds
- Feature extraction from forest sound recordings
- Machine learning classification pipeline
- Active learning strategy to prioritise uncertain samples for manual review
- Simple prediction script for new audio files
- GitHub-ready structure for portfolio demonstration

## Technology Stack

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Joblib
- Streamlit optional demo interface

## Project Structure

```text
Wildlife-Protection-AI-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── sample_metadata.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── project_workflow.md
│
├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── predict.py
│   ├── active_learning.py
│   └── app.py
│
└── assets/
    └── project_summary.md
```

## Dataset Format

This repository does not include a real wildlife audio dataset. To train the model, place audio files in a local dataset folder and prepare a metadata CSV file in the following format:

```csv
file_path,label
path/to/audio_001.wav,gunshot
path/to/audio_002.wav,chainsaw
path/to/audio_003.wav,vehicle
path/to/audio_004.wav,normal_forest
```

Suggested labels:

- `gunshot`
- `chainsaw`
- `vehicle`
- `human_voice`
- `normal_forest`

## How It Works

### 1. Audio Feature Extraction

The system extracts audio features such as:

- MFCCs
- Chroma features
- Spectral centroid
- Spectral rolloff
- Zero crossing rate
- Root mean square energy

These features help the model identify patterns in different environmental sounds.

### 2. Model Training

The training script reads the metadata file, extracts features from each audio file and trains a Random Forest classifier.

```bash
python src/train_model.py --metadata data/sample_metadata.csv --model models/wildlife_audio_model.joblib
```

### 3. Prediction

After training, the model can predict the class of a new audio file.

```bash
python src/predict.py --audio path/to/new_audio.wav --model models/wildlife_audio_model.joblib
```

### 4. Active Learning

The active learning module identifies samples where the model is least confident. These uncertain samples can be prioritised for human labelling, helping improve the model efficiently when labelled data is limited.

```bash
python src/active_learning.py --metadata data/unlabelled_metadata.csv --model models/wildlife_audio_model.joblib
```

### 5. Optional Streamlit Demo

A simple Streamlit interface is included for portfolio demonstration.

```bash
streamlit run src/app.py
```

## Example Use Case

1. Forest sensors collect environmental audio.
2. The AI model analyses the audio and classifies the sound.
3. If the sound is suspicious, the system can flag it for review.
4. Conservation staff can prioritise high-risk areas for investigation.
5. Uncertain predictions are added to an active learning queue for further labelling.

## Resume Project Description

**Wildlife Protection AI Project**  
Developed an AI-based prototype to support wildlife conservation by detecting potential illegal poaching activities from forest audio recordings. Built a machine learning pipeline for audio feature extraction, sound classification and active learning-based sample selection. The project demonstrates practical experience in Python, machine learning, data preprocessing and AI application design for real-world environmental protection use cases.

## Future Improvements

- Use deep learning models such as CNNs on spectrogram images
- Add real-time audio stream processing
- Integrate geolocation and alert notification features
- Connect with IoT forest monitoring devices
- Build a dashboard for conservation teams

## Disclaimer

This project is a portfolio prototype for educational and demonstration purposes. Real-world deployment would require validated datasets, field testing, sensor integration and expert review.
