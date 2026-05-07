"""Streamlit demo app for the Wildlife Protection AI Project."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Wildlife Protection AI", page_icon="🦌", layout="centered")

st.title("🦌 Wildlife Protection AI Project")
st.write(
    "This portfolio prototype demonstrates how AI can support wildlife conservation "
    "by analysing forest audio recordings and identifying suspicious sounds such as "
    "gunshots, chainsaws, vehicles or unusual human activity."
)

st.subheader("Project Highlights")
st.markdown(
    """
- Audio feature extraction from environmental sound recordings
- Machine learning classification workflow
- Active learning strategy for selecting uncertain samples
- Streamlit interface for portfolio demonstration
"""
)

st.info(
    "Note: This repository is a portfolio prototype. To run real predictions, a trained "
    "model file must be available in the models folder."
)

model_path = st.text_input(
    "Model path",
    value="models/wildlife_audio_model.joblib",
)

uploaded_file = st.file_uploader(
    "Upload an audio file for prediction",
    type=["wav", "mp3", "flac"],
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Prediction"):
        try:
            from predict import predict

            label, confidence, probability_map = predict(temp_path, model_path)
            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.bar_chart(probability_map)
        except FileNotFoundError:
            st.error(
                "Model file not found. Please train the model first and save it as "
                "models/wildlife_audio_model.joblib."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.subheader("How to Run Locally")
st.code(
    "pip install -r requirements.txt\nstreamlit run src/app.py",
    language="bash",
)

st.subheader("GitHub Repository")
st.write("https://github.com/mia-cl/Wildlife-Protection-AI-Project")
