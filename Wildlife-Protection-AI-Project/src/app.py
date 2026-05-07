"""Optional Streamlit demo app for wildlife audio prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from predict import predict

st.set_page_config(page_title="Wildlife Protection AI", page_icon="🦌")

st.title("Wildlife Protection AI Project")
st.write(
    "Upload a forest audio recording and the model will classify whether it may contain "
    "a suspicious sound such as a gunshot, chainsaw, vehicle or other activity."
)

model_path = st.text_input(
    "Model path",
    value="models/wildlife_audio_model.joblib",
)

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac"],
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    if st.button("Run Prediction"):
        try:
            label, confidence, probability_map = predict(temp_path, model_path)
            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.bar_chart(probability_map)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
