"""Predict the class of a new wildlife audio recording."""

from __future__ import annotations

import argparse

import joblib

from feature_extraction import extract_features


def predict(audio_path: str, model_path: str):
    """Predict label and probability for one audio file."""
    model = joblib.load(model_path)
    features = extract_features(audio_path).reshape(1, -1)

    label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    confidence = max(probabilities)
    probability_map = dict(zip(classes, probabilities))

    return label, confidence, probability_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wildlife audio class")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="models/wildlife_audio_model.joblib", help="Path to trained model")
    args = parser.parse_args()

    predicted_label, confidence_score, class_probabilities = predict(args.audio, args.model)

    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence_score:.2%}")
    print("Class probabilities:")
    for class_name, probability in class_probabilities.items():
        print(f"- {class_name}: {probability:.2%}")
