"""Active learning helper for selecting uncertain audio samples."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from feature_extraction import extract_features


def find_uncertain_samples(metadata_path: str, model_path: str, top_k: int = 10):
    """Return audio samples with the lowest prediction confidence."""
    metadata = pd.read_csv(metadata_path)

    if "file_path" not in metadata.columns:
        raise ValueError("Metadata CSV must contain a 'file_path' column.")

    model = joblib.load(model_path)
    results = []

    for _, row in metadata.iterrows():
        file_path = row["file_path"]
        try:
            features = extract_features(file_path).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            predicted_label = model.classes_[probabilities.argmax()]
            confidence = probabilities.max()
            uncertainty = 1 - confidence

            results.append({
                "file_path": file_path,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "uncertainty": uncertainty,
            })
        except Exception as exc:
            print(f"Skipped {file_path}: {exc}")

    uncertain_samples = pd.DataFrame(results).sort_values(
        by="uncertainty",
        ascending=False,
    )

    return uncertain_samples.head(top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select uncertain samples for active learning")
    parser.add_argument("--metadata", required=True, help="Path to unlabelled metadata CSV")
    parser.add_argument("--model", default="models/wildlife_audio_model.joblib", help="Path to trained model")
    parser.add_argument("--top_k", type=int, default=10, help="Number of samples to return")
    args = parser.parse_args()

    output = find_uncertain_samples(args.metadata, args.model, args.top_k)
    print(output.to_string(index=False))
