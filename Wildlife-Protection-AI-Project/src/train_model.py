"""Train a wildlife audio classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE
from feature_extraction import extract_features


def build_dataset(metadata_path: str):
    """Build feature and label arrays from a metadata CSV file."""
    metadata = pd.read_csv(metadata_path)

    if not {"file_path", "label"}.issubset(metadata.columns):
        raise ValueError("Metadata CSV must contain 'file_path' and 'label' columns.")

    features = []
    labels = []

    for _, row in metadata.iterrows():
        file_path = row["file_path"]
        label = row["label"]
        try:
            features.append(extract_features(file_path))
            labels.append(label)
        except Exception as exc:
            print(f"Skipped {file_path}: {exc}")

    if not features:
        raise ValueError("No valid audio features were extracted. Please check your audio paths.")

    return features, labels


def train(metadata_path: str, model_output_path: str):
    """Train and save the classification model."""
    x, y = build_dataset(metadata_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(set(y)) > 1 else None,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wildlife audio classifier")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--model", default="models/wildlife_audio_model.joblib", help="Output model path")
    args = parser.parse_args()

    train(args.metadata, args.model)
