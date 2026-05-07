# Project Workflow

## Step 1: Problem Definition

The goal is to support wildlife protection teams by detecting suspicious acoustic events in forest areas. The model focuses on audio events that may indicate illegal poaching activities, such as gunshots, chainsaws, vehicles or unusual human presence.

## Step 2: Data Collection

Audio recordings can be collected from field sensors, public environmental sound datasets or simulated recordings for prototype testing.

## Step 3: Feature Engineering

The system converts raw audio into numerical features. Useful audio features include MFCCs, spectral centroid, spectral rolloff, chroma features, RMS energy and zero crossing rate.

## Step 4: Model Training

A Random Forest classifier is used as the baseline model because it is interpretable, handles structured numerical features well and is suitable for an initial prototype.

## Step 5: Active Learning

Because labelled wildlife audio can be limited, active learning is used to identify uncertain predictions. Human reviewers can label these uncertain samples first, improving model performance with fewer labelled examples.

## Step 6: Evaluation

The model can be evaluated using accuracy, precision, recall and F1-score. In real-world conservation work, recall for high-risk classes such as gunshot or chainsaw may be more important than overall accuracy.

## Step 7: Future Deployment

Future versions could integrate sensor networks, real-time alerts, geolocation and a dashboard for conservation staff.
