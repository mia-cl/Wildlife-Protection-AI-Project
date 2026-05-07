# Wildlife Protection AI Project

A Streamlit-based portfolio prototype demonstrating how AI can support wildlife conservation by analysing forest audio recordings and identifying suspicious sound patterns such as vehicles, chainsaws, gunshot-like sounds or unusual human activity.

## Live Demo

Deploy this project with Streamlit Community Cloud and use:

```text
src/app.py
```

as the main file path.

## Project Features

- Interview-ready Streamlit interface
- Built-in quick demo scenarios
- Audio upload interface
- Simulated risk-level prediction output
- Machine learning workflow structure
- Active learning concept for selecting uncertain samples
- Portfolio-friendly GitHub documentation

## Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn workflow design
- Audio feature extraction concept
- Active learning strategy

## Demo Scenarios

The app includes three built-in demo scenarios:

1. Normal Forest Sound — Low Risk
2. Vehicle-like Sound — Medium Risk
3. Chainsaw-like Sound — High Risk

These demo scenarios make the project easier to present in interviews because reviewers do not need to prepare external audio files.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure

```text
Wildlife-Protection-AI-Project/
├── README.md
├── requirements.txt
├── src/
│   ├── app.py
│   ├── active_learning.py
│   ├── config.py
│   ├── feature_extraction.py
│   ├── predict.py
│   └── train_model.py
├── data/
│   └── sample_metadata.csv
├── models/
│   └── .gitkeep
├── assets/
│   └── project_summary.md
└── notebooks/
    └── project_workflow.md
```

## Important Note

This is a portfolio prototype, not a production wildlife surveillance system. The Streamlit interface uses simulated demo outputs to show the intended user workflow. A production-ready version would require a labelled wildlife audio dataset, trained model files, validation metrics, field testing and integration with real acoustic sensors.

## Resume Description

**Wildlife Protection AI Project | Python, Machine Learning, Streamlit**

Developed a Streamlit-based AI portfolio prototype to demonstrate how forest audio recordings could be analysed for suspicious activity detection. Designed an audio classification workflow, built an interview-ready demo interface, and included active learning logic to show how uncertain samples could be prioritised for human review.
