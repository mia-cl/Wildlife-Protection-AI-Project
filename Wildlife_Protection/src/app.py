import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wildlife Protection AI Project",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f5132 0%, #1f7a4d 48%, #8fbf75 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 16px 42px rgba(15, 81, 50, 0.18);
    }
    .hero h1 {
        font-size: 2.65rem;
        line-height: 1.08;
        margin: 0 0 0.75rem 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 900px;
        opacity: 0.96;
        margin: 0;
    }
    .pill-row { margin-top: 1.2rem; }
    .pill {
        display: inline-block;
        padding: 0.42rem 0.75rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.28);
        color: white;
        font-size: 0.88rem;
    }
    .section-title {
        margin-top: 2rem;
        margin-bottom: 0.7rem;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .soft-card {
        padding: 1.25rem 1.25rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e8ece8;
        box-shadow: 0 8px 28px rgba(20, 54, 37, 0.06);
        height: 100%;
    }
    .soft-card h3 {
        margin-top: 0;
        font-size: 1.1rem;
        letter-spacing: -0.01em;
    }
    .muted { color: #667085; font-size: 0.94rem; line-height: 1.55; }
    .result-card {
        padding: 1.35rem 1.45rem;
        border-radius: 24px;
        background: #f7faf7;
        border: 1px solid #dfe9df;
        margin-top: 1rem;
    }
    .risk-low { color: #16794c; font-weight: 800; }
    .risk-medium { color: #b7791f; font-weight: 800; }
    .risk-high { color: #b42318; font-weight: 800; }
    .small-caption { color: #667085; font-size: 0.84rem; margin-top: 0.4rem; }
    .footer-box {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: #f2f6f3;
        border: 1px solid #e1e8e2;
        color: #475467;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8ece8;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(20, 54, 37, 0.04);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEMO_CASES = {
    "Normal Forest Sound": {
        "icon": "🌲",
        "risk": "Low Risk",
        "risk_class": "risk-low",
        "label": "Natural Forest Ambience",
        "confidence": 0.91,
        "summary": "Background forest ambience with no strong suspicious acoustic pattern detected.",
        "recommendation": "No immediate action required. Continue passive monitoring.",
    },
    "Vehicle-like Sound": {
        "icon": "🚙",
        "risk": "Medium Risk",
        "risk_class": "risk-medium",
        "label": "Vehicle-like Activity",
        "confidence": 0.84,
        "summary": "Low-frequency engine-like patterns may indicate vehicle movement near a protected area.",
        "recommendation": "Flag for ranger review and compare with nearby sensor activity.",
    },
    "Chainsaw-like Sound": {
        "icon": "⚠️",
        "risk": "High Risk",
        "risk_class": "risk-high",
        "label": "Chainsaw / Mechanical Cutting Sound",
        "confidence": 0.88,
        "summary": "Sustained mechanical frequency patterns may indicate illegal logging or poaching-related activity.",
        "recommendation": "Escalate for urgent human review and location-based field verification.",
    },
}


def deterministic_upload_result(uploaded_file):
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    index = int(digest[:2], 16) % 3
    case_name = list(DEMO_CASES.keys())[index]
    result = DEMO_CASES[case_name].copy()
    result["filename"] = uploaded_file.name
    result["confidence"] = round(0.76 + (int(digest[2:4], 16) % 18) / 100, 2)
    return result


def show_result(result, source_label="Demo Result"):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"### {result['icon']} {source_label}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted class", result["label"])
    with c2:
        st.metric("Confidence", f"{int(result['confidence'] * 100)}%")
    with c3:
        st.markdown("Risk level")
        st.markdown(f"<div class='{result['risk_class']}'>{result['risk']}</div>", unsafe_allow_html=True)
    st.write(result["summary"])
    st.info(result["recommendation"])
    st.markdown("<div class='small-caption'>This is a portfolio prototype. Demo outputs are simulated to show the intended user workflow.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.title("🌿 Project Menu")
    st.markdown("**Wildlife Protection AI Project**")
    st.caption("Portfolio prototype for AI-assisted conservation monitoring.")
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("- Python\n- Streamlit\n- Machine Learning workflow\n- Audio feature extraction\n- Active learning design")
    st.divider()
    st.markdown("**Links**")
    st.link_button("GitHub Repository", "https://github.com/mia-cl/Wildlife-Protection-AI-Project")

st.markdown(
    """
    <div class="hero">
        <h1>Wildlife Protection AI Project</h1>
        <p>
            A portfolio prototype showing how AI can support wildlife conservation by analysing forest audio recordings
            and identifying suspicious sounds such as vehicles, chainsaws, gunshot-like patterns, or unusual human activity.
        </p>
        <div class="pill-row">
            <span class="pill">Python</span>
            <span class="pill">Streamlit</span>
            <span class="pill">Machine Learning</span>
            <span class="pill">Audio Analysis</span>
            <span class="pill">Active Learning</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Demo modes", "3")
with m2:
    st.metric("Input type", "Audio")
with m3:
    st.metric("Output", "Risk label")
with m4:
    st.metric("Purpose", "Portfolio")

st.markdown("<div class='section-title'>Quick Demo</div>", unsafe_allow_html=True)
st.write("Click a scenario below to see how the system would present a detection result during an interview or portfolio review.")

cols = st.columns(3)
for idx, (name, case) in enumerate(DEMO_CASES.items()):
    with cols[idx]:
        st.markdown(
            f"""
            <div class="soft-card">
                <h3>{case['icon']} {name}</h3>
                <p class="muted">{case['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Run {name}", key=f"demo_{idx}", use_container_width=True):
            st.session_state["selected_result"] = case
            st.session_state["selected_source"] = name

if "selected_result" in st.session_state:
    show_result(st.session_state["selected_result"], st.session_state.get("selected_source", "Demo Result"))
else:
    st.markdown(
        "<div class='footer-box'>Tip: For interviews, start with the <b>Chainsaw-like Sound</b> demo to quickly show the full project workflow and risk output.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Upload Your Own Audio</div>", unsafe_allow_html=True)
st.write("Upload a WAV, MP3 or FLAC file to test the interface. In this portfolio version, upload results are simulated for demonstration.")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner("Analysing audio features..."):
        time.sleep(0.8)
    upload_result = deterministic_upload_result(uploaded_file)
    show_result(upload_result, f"Uploaded file: {uploaded_file.name}")

st.markdown("<div class='section-title'>Project Workflow</div>", unsafe_allow_html=True)
w1, w2, w3, w4 = st.columns(4)
workflow = [
    ("1", "Audio Input", "Collect environmental sound recordings from protected areas."),
    ("2", "Feature Extraction", "Convert raw audio into machine-readable acoustic features."),
    ("3", "Classification", "Predict whether the sound pattern is normal or suspicious."),
    ("4", "Active Learning", "Prioritise uncertain samples for human review and model improvement."),
]
for col, (num, title, text) in zip([w1, w2, w3, w4], workflow):
    with col:
        st.markdown(
            f"""
            <div class="soft-card">
                <h3>{num}. {title}</h3>
                <p class="muted">{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div class='section-title'>How to Run Locally</div>", unsafe_allow_html=True)
st.code("pip install -r requirements.txt\nstreamlit run src/app.py", language="bash")

st.markdown("<div class='section-title'>Project Notes</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer-box">
        This project is designed for portfolio and interview demonstration. It presents a realistic AI product workflow,
        including audio upload, risk-level output, active learning design and a simple reviewer-friendly interface.
        A production version would require a larger labelled audio dataset, model validation, field testing and integration
        with real sensor devices.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Last updated for portfolio demo: {datetime.now().strftime('%Y-%m-%d')}")
