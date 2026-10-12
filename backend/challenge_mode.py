"""
AI vs Human Challenge Mode
==========================
Real scoring: the AI's predictions are compared against true labels.
No fake 5/5 scores. Honest exhibition.

Uses images from test_data with known labels.
"""

import os
import cv2
import numpy as np
import streamlit as st

from backend.dl_analyzer import predict_pneumonia


# ---------- Challenge cases pulled from test_data ----------
CHALLENGE_CASES = [
    {"source": "normal",    "index": 0},
    {"source": "pneumonia", "index": 0},
    {"source": "normal",    "index": 1},
    {"source": "pneumonia", "index": 1},
    {"source": "pneumonia", "index": 2},
]

NORMAL_DIR = "test_data/normal"
PNEUMONIA_DIR = "test_data/pneumonia"


def init_challenge():
    if "challenge_active" not in st.session_state:
        st.session_state.challenge_active = False
    if "challenge_round" not in st.session_state:
        st.session_state.challenge_round = 0
    if "challenge_score_user" not in st.session_state:
        st.session_state.challenge_score_user = 0
    if "challenge_score_ai" not in st.session_state:
        st.session_state.challenge_score_ai = 0
    if "challenge_finished" not in st.session_state:
        st.session_state.challenge_finished = False
    if "challenge_revealed" not in st.session_state:
        st.session_state.challenge_revealed = False
    if "challenge_ai_prediction" not in st.session_state:
        st.session_state.challenge_ai_prediction = None


def _load_case_image(case):
    """Load a real image from test_data."""
    if case["source"] == "normal":
        folder = NORMAL_DIR
        true_label = "normal"
    else:
        folder = PNEUMONIA_DIR
        true_label = "pneumonia"

    if not os.path.exists(folder):
        return None, true_label

    files = sorted(os.listdir(folder))
    if case["index"] >= len(files):
        return None, true_label

    path = os.path.join(folder, files[case["index"]])
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img, true_label


def get_current_case():
    idx = st.session_state.challenge_round % len(CHALLENGE_CASES)
    case = CHALLENGE_CASES[idx]
    img, true_label = _load_case_image(case)
    return {
        "image": img,
        "true_label": true_label,
        "source": case["source"],
        "index": case["index"],
    }


def record_guess(user_guess):
    """Compare user guess AND AI prediction against true label."""
    case = get_current_case()
    true_label = case["true_label"]
    img = case["image"]

    if user_guess == true_label:
        st.session_state.challenge_score_user += 1

    ai_prediction = "unknown"
    ai_confidence = 0.0

    if img is not None:
        try:
            result = predict_pneumonia(img)
            ai_prediction = result["prediction"]
            ai_confidence = result["confidence"]
            if ai_prediction == true_label:
                st.session_state.challenge_score_ai += 1
        except Exception:
            pass

    st.session_state.challenge_ai_prediction = {
        "prediction": ai_prediction,
        "confidence": ai_confidence,
        "true_label": true_label,
    }
    st.session_state.challenge_revealed = True


def next_round():
    st.session_state.challenge_round += 1
    st.session_state.challenge_revealed = False
    st.session_state.challenge_ai_prediction = None

    if st.session_state.challenge_round >= 5:
        st.session_state.challenge_finished = True


def reset_challenge():
    st.session_state.challenge_round = 0
    st.session_state.challenge_score_user = 0
    st.session_state.challenge_score_ai = 0
    st.session_state.challenge_finished = False
    st.session_state.challenge_revealed = False
    st.session_state.challenge_ai_prediction = None


def exit_challenge():
    st.session_state.challenge_active = False
    reset_challenge()