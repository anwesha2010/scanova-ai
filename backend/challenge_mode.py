"""
AI vs Human Challenge Mode
==========================
Educational exhibition feature. Uses pre-loaded sample images
and compares the user's guess with the AI's prediction.
"""

import streamlit as st


# Sample cases — use existing project images
CHALLENGE_CASES = [
    {
        "id": 1,
        "image_path": "assets/sample_xray.png",
        "label": "possible_abnormality",
        "explanation": "AI flagged 3 regions of interest in the lung fields.",
    },
    {
        "id": 2,
        "image_path": "branding/hero_xray.png",
        "label": "normal",
        "explanation": "AI found no significant regions. This image resembles typical training data.",
    },
    {
        "id": 3,
        "image_path": "branding/hero_mri.png",
        "label": "normal",
        "explanation": "AI detected no anomalies. Structure appears symmetric.",
    },
    {
        "id": 4,
        "image_path": "branding/hero_bone.png",
        "label": "possible_abnormality",
        "explanation": "AI flagged a density discontinuity near the joint region.",
    },
    {
        "id": 5,
        "image_path": "assets/sample_xray.png",
        "label": "possible_abnormality",
        "explanation": "AI flagged 2 regions. Please review carefully.",
    },
]


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


def get_current_case():
    idx = st.session_state.challenge_round % len(CHALLENGE_CASES)
    return CHALLENGE_CASES[idx]


def record_guess(guess):
    """Record user's guess and reveal AI's answer."""
    case = get_current_case()
    ai_guess = case["label"]

    if guess == ai_guess:
        st.session_state.challenge_score_user += 1

    # AI always gets it right (ground truth)
    st.session_state.challenge_score_ai += 1
    st.session_state.challenge_revealed = True


def next_round():
    st.session_state.challenge_round += 1
    st.session_state.challenge_revealed = False

    if st.session_state.challenge_round >= 5:
        st.session_state.challenge_finished = True


def reset_challenge():
    st.session_state.challenge_round = 0
    st.session_state.challenge_score_user = 0
    st.session_state.challenge_score_ai = 0
    st.session_state.challenge_finished = False
    st.session_state.challenge_revealed = False


def exit_challenge():
    st.session_state.challenge_active = False
    reset_challenge()