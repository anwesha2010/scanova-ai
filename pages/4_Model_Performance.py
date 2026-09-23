import streamlit as st
import os
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from backend.dl_analyzer import predict_pneumonia

st.set_page_config(
    page_title="Model Performance — SCANOVA",
    page_icon="📊",
    layout="wide",
)

try:
    with open("branding/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("📊 Model Performance")
st.caption("Real accuracy, computed live on labeled test images.")

st.warning(
    "⚠️ **Not clinical validation.** These metrics come from a small "
    "in-house test set. Real clinical validation requires hospital "
    "partnership and regulatory review."
)

TEST_DIR = "test_data"
NORMAL_DIR = os.path.join(TEST_DIR, "normal")
PNEUMONIA_DIR = os.path.join(TEST_DIR, "pneumonia")

if not os.path.exists(NORMAL_DIR) or not os.path.exists(PNEUMONIA_DIR):
    st.error(
        "📁 Test data not found. Add `test_data/normal/` "
        "and `test_data/pneumonia/` folders."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_test_set():
    images, labels = [], []
    for fname in os.listdir(NORMAL_DIR):
        img = cv2.imread(os.path.join(NORMAL_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(cv2.resize(img, (512, 512)))
        labels.append(0)
    for fname in os.listdir(PNEUMONIA_DIR):
        img = cv2.imread(os.path.join(PNEUMONIA_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(cv2.resize(img, (512, 512)))
        labels.append(1)
    return images, np.array(labels)


images, labels = load_test_set()

st.markdown(f"### Test Set: {len(images)} images")
c1, c2 = st.columns(2)
c1.metric("Normal", int(np.sum(labels == 0)))
c2.metric("Pneumonia", int(np.sum(labels == 1)))

st.divider()

if st.button("🚀 Run Live Evaluation", type="primary"):
    with st.spinner(f"Running AI on {len(images)} images..."):
        predictions = []
        confidences = []
        progress = st.progress(0)

        for i, img in enumerate(images):
            result = predict_pneumonia(img)
            pred = 1 if result["prediction"] == "pneumonia" else 0
            predictions.append(pred)
            confidences.append(result["confidence"])
            progress.progress((i + 1) / len(images))

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    cm = confusion_matrix(labels, predictions)

    st.divider()
    st.markdown("### 🎯 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc * 100:.1f}%")
    m2.metric("Precision", f"{prec * 100:.1f}%")
    m3.metric("Recall", f"{rec * 100:.1f}%")
    m4.metric("F1 Score", f"{f1 * 100:.1f}%")

    st.divider()
    st.markdown("### Confusion Matrix")

    st.markdown(f"""
|  | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | {cm[0][0]} ✅ | {cm[0][1]} ❌ |
| **Actual Pneumonia** | {cm[1][0]} ❌ | {cm[1][1]} ✅ |
""")

    st.info(
        f"**True Positives:** {cm[1][1]} · "
        f"**True Negatives:** {cm[0][0]} · "
        f"**False Positives:** {cm[0][1]} · "
        f"**False Negatives:** {cm[1][0]}"
    )

    st.success(
        f"✅ **Accuracy on this test set: {acc * 100:.1f}%** "
        f"({int(np.sum(predictions == labels))} of {len(labels)} correct)"
    )

    st.caption(
        "This is a small, in-house test set. Real clinical validation "
        "requires thousands of images from multiple hospitals."
    )