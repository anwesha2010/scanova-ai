import streamlit as st
import os
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from backend.dl_analyzer import predict_pneumonia

# Try importing Grad-CAM (optional — requires torch, not available on Cloud free tier)
try:
    from backend.grad_cam import generate_gradcam
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False
    def generate_gradcam(*args, **kwargs):
        return {
            "success": False,
            "error": "Grad-CAM unavailable on this deployment (torch not installed).",
        }


# ==========================================================
# PAGE SETUP
# ==========================================================
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


# ==========================================================
# HEADER
# ==========================================================
st.title("📊 Model Performance")
st.caption("Real accuracy + Grad-CAM explainability, computed live.")

st.warning(
    "⚠️ **Not clinical validation.** These metrics come from a small "
    "in-house test set. Real clinical validation requires hospital "
    "partnership and regulatory review."
)


# ==========================================================
# TEST DATA PATHS
# ==========================================================
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


# ==========================================================
# TABS
# ==========================================================
tab_metrics, tab_gradcam = st.tabs(
    ["📈 Live Metrics", "🔥 Grad-CAM Explainability"]
)


# ==========================================================
# TAB 1 — LIVE METRICS
# ==========================================================
with tab_metrics:
    st.markdown(f"### Test Set: {len(images)} images")
    c1, c2 = st.columns(2)
    c1.metric("Normal", int(np.sum(labels == 0)))
    c2.metric("Pneumonia", int(np.sum(labels == 1)))

    st.divider()

    if st.button("🚀 Run Live Evaluation", type="primary", key="eval_btn"):
        with st.spinner(f"Running AI on {len(images)} images..."):
            predictions = []
            progress = st.progress(0)

            for i, img in enumerate(images):
                result = predict_pneumonia(img)
                pred = 1 if result["prediction"] == "pneumonia" else 0
                predictions.append(pred)
                progress.progress((i + 1) / len(images))

        predictions = np.array(predictions)

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


# ==========================================================
# TAB 2 — GRAD-CAM EXPLAINABILITY
# ==========================================================
with tab_gradcam:
    st.markdown("### 🔥 Grad-CAM Explainability")
    st.caption(
        "Grad-CAM shows WHERE the model looked to make its decision. "
        "Red = high attention, blue = low attention."
    )

    st.info(
        "**What is Grad-CAM?** Gradient-weighted Class Activation Mapping uses "
        "gradients from the model's last convolutional layer to highlight the "
        "regions that most influenced its prediction."
    )

    st.divider()

    image_source = st.radio(
        "Choose image source:",
        [
            "Sample from test set (pneumonia)",
            "Sample from test set (normal)",
            "Upload your own",
        ],
        horizontal=True,
    )

    selected_image = None
    selected_label = ""

    if image_source == "Sample from test set (pneumonia)":
        files = sorted(os.listdir(PNEUMONIA_DIR))[:10]
        selected_file = st.selectbox("Pick a pneumonia image:", files)
        selected_image = cv2.imread(
            os.path.join(PNEUMONIA_DIR, selected_file),
            cv2.IMREAD_GRAYSCALE,
        )
        selected_label = f"Pneumonia: {selected_file}"

    elif image_source == "Sample from test set (normal)":
        files = sorted(os.listdir(NORMAL_DIR))[:10]
        selected_file = st.selectbox("Pick a normal image:", files)
        selected_image = cv2.imread(
            os.path.join(NORMAL_DIR, selected_file),
            cv2.IMREAD_GRAYSCALE,
        )
        selected_label = f"Normal: {selected_file}"

    else:
        uploaded = st.file_uploader(
            "Upload a chest X-ray", type=["png", "jpg", "jpeg"]
        )
        if uploaded is not None:
            import io
            from PIL import Image
            pil_img = Image.open(io.BytesIO(uploaded.read())).convert("L")
            selected_image = np.array(pil_img, dtype=np.uint8)
            selected_label = f"Uploaded: {uploaded.name}"

    if selected_image is not None:
        st.divider()

        prediction_result = predict_pneumonia(selected_image)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📷 Original")
            st.image(
                selected_image,
                caption=selected_label,
                use_container_width=True,
            )

            st.markdown("#### 🎯 Prediction")
            pred = prediction_result["prediction"]
            conf = prediction_result["confidence"]
            if pred == "pneumonia":
                st.error(f"**{pred.upper()}** · Confidence: **{conf * 100:.1f}%**")
            else:
                st.success(f"**{pred.upper()}** · Confidence: **{conf * 100:.1f}%**")

            st.markdown("**Probabilities:**")
            st.markdown(
                f"- Normal: `{prediction_result['probabilities']['normal']:.3f}`"
            )
            st.markdown(
                f"- Pneumonia: `{prediction_result['probabilities']['pneumonia']:.3f}`"
            )

        with col_b:
            st.markdown("#### 🔥 Grad-CAM Overlay")

            if not GRADCAM_AVAILABLE:
                st.info(
                    "**Grad-CAM runs on the local deployment.** "
                    "Streamlit Cloud's free tier doesn't support PyTorch "
                    "(required for gradient computation)."
                )
                st.markdown(
                    "**Deployment note:** Grad-CAM is available in the "
                    "local version of SCANOVA. During the InnoEx demo, we run "
                    "the local app to show gradient-based explainability."
                )
            else:
                gradcam_result = generate_gradcam(
                    selected_image, target_class=1
                )

                if gradcam_result["success"]:
                    st.image(
                        gradcam_result["overlay"],
                        caption="Grad-CAM: where the model looked",
                        channels="BGR",
                        use_container_width=True,
                    )
                    st.caption(
                        "**Red = high attention** | **Blue = low attention** "
                        "| The model's decision was driven by the red regions."
                    )
                else:
                    st.warning(
                        f"Grad-CAM unavailable: "
                        f"{gradcam_result.get('error', 'unknown error')}"
                    )

        st.divider()
        st.markdown("#### 💡 Interpretation")
        if prediction_result["prediction"] == "pneumonia":
            st.markdown(
                "The model predicts **pneumonia**. The Grad-CAM heatmap shows "
                "which regions of the X-ray influenced this decision. "
                "Radiologists look for opacities, consolidation, or fluid in "
                "the lung fields — the model is likely focusing on similar "
                "features."
            )
        else:
            st.markdown(
                "The model predicts **normal**. The Grad-CAM heatmap shows "
                "relatively diffuse attention, which is consistent with a "
                "normal X-ray where no single region dominates the decision."
            )

        st.caption(
            "⚠️ **Note:** Grad-CAM is a visualization tool for research and "
            "education. It does not provide a diagnosis. Always consult a "
            "qualified healthcare professional."
        )