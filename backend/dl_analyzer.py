"""
Multi-Modal Deep Learning Analyzer
====================================
Loads 4 specialized ONNX autoencoders + 1 modality classifier.
Auto-routes input images to the correct model.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort


# ---------- Model paths ----------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATHS = {
     "chest_xray": os.path.join(BASE_DIR, "chest_xray.onnx"),
    "brain_mri":  os.path.join(BASE_DIR, "brain_mri.onnx"),
    "chest_ct":   os.path.join(BASE_DIR, "chest_ct.onnx"),
    "bone_scan":  os.path.join(BASE_DIR, "bone_scan.onnx"),
    "classifier": os.path.join(BASE_DIR, "modality_classifier.onnx"),
    "pneumonia":  os.path.join(BASE_DIR, "pneumonia_classifier.onnx"),
}

CLASS_NAMES = ["chest_xray", "brain_mri", "chest_ct", "bone_scan"]

# ---------- Model cache ----------
_SESSIONS = {}


def _load_session(name):
    """Load (and cache) an ONNX session."""
    if name in _SESSIONS:
        return _SESSIONS[name]
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        return None
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    _SESSIONS[name] = sess
    return sess


# ---------- Public API ----------
def predict_modality(image):
    """
    Predict which modality an image belongs to.
    Returns: (modality_name, confidence_dict)
    """
    sess = _load_session("classifier")
    if sess is None:
        return "chest_xray", {}

    img_128 = _to_128(image)
    inp = img_128.reshape(1, 1, 128, 128).astype(np.float32)
    out = sess.run(None, {"input": inp})[0][0]

    # softmax
    exp = np.exp(out - np.max(out))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    modality = CLASS_NAMES[idx]
    confidence = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
    return modality, confidence


def detect_anomalies_dl(image, modality=None, block_size=32, **kwargs):
    """
    Multi-modal deep learning anomaly detection.

    Args:
        image: 2D grayscale numpy array (0-1 float OR 0-255 uint8)
        modality: optional forced modality
                  ("chest_xray" | "brain_mri" | "chest_ct" | "bone_scan")

    Returns:
        anomaly_map: 2D numpy array in [0, 1]
    """
    # Auto-detect modality
    if modality is None or modality == "auto":
        modality, _ = predict_modality(image)

    # Load the correct autoencoder
    sess = _load_session(modality)
    if sess is None:
        return np.zeros_like(image, dtype=np.float32)

    # Prepare input
    img_128 = _to_128(image)
    inp = img_128.reshape(1, 1, 128, 128).astype(np.float32)

    # Run inference
    reconstructed = sess.run(None, {"input": inp})[0][0, 0]

    # Reconstruction error
    error = np.abs(img_128 - reconstructed)

    # Smooth
    error = cv2.GaussianBlur(error.astype(np.float32), (7, 7), 0)

    # Resize to original
    h_orig, w_orig = image.shape
    error = cv2.resize(error, (w_orig, h_orig))

    # Normalize to [0, 1]
    error = (error - error.min()) / (error.max() - error.min() + 1e-8)

    # Amplify contrast
    error = np.power(error, 1.5)

    return error.astype(np.float32)


# ---------- Helpers ----------
def _to_128(image):
    """Normalize + resize to 128x128 float in [0, 1]."""
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
    return cv2.resize(img, (128, 128))


# ---------- Reuse region detection ----------
from backend.analyzer import find_anomaly_regions  # noqa: E402
def predict_pneumonia(image):
    """
    Predict pneumonia using the trained CNN classifier.

    Args:
        image: 2D grayscale numpy array (0-255)

    Returns:
        dict with:
            - prediction: "normal" or "pneumonia"
            - confidence: float (0-1)
            - probabilities: {"normal": float, "pneumonia": float}
    """
    sess = _load_session("pneumonia")
    if sess is None:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {"normal": 0.0, "pneumonia": 0.0},
            "error": "Pneumonia classifier model not found.",
        }

    # Prepare input: resize to 128x128, normalize to [0, 1]
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    img_128 = cv2.resize(img, (128, 128))
    inp = img_128.reshape(1, 1, 128, 128).astype(np.float32)

    # Run inference
    output = sess.run(None, {"input": inp})[0][0]  # shape (2,)

    # Softmax (if model outputs raw logits)
    exp = np.exp(output - np.max(output))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    prediction = "pneumonia" if idx == 1 else "normal"
    confidence = float(probs[idx])

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "normal": float(probs[0]),
            "pneumonia": float(probs[1]),
        },
    }