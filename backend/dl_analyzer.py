"""
Deep Learning Analyzer using trained PyTorch Autoencoder.
Detects anomalies via reconstruction error.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn


# ---------- Model architecture ----------
class XRayAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------- Load model once (module-level cache) ----------
_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "scanova_autoencoder.pth")


def load_model():
    """Load the trained autoencoder. Caches after first call."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = XRayAutoencoder()
    if os.path.exists(_MODEL_PATH):
        state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        return _MODEL
    return None


# ---------- Public API (mirrors analyzer.py) ----------
def detect_anomalies_dl(image, block_size=32, **kwargs):
    """
    DL-based anomaly detection via reconstruction error.
    
    Args:
        image: 2D grayscale numpy array (0-1 float OR 0-255 uint8)
        block_size: unused (kept for API compatibility)
    
    Returns:
        anomaly_map: 2D numpy array in [0, 1]
    """
    model = load_model()
    if model is None:
        # Fallback: return uniform map if model missing
        return np.zeros_like(image, dtype=np.float32)

    # Ensure 2D float in [0, 1]
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    # Resize to model input size
    target = 128
    h_orig, w_orig = img.shape
    img_128 = cv2.resize(img, (target, target))

    # Run through model
    tensor = torch.from_numpy(img_128).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(tensor)

    # Reconstruction error
    error = torch.abs(tensor - reconstructed).squeeze().numpy()

    # Smooth the error map
    error = cv2.GaussianBlur(error, (11, 11), 0)

    # Resize error map back to original size
    error = cv2.resize(error, (w_orig, h_orig))

    # Normalize to [0, 1]
    error = (error - error.min()) / (error.max() - error.min() + 1e-8)
    return error.astype(np.float32)


# ---------- Reuse region detection from statistical analyzer ----------
from backend.analyzer import find_anomaly_regions  # noqa: E402