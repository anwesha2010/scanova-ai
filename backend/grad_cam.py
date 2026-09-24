"""
Grad-CAM Implementation
========================
Computes Gradient-weighted Class Activation Maps for the pneumonia CNN classifier.
Requires the .pth model file (PyTorch format — supports gradients).

Note: This is REAL Grad-CAM, not a reconstruction-error approximation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2


# ---------- Model class definition ----------
class PneumoniaCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- Model cache ----------
_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_classifier.pth")


def load_pneumonia_model():
    """Load the PyTorch model once and cache it."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(_MODEL_PATH):
        return None

    model = PneumoniaCNN()
    try:
        state = torch.load(_MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception as e:
        print(f"[Grad-CAM] Failed to load model: {e}")
        return None


# ---------- Grad-CAM ----------
class GradCAM:
    """Grad-CAM: uses gradients to weight feature maps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=1):
        """
        Generate CAM for a specific class.

        Args:
            input_tensor: torch tensor (1, 1, 128, 128)
            class_idx: which class to explain (0=normal, 1=pneumonia)

        Returns:
            cam: numpy array (128, 128) in [0, 1]
        """
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)
        score = output[0, class_idx]

        # Backward
        score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]        # (C, H, W)
        activations = self.activations[0]    # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def generate_gradcam(image, target_class=1):
    """
    Generate a Grad-CAM heatmap for a chest X-ray image.

    Args:
        image: 2D grayscale numpy array (0-255)
        target_class: 0 = normal, 1 = pneumonia (default)

    Returns:
        dict with:
            - cam: heatmap (128, 128) in [0, 1]
            - overlay: RGB image with Grad-CAM overlay
            - success: bool
    """
    model = load_pneumonia_model()
    if model is None:
        return {
            "success": False,
            "cam": None,
            "overlay": None,
            "error": "Model not loaded. Ensure pneumonia_classifier.pth exists.",
        }

    # Preprocess
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    img_128 = cv2.resize(img, (128, 128))
    input_tensor = torch.from_numpy(img_128).unsqueeze(0).unsqueeze(0).float()

    # Target layer = last conv in features
    target_layer = model.features[-2]   # the last ReLU before AdaptiveAvgPool

    # Generate CAM
    try:
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate(input_tensor, class_idx=target_class)
    except Exception as e:
        return {
            "success": False,
            "cam": None,
            "overlay": None,
            "error": f"Grad-CAM failed: {e}",
        }

    # Resize CAM to original image size
    h_orig, w_orig = image.shape
    cam_resized = cv2.resize(cam, (w_orig, h_orig))

    # Apply colormap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Convert original to RGB
    original_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blend
    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)

    return {
        "success": True,
        "cam": cam_resized,
        "overlay": overlay,
        "heatmap": heatmap,
    }