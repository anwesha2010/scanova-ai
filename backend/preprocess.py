import cv2
import numpy as np
from PIL import Image
import io

from backend.dicom_loader import is_dicom, load_dicom

TARGET_SIZE = (512, 512)


def load_image(file_bytes):
    """
    Load an image from bytes.
    Handles DICOM, PNG, JPG, JPEG.
    Returns grayscale uint8 numpy array.
    """

    # ---- Try DICOM first (only if it really looks like DICOM) ----
    try:
        if is_dicom(file_bytes):
            dicom_img = load_dicom(file_bytes)
            if dicom_img is not None and dicom_img.size > 0:
                return dicom_img
    except Exception:
        pass  # Not DICOM, continue

    # ---- Try PIL (PNG/JPG) ----
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        arr = np.array(pil_img, dtype=np.uint8)
        if arr.size > 0:
            return arr
    except Exception:
        pass  # Not PIL-readable

    # ---- Try OpenCV (last resort) ----
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    # ---- All failed ----
    raise ValueError("Could not load image. Unsupported format or corrupted file.")


def resize_image(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def denoise(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def normalize(img):
    return img.astype(np.float32) / 255.0


def preprocess(file_bytes):
    """Full pipeline: bytes → (original, enhanced, normalized)."""
    original = resize_image(load_image(file_bytes))
    denoised = denoise(original)
    enhanced = enhance_contrast(denoised)
    normalized = normalize(enhanced)
    return original, enhanced, normalized