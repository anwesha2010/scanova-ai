import cv2
import numpy as np
from PIL import Image
import io

TARGET_SIZE = (512, 512)

def preprocess(file_bytes):
    """
    Takes uploaded image bytes.
    Returns: (original, enhanced, normalized)
    """
    # 1. Read the uploaded bytes into an image
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = np.array(pil_img, dtype=np.uint8)

    # 2. Resize to 512x512
    img = cv2.resize(img, TARGET_SIZE)

    # 3. Remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 5. Normalize to 0-1
    normalized = enhanced.astype(np.float32) / 255.0

    return img, enhanced, normalized