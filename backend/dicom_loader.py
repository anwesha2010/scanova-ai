"""
DICOM File Loader
==================
Converts a .dcm file to a grayscale numpy array.
Handles 8-bit, 12-bit, 16-bit depth and various photometric interpretations.

Also provides a strict `is_dicom()` check so that non-DICOM files
(PNG, JPG, etc.) don't accidentally get routed here.
"""

import io
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


def is_dicom(file_bytes):
    """
    Strict DICOM magic check.

    DICOM files have 'DICM' at byte offset 128
    (128-byte preamble + 'DICM' magic).

    We also verify with pydicom that the file actually parses as DICOM,
    to avoid false positives on random binary data.
    """
    if file_bytes is None or len(file_bytes) < 132:
        return False

    # 1. Check the 4-byte magic at offset 128
    if file_bytes[128:132] != b'DICM':
        return False

    # 2. Sanity-check with pydicom — if it can't even read the header,
    #    it's not really a DICOM file.
    try:
        pydicom.dcmread(
            io.BytesIO(file_bytes),
            stop_before_pixels=True,
            force=False,
        )
        return True
    except Exception:
        return False


def load_dicom(file_bytes):
    """
    Load a DICOM file from bytes and return a normalized uint8 grayscale array.

    Args:
        file_bytes: raw bytes of a .dcm file

    Returns:
        numpy array (uint8, 0-255) — grayscale image
        None if loading fails
    """
    try:
        # Parse DICOM from bytes
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

        # Get pixel data
        pixel_array = ds.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        # MONOCHROME1 = inverted (0 = white, max = black)
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = pixel_array.max() - pixel_array

        # Handle Rescale Slope/Intercept (Hounsfield units for CT)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            try:
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            except Exception:
                pass

        # Handle Window Center / Width for better contrast
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            try:
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                # These can be MultiValue or single values
                if isinstance(wc, (list, tuple)):
                    wc = wc[0]
                if isinstance(ww, (list, tuple)):
                    ww = ww[0]
                wc = float(wc)
                ww = float(ww)

                low = wc - ww / 2
                high = wc + ww / 2
                pixel_array = np.clip(pixel_array, low, high)
            except Exception:
                pass

        # Handle multi-frame (take middle frame)
        if pixel_array.ndim == 3:
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

        # Normalize to 0-255
        pmin = float(pixel_array.min())
        pmax = float(pixel_array.max())

        if pmax - pmin < 1e-6:
            return np.zeros(pixel_array.shape, dtype=np.uint8)

        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        return pixel_array.astype(np.uint8)

    except InvalidDicomError:
        return None
    except Exception as e:
        print(f"[DICOM] Failed to load: {e}")
        return None


def get_dicom_info(file_bytes):
    """
    Extract metadata from a DICOM file (for display).
    Returns a dictionary of metadata fields.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        info = {}

        for field in [
            'PatientName', 'PatientID', 'Modality',
            'BodyPartExamined', 'StudyDescription',
            'SeriesDescription', 'Manufacturer',
            'StudyDate', 'InstitutionName',
        ]:
            if hasattr(ds, field):
                val = getattr(ds, field)
                info[field] = str(val)

        return info
    except Exception:
        return {}