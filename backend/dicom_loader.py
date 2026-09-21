"""
DICOM File Loader — strict magic byte check
"""

import io
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


def is_dicom(file_bytes):
    """
    Strict DICOM check.
    DICOM files have 'DICM' at byte offset 128.
    """
    if file_bytes is None or len(file_bytes) < 132:
        return False

    if file_bytes[128:132] != b'DICM':
        return False

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
    """Load DICOM bytes into a uint8 grayscale array."""
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        pixel_array = ds.pixel_array.astype(np.float32)

        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = pixel_array.max() - pixel_array

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            try:
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            except Exception:
                pass

        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            try:
                wc = ds.WindowCenter
                ww = ds.WindowWidth
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

        if pixel_array.ndim == 3:
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

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
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        info = {}
        for field in ['PatientName', 'PatientID', 'Modality',
                       'BodyPartExamined', 'StudyDescription',
                       'SeriesDescription', 'Manufacturer',
                       'StudyDate', 'InstitutionName']:
            if hasattr(ds, field):
                info[field] = str(getattr(ds, field))
        return info
    except Exception:
        return {}