# 🩻 SCANOVA AI

AI-powered medical image analysis assistant. Upload X-ray, MRI, or CT scans to detect potentially unusual regions and view them on a heatmap.

## ⚠️ Disclaimer
**Assistive tool only.** Does not replace professional medical diagnosis.

## Features
- Upload medical images (PNG, JPG)
- Image preprocessing (resize, denoise, contrast enhancement)
- Statistical anomaly detection
- Colored heatmap visualization
- Detection overlay with bounding boxes
- Automated report generation

## Workflow
`Upload → Preprocess → Analyze → Heatmap → Report`

## Tech Stack
- **Frontend:** Streamlit
- **Image Processing:** OpenCV, PIL
- **Numerical:** NumPy
- **Deployment:** Streamlit Cloud

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py