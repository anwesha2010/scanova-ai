# 🩻 SCANOVA AI

**See. Analyze. Explain.**

An AI-powered medical imaging assistant that analyzes X-rays, MRIs, CT scans, and bone scans — flags potentially unusual regions, and shows **exactly where the AI looked**.

> ⚠️ **Not a medical device.** This is an educational / research prototype. It does not replace a radiologist or physician. All results must be reviewed by a qualified healthcare professional.

🌐 **Live demo:** [scanova-ai.streamlit.app](https://scanova-ai.streamlit.app)

---

## 🎯 What It Does

1. **Upload** an X-ray, MRI, CT, or bone scan (PNG / JPG / DICOM)
2. **Pre-scan** checks image quality (resolution, contrast, sharpness)
3. **AI analysis** — routes to the correct model based on modality
4. **Explainable AI** — Grad-CAM heatmap shows where the model looked
5. **Patient-friendly view** — plain-language explanation
6. **Report** — downloadable PNG + PDF with medical disclaimer

---

## 🧠 Architecture

**Pipeline:** Preprocess → Modality Classifier → Anomaly Detection → Grad-CAM → Report

- **Modality Classifier** (ONNX, 99% accuracy) routes the image to one of 4 specialized autoencoders (chest X-ray, brain MRI, chest CT, bone scan).
- **Anomaly Detection** uses 4 ONNX autoencoders — one per modality — flagging regions with high reconstruction error.
- **Pneumonia Classifier** (CNN, 74% accuracy) runs alongside for disease-specific scoring on chest X-rays.
- **Grad-CAM Explainer** visualizes which regions drove the decision (local only — PyTorch too large for Streamlit Cloud free tier).
- **Output:** heatmap overlay + downloadable PDF report with patient-friendly and clinical views.

---

## 📊 Measured Results

All results computed **live in the app** on a 50-image test set.

| Metric | Value | Note |
|--------|-------|------|
| **Accuracy** | **74.0%** | 37 of 50 correct |
| **Precision** | 65.8% | 2 of 3 flags are real |
| **Recall** | **100.0%** | Never missed a pneumonia case |
| **F1 Score** | 79.4% | Balanced precision/recall |

**Confusion matrix:**

|  | Pred Normal | Pred Pneumonia |
|---|---|---|
| **Actual Normal** | 12 | 13 |
| **Actual Pneumonia** | 0 | 25 |

**Modality classifier:** 99% accuracy on held-out validation set (300 images).

**Design choice:** We tuned the model for **100% recall** — because in medical screening, missing a sick patient is far worse than a false alarm. This means precision is lower by design.

---

## 🔥 Grad-CAM Explainability

SCANOVA uses **real Grad-CAM** (Gradient-weighted Class Activation Mapping) — not a fake heatmap.

- **Red** = high model attention
- **Blue** = low model attention
- Based on gradients flowing through the last convolutional layer

**Availability:**
- ✅ **Local deployment** (with PyTorch installed)
- ❌ **Streamlit Cloud free tier** (PyTorch exceeds size limit)

The Cloud app shows an info box explaining this when Grad-CAM is unavailable.

---

## 🏆 InnoEx Features

- **Judge Mode** — guided 5-minute demo walkthrough
- **AI vs Human Challenge** — play 5 rounds against the AI (both scored honestly, not rigged)
- **Live Model Performance** — real metrics recomputed on demand
- **Compare Mode** — statistical engine vs deep learning side-by-side

---

## 🚀 Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit |
| Image processing | OpenCV, Pillow |
| Deep learning | PyTorch, ONNX Runtime |
| Explainability | Grad-CAM (custom implementation) |
| DICOM | pydicom |
| PDF reports | ReportLab |
| Deployment | Streamlit Cloud |
| Version control | GitHub |

---

## 📁 Repository Structure

- `app.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT
- `backend/` — Preprocessing, analysis, Grad-CAM, PDF reports, DICOM loader, ONNX models
- `pages/` — About, Analytics, FAQ, Model Performance
- `branding/` — Design system (CSS, hero images, favicon)
- `utils/` — Analytics helpers
- `test_data/` — 50 labeled images (normal + pneumonia)
- `assets/` — Sample images

---

## ⚠️ Limitations

We're honest about what this tool does **not** do:

- **Not clinically validated.** Tested on 50 images, not thousands.
- **Pediatric data only.** Chest X-ray training set is from ages 1–5.
- **Single-disease classifier.** Only pneumonia has a diagnostic model.
- **74% accuracy is not enough for clinical use.** Real medical AI needs 95%+.
- **Grad-CAM runs locally only.** Cloud free tier can't host PyTorch.
- **No patient data storage.** Images are processed in memory only.
- **No regulatory approval.** Not FDA-cleared, not CE-marked.

---

## 🔮 Roadmap

- [ ] Retrain on larger chest X-ray datasets (5,000+ images)
- [ ] Fine-tune ResNet-50 for higher accuracy
- [ ] Add more disease classifiers (nodules, fractures)
- [ ] Clinical validation with a medical partner
- [ ] On-device mobile deployment (TFLite)
- [ ] Multilingual patient reports

---

## 🛡️ Safety

- Medical disclaimer on every page
- Safety-first threshold (100% recall prioritized)
- No patient data stored
- Open-source (MIT license)
- Human-in-the-loop by design

---

## 🎓 About This Project

Built by **Anwesha Rout** and the **SCANOVA Team** for **InnoEx 2026**.

**Mission:** Demonstrate how AI can assist — not replace — medical professionals.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)