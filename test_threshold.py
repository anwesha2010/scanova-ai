from backend.dl_analyzer import predict_pneumonia
import cv2
import os

print("=" * 60)
print("Testing predict_pneumonia() outside Streamlit")
print("=" * 60)

normal_dir = "test_data/normal"
pneumonia_dir = "test_data/pneumonia"

print("\n📁 Testing NORMAL images:")
for fname in sorted(os.listdir(normal_dir))[:5]:
    img = cv2.imread(os.path.join(normal_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    result = predict_pneumonia(img)
    print(f"  {fname[:20]:20s} → {result['prediction']:10s} ({result['confidence']:.3f})")

print("\n📁 Testing PNEUMONIA images:")
for fname in sorted(os.listdir(pneumonia_dir))[:5]:
    img = cv2.imread(os.path.join(pneumonia_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    result = predict_pneumonia(img)
    print(f"  {fname[:20]:20s} → {result['prediction']:10s} ({result['confidence']:.3f})")

print("\n" + "=" * 60)