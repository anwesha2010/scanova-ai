from backend.grad_cam import generate_gradcam
import cv2
import os
import numpy as np

print("=" * 60)
print("Testing Grad-CAM on pneumonia classifier")
print("=" * 60)

# Test on a pneumonia image (should show strong activation)
pneumonia_dir = "test_data/pneumonia"
normal_dir = "test_data/normal"

pneumonia_files = sorted(os.listdir(pneumonia_dir))[:2]
normal_files = sorted(os.listdir(normal_dir))[:2]

print("\n📁 PNEUMONIA images (expect strong Grad-CAM):")
for fname in pneumonia_files:
    img = cv2.imread(os.path.join(pneumonia_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    result = generate_gradcam(img, target_class=1)
    if result["success"]:
        cam = result["cam"]
        print(f"  {fname[:25]:25s} → CAM range: {cam.min():.3f} to {cam.max():.3f}, mean: {cam.mean():.3f}")
    else:
        print(f"  {fname[:25]:25s} → FAILED: {result.get('error', 'unknown')}")

print("\n📁 NORMAL images (expect weaker Grad-CAM):")
for fname in normal_files:
    img = cv2.imread(os.path.join(normal_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    result = generate_gradcam(img, target_class=1)
    if result["success"]:
        cam = result["cam"]
        print(f"  {fname[:25]:25s} → CAM range: {cam.min():.3f} to {cam.max():.3f}, mean: {cam.mean():.3f}")
    else:
        print(f"  {fname[:25]:25s} → FAILED: {result.get('error', 'unknown')}")

print("\n" + "=" * 60)

# Save one example overlay for visual check
print("\nSaving example overlay...")
img = cv2.imread(os.path.join(pneumonia_dir, pneumonia_files[0]), cv2.IMREAD_GRAYSCALE)
result = generate_gradcam(img, target_class=1)
if result["success"]:
    cv2.imwrite("test_gradcam_output.png", result["overlay"])
    print("✅ Saved test_gradcam_output.png")
else:
    print(f"❌ Failed: {result.get('error')}")