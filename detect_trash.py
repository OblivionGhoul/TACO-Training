import os
import cv2
from ultralytics import YOLO

# Load trained model (updated to train2)
model = YOLO("runs/detect/train2/weights/best.pt")
print("Model classes:", model.names)
print("Number of classes:", len(model.names))
test_folder = "test_images"

# Filter out unreadable images before passing to YOLO
valid_images = []
for fname in os.listdir(test_folder):
    fpath = os.path.join(test_folder, fname)
    img = cv2.imread(fpath)
    if img is None:
        print(f"⚠️  Skipping unreadable image: {fname}")
    else:
        valid_images.append(fpath)

if not valid_images:
    print("No valid images found in test_images folder.")
else:
    print(f"Running detection on {len(valid_images)} valid image(s)...\n")
    results = model.predict(
        source=valid_images,
        save=True,
        conf=0.25,
        verbose=True
    )

    for r in results:
        print(f"\n📄 {os.path.basename(r.path)}")
        if len(r.boxes) == 0:
            print("  → No detections. Try lowering conf threshold.")
        else:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                print(f"  → Detected: {label} ({conf:.1%} confidence)")

    print("\n✅ Detection complete! Results saved to runs/detect/")