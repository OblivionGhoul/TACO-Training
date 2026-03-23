import os
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
print("Model classes:", model.names)
print("Number of classes:", len(model.names))

test_folder = "test_images"

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
    print("Press 'q' to quit the live feed.\n")

    results = model.predict(
        source=0,
        save=True,
        conf=0.25,
        show=True,
        stream=True
    )

    try:
        for r in results:
            print(f"\n📄 Frame detected")
            if len(r.boxes) == 0:
                print("  → No detections. Try lowering conf threshold.")
            else:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    conf = float(box.conf)
                    print(f"  → Detected: {label} ({conf:.1%} confidence)")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Feed closed by user.")
                break

    finally:
        cv2.destroyAllWindows()

    print("\n✅ Detection complete! Results saved to runs/detect/")
