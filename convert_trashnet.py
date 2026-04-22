import os
import shutil
import random
from PIL import Image

# ── UPDATE THIS PATH to your unzipped dataset-resized folder ──────────────────
TRASHNET_PATH = r"C:\Users\Michael Collantes\OneDrive\Documents\YOLO_TACO\dataset-resized"
# ──────────────────────────────────────────────────────────────────────────────

output_path = "trashnet_yolo_dataset"

train_img_dir = os.path.join(output_path, "images/train")
val_img_dir   = os.path.join(output_path, "images/val")
train_lbl_dir = os.path.join(output_path, "labels/train")
val_lbl_dir   = os.path.join(output_path, "labels/val")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir,   exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir,   exist_ok=True)

# TrashNet folder name → your 8-class YOLO id
# 0=Bottle, 1=Can, 2=Cup, 3=Bag, 4=Wrapper, 5=Carton, 6=Cigarette, 7=Other
FOLDER_TO_CLASS = {
    "glass":     0,   # → Bottle  (glass bottles/jars)
    "metal":     1,   # → Can     (metal cans, tins)
    "paper":     4,   # → Wrapper (paper scraps, newspaper)
    "cardboard": 5,   # → Carton  (boxes, cartons)
    "plastic":   3,   # → Bag     (plastic items / bags)
    "trash":     7,   # → Other   (miscellaneous)
}

valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

print("Converting TrashNet dataset to YOLO format...\n")

counts = {cls: 0 for cls in range(8)}
skipped = 0

for folder_name, class_id in FOLDER_TO_CLASS.items():
    folder_path = os.path.join(TRASHNET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        print(f"  ⚠️  Folder not found, skipping: {folder_path}")
        continue

    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1] in valid_extensions
    ]

    print(f"  {folder_name:12s} (class {class_id}) → {len(image_files)} images")

    for fname in image_files:
        src_path = os.path.join(folder_path, fname)

        # Get image dimensions
        try:
            with Image.open(src_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"    ⚠️  Could not read {fname}: {e}")
            skipped += 1
            continue

        is_train  = random.random() < 0.8
        img_dest  = train_img_dir if is_train else val_img_dir
        lbl_dest  = train_lbl_dir if is_train else val_lbl_dir

        # Prefix filename to avoid collisions when merging with TACO
        dest_name = f"trashnet_{folder_name}_{fname}"
        shutil.copy(src_path, os.path.join(img_dest, dest_name))

        # TrashNet has no bounding boxes — treat the whole image as one object
        # YOLO label: class cx cy w h  (all normalised, full image = 0.5 0.5 1.0 1.0)
        label_path = os.path.join(lbl_dest, os.path.splitext(dest_name)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n")

        counts[class_id] += 1

print("\nTrashNet conversion complete!")
print("\nImages written per class:")
class_names = ["Bottle", "Can", "Cup", "Bag", "Wrapper", "Carton", "Cigarette", "Other"]
for cls_id, name in enumerate(class_names):
    if counts[cls_id]:
        print(f"  {cls_id}: {name:12s} → {counts[cls_id]:4d} images")
if skipped:
    print(f"\n  ⚠️  Skipped {skipped} unreadable images.")
print("\nTrashNet YOLO dataset created in:", output_path)
