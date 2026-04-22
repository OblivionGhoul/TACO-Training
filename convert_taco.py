import json
import os
import shutil
import random
from collections import defaultdict

taco_data_path = r"C:\Users\Michael Collantes\OneDrive\Documents\Python\TACO\TACO\data"
annotations_file = os.path.join(taco_data_path, "annotations.json")

output_path = "taco_yolo_dataset"

train_img_dir = os.path.join(output_path, "images/train")
val_img_dir   = os.path.join(output_path, "images/val")
train_lbl_dir = os.path.join(output_path, "labels/train")
val_lbl_dir   = os.path.join(output_path, "labels/val")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir,   exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir,   exist_ok=True)

CATEGORY_MAP = {
    # Bottle (0)
    "Bottle": 0,
    "Milk bottle": 0,
    "Other plastic bottle": 0,
    "Glass bottle": 0,
    "Plastic bottle": 0,
    "Clear plastic bottle": 0,
    "Glass jar": 0,

    # Can (1)
    "Can": 1,
    "Drink can": 1,
    "Food Can": 1,
    "Aerosol": 1,
    "Scrap metal": 1,
    "Pop tab": 1,
    "Metal bottle cap": 1,
    "Metal lid": 1,

    # Cup (2)
    "Cup": 2,
    "Clear plastic cup": 2,
    "Disposable plastic cup": 2,
    "Foam cup": 2,
    "Glass cup": 2,
    "Plastic cup lid": 2,
    "Paper cup": 2,
    "Other plastic cup": 2,
    "Foam food container": 2,

    # Bag (3)
    "Plastic bag & wrapper": 3,
    "Paper bag": 3,
    "Single-use carrier bag": 3,
    "Plastic film": 3,
    "Plastic gloves": 3,
    "Plastic glooves": 3,
    "Plastified paper bag": 3,
    "Garbage bag": 3,
    "Polypropylene bag": 3,
    "Rope & strings": 3,

    # Wrapper (4)
    "Snack wrapper": 4,
    "Crisp packet": 4,
    "Aluminium foil": 4,
    "Aluminium blister pack": 4,
    "Carded blister pack": 4,
    "Tobacco pouch": 4,
    "Squeezable tube": 4,
    "Spread tub": 4,
    "Tissues & napkins": 4,
    "Tissues": 4,
    "Wrapping paper": 4,
    "Normal paper": 4,
    "Magazine paper": 4,
    "Other plastic wrapper": 4,
    "Other plastic": 4,

    # Carton (5)
    "Carton": 5,
    "Drink carton": 5,
    "Egg carton": 5,
    "Meal carton": 5,
    "Other carton": 5,
    "Corrugated carton": 5,
    "Toilet tube": 5,
    "Pizza box": 5,
    "Disposable food container": 5,
    "Plastic container": 5,
    "Tupperware": 5,
    "Other plastic container": 5,
    "Plastic bottle cap": 7,

    # Cigarette (6)
    "Cigarette": 6,

    # Other (7)
    "Battery": 7,
    "Bottle cap": 7,
    "Broken glass": 7,
    "Lid": 7,
    "Plastic lid": 7,
    "Shoe": 7,
    "Paper straw": 7,
    "Plastic straw": 7,
    "Straw": 7,
    "Styrofoam piece": 7,
    "Unlabeled litter": 7,
    "Plastic utensils": 7,
    "Six pack rings": 7,
    "Food waste": 7,
}

print("Loading TACO annotations...")
with open(annotations_file) as f:
    coco = json.load(f)

images      = coco["images"]
annotations = coco["annotations"]
categories  = coco["categories"]

cat_id_to_class = {}
for cat in categories:
    name = cat["name"]
    if name in CATEGORY_MAP:
        cat_id_to_class[cat["id"]] = CATEGORY_MAP[name]
    else:
        print(f"  ⚠️  Unmapped category '{name}' → class 7 (Other)")
        cat_id_to_class[cat["id"]] = 7

ann_map = defaultdict(list)
for ann in annotations:
    ann_map[ann["image_id"]].append(ann)

class_counts = defaultdict(int)
for ann in annotations:
    cls = cat_id_to_class[ann["category_id"]]
    class_counts[cls] += 1

print("\nAnnotation counts per class (before conversion):")
class_names = ["Bottle", "Can", "Cup", "Bag", "Wrapper", "Carton", "Cigarette", "Other"]
for cls_id, name in enumerate(class_names):
    print(f"  {cls_id}: {name:12s} → {class_counts[cls_id]:4d} annotations")

non_other_max = max(class_counts[i] for i in range(7))
other_cap     = int(non_other_max * 1.5)
print(f"\n  Max non-Other class: {non_other_max}, capping Other at: {other_cap}")
print("\nProcessing images...")

written_other        = 0
skipped_other_images = 0

for img in images:
    file_name = img["file_name"]
    img_id    = img["id"]
    img_path  = os.path.join(taco_data_path, file_name)

    if not os.path.exists(img_path):
        continue

    width  = img["width"]
    height = img["height"]

    img_anns = ann_map.get(img_id, [])
    if not img_anns:
        continue

    classes_in_image = [cat_id_to_class[ann["category_id"]] for ann in img_anns]
    all_other        = all(c == 7 for c in classes_in_image)

    if all_other and written_other >= other_cap:
        skipped_other_images += 1
        continue

    is_train = random.random() < 0.8
    img_dest = train_img_dir if is_train else val_img_dir
    lbl_dest = train_lbl_dir if is_train else val_lbl_dir

    dest_name = "taco_" + os.path.basename(file_name)          # prefix to avoid merge collisions
    shutil.copy(img_path, os.path.join(img_dest, dest_name))

    label_path = os.path.join(lbl_dest, os.path.splitext(dest_name)[0] + ".txt")
    with open(label_path, "w") as f:
        for ann in img_anns:
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm   = w / width
            h_norm   = h / height
            class_id = cat_id_to_class[ann["category_id"]]
            if class_id == 7:
                written_other += 1
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print(f"\nConversion complete. Skipped {skipped_other_images} purely-Other images (cap reached).")
print("TACO YOLO dataset created in:", output_path)