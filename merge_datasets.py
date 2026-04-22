import os
import shutil
from collections import defaultdict

# ── Inputs & output ───────────────────────────────────────────────────────────
TACO_PATH     = "taco_yolo_dataset"
TRASHNET_PATH = "trashnet_yolo_dataset"
OUTPUT_PATH   = "combined_yolo_dataset"
# ──────────────────────────────────────────────────────────────────────────────

SPLITS = ["train", "val"]

for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", split), exist_ok=True)

class_names = ["Bottle", "Can", "Cup", "Bag", "Wrapper", "Carton", "Cigarette", "Other"]
img_counts  = defaultdict(int)
lbl_counts  = defaultdict(int)
collisions  = 0


def copy_split(src_dataset, split):
    """Copy all images + labels from src_dataset/split into combined output."""
    global collisions

    img_src = os.path.join(src_dataset, "images", split)
    lbl_src = os.path.join(src_dataset, "labels", split)
    img_dst = os.path.join(OUTPUT_PATH, "images", split)
    lbl_dst = os.path.join(OUTPUT_PATH, "labels", split)

    if not os.path.isdir(img_src):
        print(f"  ⚠️  Missing: {img_src} — skipping.")
        return

    for fname in os.listdir(img_src):
        if not os.path.isfile(os.path.join(img_src, fname)):
            continue

        dest_img = os.path.join(img_dst, fname)
        if os.path.exists(dest_img):
            collisions += 1
            print(f"  ⚠️  Collision (already exists): {fname}")
            continue

        shutil.copy(os.path.join(img_src, fname), dest_img)
        img_counts[split] += 1

        # Copy matching label file
        stem      = os.path.splitext(fname)[0]
        lbl_fname = stem + ".txt"
        lbl_src_f = os.path.join(lbl_src, lbl_fname)
        if os.path.exists(lbl_src_f):
            shutil.copy(lbl_src_f, os.path.join(lbl_dst, lbl_fname))
            lbl_counts[split] += 1
        else:
            print(f"  ⚠️  No label found for: {fname}")


print("=" * 55)
print("Merging TACO + TrashNet → combined_yolo_dataset")
print("=" * 55)

for split in SPLITS:
    print(f"\n[{split.upper()}] Copying TACO...")
    copy_split(TACO_PATH, split)
    print(f"[{split.upper()}] Copying TrashNet...")
    copy_split(TRASHNET_PATH, split)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Merge complete!")
print("=" * 55)
for split in SPLITS:
    print(f"  {split:5s}: {img_counts[split]:5d} images | {lbl_counts[split]:5d} labels")
if collisions:
    print(f"\n  ⚠️  {collisions} filename collision(s) — check prefixes in convert scripts.")
else:
    print("\n  ✅  No filename collisions.")

# ── Class distribution in combined dataset ────────────────────────────────────
print("\nClass distribution in combined dataset:")
combined_counts = defaultdict(int)
for split in SPLITS:
    lbl_dir = os.path.join(OUTPUT_PATH, "labels", split)
    if not os.path.isdir(lbl_dir):
        continue
    for fname in os.listdir(lbl_dir):
        fpath = os.path.join(lbl_dir, fname)
        if not fpath.endswith(".txt"):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    combined_counts[int(parts[0])] += 1

for cls_id, name in enumerate(class_names):
    print(f"  {cls_id}: {name:12s} → {combined_counts[cls_id]:5d} annotations")

print(f"\nOutput dataset ready in: {OUTPUT_PATH}")
print("Next step: update taco.yaml  →  path: combined_yolo_dataset")
print("Then run train_taco.py as normal.")
