from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="taco.yaml",
    epochs=100,
    imgsz=768,
    batch=8,
    device="cpu",
    patience=30,            # early stopping: saves CPU time

    # Loss weights - MOST IMPORTANT FIX
    cls=2.5,                # default is 0.5; raising this forces the model to care about class identity
    box=7.5,                # default is 7.5; keeping explicit

    # Optimizer
    lr0=0.005,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,

    # Augmentation
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    copy_paste=0.4,         # helps rare classes by pasting them into other images
    mixup=0.2,              # mild mixup for generalization

    # Save settings
    save_period=10,         # save checkpoint every 10 epochs
    verbose=True,
)

print("Training complete!")