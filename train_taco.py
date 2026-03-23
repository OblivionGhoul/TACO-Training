from ultralytics import YOLO          

model = YOLO("yolov8s.pt")

model.train(
    data="taco.yaml",                 # dataset config (classes, train/val paths)
    epochs=100,                       # max training iterations
    imgsz=768,                        # input image resolution
    batch=8,                          # images processed per step
    device="cpu",                     # run on CPU
    patience=30,                      # stop early if no improvement for 30 epochs

    # Loss weights
    cls=2.5,                          # classification loss weight (higher = more class-aware)
    box=7.5,                          # bounding box regression loss weight

    # Optimizer
    lr0=0.005,                        # initial learning rate
    lrf=0.01,                         # final learning rate (as fraction of lr0)
    weight_decay=0.0005,              # L2 regularization to reduce overfitting
    warmup_epochs=5,                  # gradually ramp up lr for first 5 epochs

    # Augmentation
    augment=True,                     # enable built-in augmentations
    hsv_h=0.015,                      # random hue shift
    hsv_s=0.7,                        # random saturation shift
    hsv_v=0.4,                        # random brightness shift
    flipud=0.5,                       # 50% chance vertical flip
    fliplr=0.5,                       # 50% chance horizontal flip
    mosaic=1.0,                       # always use mosaic (4-image collage) augmentation
    copy_paste=0.4,                   # 40% chance to paste objects from other images
    mixup=0.2,                        # 20% chance to blend two images together

    # Save settings
    save_period=10,                   # checkpoint every 10 epochs
    verbose=True,                     # print training logs
)

print("Training complete!")
