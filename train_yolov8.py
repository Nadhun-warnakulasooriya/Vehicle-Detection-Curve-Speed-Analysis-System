"""
YOLOv8 Vehicle Detection Training Script
Dataset: 9991 images — 9 classes
Hardware: GTX 3050 Laptop GPU (4GB VRAM) + 16GB RAM
"""

import os
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────
#  ENVIRONMENT SETUP
# ─────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def check_gpu():
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU   : {gpu}")
        print(f"   VRAM  : {vram:.1f} GB")
        print(f"   CUDA  : {torch.version.cuda}")
    else:
        print("⚠️  No GPU found — training on CPU (very slow!)")
    print()


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "data_yaml"   : "C:/Final project/data.yaml",  # ← update if different
    "project"     : "C:/Final project/runs",
    "name"        : "vehicle_detection_v1",

    # ── Model ──────────────────────────────────────────────────────────────
    # With 9991 images you can comfortably use 's' or 'm'.
    # yolov8s → safe for 4GB VRAM at batch=8  (mAP50 ~0.78–0.85)
    # yolov8m → better accuracy, needs batch=4 (mAP50 ~0.82–0.89)
    "model"       : "yolov8s.pt",     # swap to "yolov8m.pt" + batch=4 for max accuracy

    # ── Core Training ──────────────────────────────────────────────────────
    "epochs"      : 100,   # 9991 imgs → 100 epochs is plenty; early stop handles the rest
    "patience"    : 20,    # stop if no mAP gain for 20 epochs
    "batch"       : 8,     # 4GB VRAM sweet spot for yolov8s; use 4 for yolov8m
    "imgsz"       : 640,

    # ── Optimizer ──────────────────────────────────────────────────────────
    "optimizer"   : "AdamW",
    "lr0"         : 0.001,
    "lrf"         : 0.01,
    "momentum"    : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr"      : True,

    # ── Augmentation ───────────────────────────────────────────────────────
    # Large dataset (10K) → moderate augmentation is enough
    "hsv_h"       : 0.015,
    "hsv_s"       : 0.7,
    "hsv_v"       : 0.4,
    "degrees"     : 3.0,    # small rotation — vehicles rarely tilt much
    "translate"   : 0.1,
    "scale"       : 0.5,
    "shear"       : 1.0,
    "flipud"      : 0.0,    # no vertical flip
    "fliplr"      : 0.5,
    "mosaic"      : 1.0,    # mosaic ON for majority of training
    "close_mosaic": 15,     # turn mosaic OFF for last 15 epochs → cleaner fine-tuning
    "mixup"       : 0.05,   # lighter mixup for large dataset
    "copy_paste"  : 0.05,

    # ── Memory / Speed ─────────────────────────────────────────────────────
    # 9991 images × ~1.5MB ≈ 15GB uncompressed → DO NOT use cache="ram"
    # "disk" caches decoded images to disk (fast reads, no RAM pressure)
    # False = re-decode every epoch (slowest but zero extra memory)
    "cache"       : "disk",   # ← best choice for 10K images on 16GB RAM
    "workers"     : 6,        # more workers = faster data loading for large dataset
    "amp"         : True,     # FP16 mixed precision — saves ~50% VRAM
    "rect"        : False,    # keep False for best mAP
    "multi_scale" : False,    # True gives +1–2% mAP but needs more VRAM

    # ── Logging / Saving ───────────────────────────────────────────────────
    "save_period" : 10,
    "plots"       : True,
    "val"         : True,
    "verbose"     : True,
}


# ─────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────
def train():
    check_gpu()

    print("=" * 58)
    print("  YOLOv8 Vehicle Detection — Training")
    print("=" * 58)
    print(f"  Model       : {CONFIG['model']}")
    print(f"  Dataset     : 9,991 images  |  9 classes")
    print(f"  Epochs      : {CONFIG['epochs']}  (patience={CONFIG['patience']})")
    print(f"  Batch       : {CONFIG['batch']}")
    print(f"  Image size  : {CONFIG['imgsz']}")
    print(f"  AMP (FP16)  : {CONFIG['amp']}")
    print(f"  Cache       : {CONFIG['cache']}")
    print(f"  Close mosaic: last {CONFIG['close_mosaic']} epochs")
    print("=" * 58)
    print()

    model = YOLO(CONFIG["model"])

    results = model.train(
        data          = CONFIG["data_yaml"],
        epochs        = CONFIG["epochs"],
        patience      = CONFIG["patience"],
        batch         = CONFIG["batch"],
        imgsz         = CONFIG["imgsz"],

        optimizer     = CONFIG["optimizer"],
        lr0           = CONFIG["lr0"],
        lrf           = CONFIG["lrf"],
        momentum      = CONFIG["momentum"],
        weight_decay  = CONFIG["weight_decay"],
        warmup_epochs = CONFIG["warmup_epochs"],
        cos_lr        = CONFIG["cos_lr"],

        hsv_h         = CONFIG["hsv_h"],
        hsv_s         = CONFIG["hsv_s"],
        hsv_v         = CONFIG["hsv_v"],
        degrees       = CONFIG["degrees"],
        translate     = CONFIG["translate"],
        scale         = CONFIG["scale"],
        shear         = CONFIG["shear"],
        flipud        = CONFIG["flipud"],
        fliplr        = CONFIG["fliplr"],
        mosaic        = CONFIG["mosaic"],
        close_mosaic  = CONFIG["close_mosaic"],
        mixup         = CONFIG["mixup"],
        copy_paste    = CONFIG["copy_paste"],

        cache         = CONFIG["cache"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        rect          = CONFIG["rect"],
        multi_scale   = CONFIG["multi_scale"],

        project       = CONFIG["project"],
        name          = CONFIG["name"],
        save_period   = CONFIG["save_period"],
        plots         = CONFIG["plots"],
        val           = CONFIG["val"],
        verbose       = CONFIG["verbose"],

        device        = 0,
    )

    print("\n✅ Training complete!")
    best = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    print(f"   Best weights : {best}")
    return results


# ─────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────
def validate(weights_path: str = None):
    if weights_path is None:
        weights_path = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"

    print(f"\n📊 Validating: {weights_path}")
    model   = YOLO(weights_path)
    metrics = model.val(
        data    = CONFIG["data_yaml"],
        imgsz   = CONFIG["imgsz"],
        batch   = CONFIG["batch"],
        device  = 0,
        plots   = True,
        project = CONFIG["project"],
        name    = CONFIG["name"] + "_val",
    )

    print(f"\n📈 Results:")
    print(f"   mAP50     : {metrics.box.map50:.4f}")
    print(f"   mAP50-95  : {metrics.box.map:.4f}")
    print(f"   Precision : {metrics.box.mp:.4f}")
    print(f"   Recall    : {metrics.box.mr:.4f}")
    return metrics


# ─────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────
def predict(image_path: str, weights_path: str = None, conf: float = 0.25):
    if weights_path is None:
        weights_path = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"

    model = YOLO(weights_path)
    return model.predict(
        source  = image_path,
        imgsz   = CONFIG["imgsz"],
        conf    = conf,
        iou     = 0.45,
        device  = 0,
        save    = True,
        project = CONFIG["project"],
        name    = "predictions",
    )


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    train()
    validate()
    # predict("C:/Final project/test/images/your_image.jpg")
