#!/usr/bin/env python3
"""
PSU Eco Racing — Perception Stack
scripts/train_stop_sign.py

Downloads the Roboflow stop-sign dataset and fine-tunes YOLOv8n.
Run this on your training machine (GPU workstation / Colab), not on the Jetson.

Dataset: https://universe.roboflow.com/yolo-ifyjn/stop-sign-detection-1
  class 0 — fake stop sign
  class 1 — stop sign
  class 2 — vandalized stop sign

Usage:
    pip install ultralytics roboflow
    python scripts/train_stop_sign.py --api-key YOUR_ROBOFLOW_KEY

Optional flags:
    --epochs  50        (default)
    --imgsz   416       (default; matches SIGN_IMG_SIZE in config.py)
    --batch   16        (default; lower if GPU VRAM < 8 GB)
    --device  0         (default; use 'cpu' if no GPU)
"""

import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True,
                   help="Roboflow API key (account → settings → Roboflow API)")
    p.add_argument("--version", type=int, default=1,
                   help="Dataset version number on Roboflow (check the URL)")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--imgsz",   type=int, default=416)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0",
                   help="'0' for first GPU, 'cpu' for CPU-only")
    return p.parse_args()


def download_dataset(api_key: str, version: int) -> str:
    from roboflow import Roboflow
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("yolo-ifyjn").project("stop-sign-detection-1")
    ds      = project.version(version).download("yolov8")
    print(f"[Dataset] Downloaded to: {ds.location}")
    return ds.location


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, device: str) -> str:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")          # downloads ~6 MB backbone if not cached
    model.train(
        data       = data_yaml,
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        device     = device,
        project    = "runs/stop_sign",
        name       = "train",
        patience   = 15,                # early-stop if no improvement for 15 epochs
        save       = True,
        exist_ok   = True,
        # Augmentations — keep mild; the sign appearance doesn't vary that much
        hsv_h      = 0.015,
        hsv_s      = 0.4,
        hsv_v      = 0.4,
        flipud     = 0.0,               # signs are always upright
        fliplr     = 0.0,               # octagon is symmetric but don't flip text
        mosaic     = 0.5,
    )
    return "runs/stop_sign/train/weights/best.pt"


def main():
    args = parse_args()

    print("=== Step 1/3  Download dataset ===")
    dataset_dir = download_dataset(args.api_key, args.version)
    data_yaml   = os.path.join(dataset_dir, "data.yaml")

    print("\n=== Step 2/3  Train YOLOv8n ===")
    best_pt = train(data_yaml, args.epochs, args.imgsz, args.batch, args.device)

    print("\n=== Step 3/3  Copy weights ===")
    os.makedirs("weights", exist_ok=True)
    dest = "weights/stop_sign.pt"
    shutil.copy(best_pt, dest)
    print(f"[Done]  Best weights → {dest}")
    print()
    print("Next step: export to TensorRT on the Jetson:")
    print("    python scripts/export_trt.py")
    print("Then update SIGN_MODEL_PATH in config.py to 'weights/stop_sign.engine'")


if __name__ == "__main__":
    main()
