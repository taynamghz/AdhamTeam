#!/usr/bin/env python3
"""
PSU Eco Racing — Perception Stack
scripts/export_trt.py

Exports weights/stop_sign.pt → weights/stop_sign.engine (TensorRT FP16).
Run this ON THE JETSON NANO itself — TRT engines are device-specific.

Requires: ultralytics, tensorrt (installed with JetPack)

Usage:
    python scripts/export_trt.py
    python scripts/export_trt.py --weights weights/stop_sign.pt --imgsz 416 --half

After export:
    Edit config.py: SIGN_MODEL_PATH = "weights/stop_sign.engine"

FP16 vs INT8:
  - FP16 (default): ~15-25 FPS on Nano, no calibration data needed.
  - INT8: ~20-30 FPS but needs a calibration dataset — skip for now.
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/stop_sign.pt",
                   help="Path to trained .pt weights")
    p.add_argument("--imgsz",   type=int, default=416,
                   help="Must match SIGN_IMG_SIZE in config.py")
    p.add_argument("--half",    action="store_true", default=True,
                   help="FP16 export (recommended for Jetson Nano)")
    p.add_argument("--workspace", type=int, default=4,
                   help="TensorRT builder workspace in GB (Nano has 4 GB shared)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Run  python scripts/train_stop_sign.py --api-key YOUR_KEY  first."
        )

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"[Export] {args.weights} → TensorRT FP16  imgsz={args.imgsz}")
    model.export(
        format    = "engine",
        imgsz     = args.imgsz,
        half      = args.half,
        device    = 0,
        workspace = args.workspace,
        verbose   = True,
    )

    engine_path = args.weights.replace(".pt", ".engine")
    print(f"\n[Done]  TensorRT engine → {engine_path}")
    print()
    print("Update config.py:")
    print(f'    SIGN_MODEL_PATH = "{engine_path}"')


if __name__ == "__main__":
    main()
