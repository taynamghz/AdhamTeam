"""
PSU Eco Racing — Perception Stack
scripts/export_trt.py  |  Export YOLO stop-sign model to TensorRT FP16.

Run ONCE on the Jetson Orin Nano after training:
    python scripts/export_trt.py

This produces  perception_stack/weights/stop_sign.engine
Then set in config.py:
    SIGN_MODEL_PATH = "perception_stack/weights/stop_sign.engine"

Expected speedup on Orin Nano 8GB:
    .pt   PyTorch     ~50-80 ms  per inference
    .engine TRT FP16  ~5-10 ms   per inference  (~8x faster)
"""

import os
import sys
import time

WEIGHTS_PT     = "perception_stack/weights/stop_sign.pt"
WEIGHTS_ENGINE = "perception_stack/weights/stop_sign.engine"
IMG_SIZE       = 416    # must match SIGN_IMG_SIZE in config.py


def main():
    if not os.path.isfile(WEIGHTS_PT):
        print(f"[export_trt] ERROR: weights not found at '{WEIGHTS_PT}'")
        print("  Train first:  python scripts/train_stop_sign.py --api-key YOUR_KEY")
        sys.exit(1)

    print(f"[export_trt] Exporting {WEIGHTS_PT}  →  TensorRT FP16")
    print(f"             imgsz={IMG_SIZE}  device=0  half=True")
    print("             This will take 2-10 minutes on first run (engine build).")

    from ultralytics import YOLO
    model = YOLO(WEIGHTS_PT)

    t0 = time.time()
    model.export(
        format   = "engine",
        imgsz    = IMG_SIZE,
        device   = 0,
        half     = True,       # FP16 — Orin Ampere GPU has native FP16 support
        simplify = True,
        dynamic  = False,      # fixed batch=1 for predictable latency
        workspace= 4,          # GB — TRT engine build workspace
    )
    elapsed = time.time() - t0

    # ultralytics exports to same directory as .pt with .engine extension
    expected_out = WEIGHTS_PT.replace(".pt", ".engine")
    if os.path.isfile(expected_out):
        if expected_out != WEIGHTS_ENGINE:
            os.rename(expected_out, WEIGHTS_ENGINE)
        print(f"\n[export_trt] Done in {elapsed:.0f}s")
        print(f"             Engine saved to: {WEIGHTS_ENGINE}")
        print(f"\n  In config.py, change:")
        print(f'    SIGN_MODEL_PATH = "{WEIGHTS_ENGINE}"')
    else:
        print(f"[export_trt] WARNING: expected output not found at {expected_out}")
        print("             Check ultralytics export output above for the actual path.")


if __name__ == "__main__":
    main()
