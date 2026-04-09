"""
PSU Eco Racing — Fine-tune Segformer-B2 on custom track images.

Strategy: keep the pretrained Cityscapes architecture (19 classes, class 0 = road).
Only supervise on annotated road pixels — everything else is ignored in the loss.
This lets 50 images shift the model toward your specific track surface without
disturbing the pretrained weights for non-road classes.

Roboflow export expected structure:
    <dataset_dir>/
        train/
            images/   *.jpg / *.png
            masks/    *.png   pixel=1 → road,  pixel=0 → background/ignore
        valid/
            images/
            masks/

Usage:
    python perception_stack/scripts/finetune_segformer.py \
        --data  /path/to/roboflow_export \
        --out   perception_stack/weights/segformer_track \
        --epochs 25
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
ROAD_CLASS  = 0      # Cityscapes class 0 = road — keep unchanged
IGNORE_IDX  = 255    # pixels not annotated as road → ignored in loss
IMG_SIZE    = (512, 512)
BATCH_SIZE  = 4
LR          = 6e-5
WEIGHT_DECAY= 0.01


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrackDataset(Dataset):
    """
    Loads image + Roboflow PNG mask pairs.
    Mask remapping:
        Roboflow pixel=1 (road)  → label ROAD_CLASS (0)
        Roboflow pixel=0 (bg)    → label IGNORE_IDX (255, not supervised)
    With augmentation for small datasets (flip + colour jitter).
    """

    def __init__(self, img_dir: Path, mask_dir: Path,
                 processor: SegformerImageProcessor, augment: bool = False):
        self.imgs      = sorted(img_dir.glob("*"))
        self.masks     = sorted(mask_dir.glob("*"))
        assert len(self.imgs) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.imgs)} vs {len(self.masks)}"
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = Image.open(self.imgs[idx]).convert("RGB").resize(IMG_SIZE)
        mask = Image.open(self.masks[idx]).convert("L").resize(
            IMG_SIZE, resample=Image.NEAREST)

        # Augmentation (horizontal flip only — safe for road mask)
        if self.augment and np.random.rand() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Colour jitter (image only)
        if self.augment:
            import torchvision.transforms.functional as TF
            img = TF.adjust_brightness(img, 0.7 + np.random.rand() * 0.6)
            img = TF.adjust_contrast(img,   0.7 + np.random.rand() * 0.6)
            img = TF.adjust_saturation(img, 0.7 + np.random.rand() * 0.6)

        # Remap mask: 1 (road) → ROAD_CLASS,  0 (bg) → IGNORE_IDX
        mask_np = np.array(mask, dtype=np.int64)
        label   = np.full_like(mask_np, IGNORE_IDX)
        label[mask_np == 1] = ROAD_CLASS

        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # (3, H, W)

        return pixel_values, torch.tensor(label, dtype=torch.long)


# ── Loss ───────────────────────────────────────────────────────────────────────

def seg_loss(logits, labels, H, W):
    """
    Upsample logits to (H, W) then cross-entropy with ignore_index.
    Only road pixels (label=0) contribute — background is ignored.
    """
    up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return F.cross_entropy(up, labels, ignore_index=IGNORE_IDX)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"  if torch.backends.mps.is_available()
              else "cpu")
    print(f"[Fine-tune] Device: {device.upper()}")

    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)

    data_root = Path(args.data)
    train_ds  = TrackDataset(data_root / "train" / "images",
                             data_root / "train" / "masks",
                             processor, augment=True)
    valid_ds  = TrackDataset(data_root / "valid" / "images",
                             data_root / "valid" / "masks",
                             processor, augment=False)

    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
    valid_dl  = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)

    print(f"[Fine-tune] Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model = model.to(device).train()

    # Differential LR: encoder gets 10× lower LR (already well-trained)
    encoder_params = list(model.segformer.parameters())
    decoder_params = list(model.decode_head.parameters())
    optimizer = AdamW([
        {"params": encoder_params, "lr": LR * 0.1},
        {"params": decoder_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for pv, labels in train_dl:
            pv, labels = pv.to(device), labels.to(device)
            H, W = labels.shape[-2], labels.shape[-1]
            logits = model(pixel_values=pv).logits
            loss   = seg_loss(logits, labels, H, W)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        road_iou_sum = 0.0
        with torch.no_grad():
            for pv, labels in valid_dl:
                pv, labels = pv.to(device), labels.to(device)
                H, W = labels.shape[-2], labels.shape[-1]
                logits = model(pixel_values=pv).logits
                val_loss += seg_loss(logits, labels, H, W).item()

                # Road IoU (class 0 only — what the pipeline uses)
                pred  = F.interpolate(logits, size=(H, W),
                                      mode="bilinear", align_corners=False)
                pred  = pred.argmax(dim=1)
                valid = labels != IGNORE_IDX
                tp = ((pred == 0) & (labels == 0) & valid).sum().item()
                fp = ((pred == 0) & (labels != 0) & valid).sum().item()
                fn = ((pred != 0) & (labels == 0) & valid).sum().item()
                iou = tp / (tp + fp + fn + 1e-6)
                road_iou_sum += iou

        val_loss     /= len(valid_dl)
        road_iou      = road_iou_sum / len(valid_dl)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"road_IoU={road_iou:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            print(f"           ↑ saved best model  (val_loss={val_loss:.4f})")

    model.save_pretrained(out_dir / "last")
    processor.save_pretrained(out_dir / "last")
    print(f"\n[Fine-tune] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[Fine-tune] Model saved to: {out_dir / 'best'}")
    print(f"\nTo use in pipeline, update config.py:")
    print(f'  SEG_MODEL_ID = "{out_dir / "best"}"')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True,
                        help="Path to Roboflow export root (contains train/ valid/)")
    parser.add_argument("--out",    default="perception_stack/weights/segformer_track",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    train(parser.parse_args())
