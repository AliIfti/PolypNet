"""
PolypNet – Stage 1: Image Validation CNN  (FIXED)
===================================================
Binary classifier:  Class 0 = Invalid  |  Class 1 = Valid (colonoscopy)

WHY THE FIRST VERSION GOT 100%
  The original negatives were trivially easy: random noise, solid colours.
  Any CNN learns that instantly. Accuracy was 100% from epoch 1.

WHAT THIS VERSION DOES INSTEAD
  Hard negatives that simulate REAL failure cases an endoscope system sees:
    1. Completely dark frames        (scope inside cap / outside body)
    2. Over-exposed / white frames   (scope touching mucosa too close)
    3. Heavy-blur ghosting           (motion artifact during withdrawal)
    4. Colour-inverted frames        (white-balance failure)
    5. Greyscale frames              (monochrome camera / format error)
    6. Heavy JPEG / block noise      (transmission corruption)
    7. Partial dark vignette only    (just the endoscope border, no content)
    8. Text-document-like patterns   (mistakenly uploaded document scan)
    9. Skin-tone patches             (external camera pointing at skin)
   10. Random mosaic blocks          (pixel permutation / format corruption)

  These are all derived from the real colonoscopy images themselves, so
  no external dataset is needed.  Expected val accuracy: 82–93%.

Usage (in your terminal):
  cd /home/ali/Desktop/fyp/fyp
  python3 train_validation_model.py
  python3 train_validation_model.py --epochs 20 --batch 32 --samples 3000

Output:
  checkpoints/validation/validation_model.pth   (best checkpoint)
  output/validation/training_curves.png
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent
POLYPS_DIR  = BASE / "PolypsSet" / "train2019" / "Image"
SAVE_DIR    = BASE / "checkpoints" / "validation"
OUTPUT_DIR  = BASE / "output" / "validation"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (150, 150)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_hard_negatives(pos_paths: list, n: int) -> list:
    """
    Generate n hard-negative images derived from real colonoscopy images.
    Returns list of numpy uint8 arrays (H×W×3).
    """
    random.seed(99)
    negatives = []
    src_pool  = [cv2.imread(str(p)) for p in random.sample(pos_paths, min(n * 3, len(pos_paths)))]
    src_pool  = [x for x in src_pool if x is not None]

    mode_fns = [
        _neg_dark_frame,
        _neg_overexposed,
        _neg_heavy_blur,
        _neg_colour_invert,
        _neg_greyscale,
        _neg_jpeg_corruption,
        _neg_border_only,
        _neg_text_document,
        _neg_skin_patch,
        _neg_mosaic_blocks,
    ]

    for i in range(n):
        src = src_pool[i % len(src_pool)]
        fn  = mode_fns[i % len(mode_fns)]
        try:
            neg = fn(src.copy())
            neg = cv2.resize(neg, IMG_SIZE)
            negatives.append(neg)
        except Exception:
            negatives.append(np.zeros((*IMG_SIZE, 3), dtype=np.uint8))

    return negatives


def _neg_dark_frame(img):
    """Scope inside cap or before insertion – nearly black."""
    dark = np.zeros_like(img)
    # Tiny central glow to simulate slight light bleed
    h, w = img.shape[:2]
    cv2.circle(dark, (w//2, h//2), random.randint(5, 20),
               (random.randint(0,60),)*3, -1)
    return dark

def _neg_overexposed(img):
    """Scope touching mucosa – blown-out white."""
    factor = random.uniform(4.0, 8.0)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _neg_heavy_blur(img):
    """Severe motion blur – unreadable frame."""
    k = random.choice([51, 71, 91, 111])
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    # Add motion kernel on top
    size = random.randint(30, 60)
    kernel = np.zeros((size, size))
    kernel[size//2, :] = 1.0 / size
    return cv2.filter2D(blurred, -1, kernel)

def _neg_colour_invert(img):
    """White-balance failure / inverted frame."""
    return cv2.bitwise_not(img)

def _neg_greyscale(img):
    """Monochrome / format error – no colour information."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Randomly shift brightness to make it look like xray
    gray = np.clip(gray.astype(np.int32) + random.randint(-60, 60), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _neg_jpeg_corruption(img):
    """Heavy JPEG block artefacts (low quality = 1-5)."""
    q  = random.randint(1, 5)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def _neg_border_only(img):
    """Only the dark circular endoscope border, no tissue content."""
    h, w = img.shape[:2]
    # Fill centre with black, keep vignette ring
    out = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//2 - 10, 255, -1)
    out[mask > 0] = 0   # black out everything inside vignette
    return out

def _neg_text_document(img):
    """Simulate a scanned document (white bg + dark text lines)."""
    h, w = img.shape[:2]
    doc  = np.ones((h, w, 3), dtype=np.uint8) * random.randint(220, 255)
    for y in range(20, h - 20, random.randint(18, 28)):
        x1  = random.randint(10, 30)
        x2  = random.randint(w - 30, w - 10)
        col = random.randint(0, 60)
        thickness = random.randint(1, 3)
        cv2.line(doc, (x1, y), (x2, y), (col, col, col), thickness)
    return doc

def _neg_skin_patch(img):
    """External camera pointing at skin – uniform pinkish/brownish patch."""
    h, w  = img.shape[:2]
    r = random.randint(140, 200)
    g = random.randint(80, 130)
    b = random.randint(60, 100)
    skin = np.full((h, w, 3), (b, g, r), dtype=np.uint8)
    # Add micro-texture
    noise = np.random.randint(-15, 15, (h, w, 3))
    return np.clip(skin.astype(np.int32) + noise, 0, 255).astype(np.uint8)

def _neg_mosaic_blocks(img):
    """Pixel-permutation / format corruption – mosaic scramble."""
    h, w = img.shape[:2]
    out  = img.copy()
    bsize = random.randint(12, 30)
    for y in range(0, h - bsize, bsize):
        for x in range(0, w - bsize, bsize):
            patch = out[y:y+bsize, x:x+bsize].copy()
            np.random.shuffle(patch)
            out[y:y+bsize, x:x+bsize] = patch
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationDataset(Dataset):
    def __init__(self, samples: list, augment: bool = False):
        """
        samples: list of (image_source, label)
            image_source = Path  → load from disk
            image_source = ndarray (BGR) → use as-is
        label: 0=invalid, 1=valid
        """
        self.samples = samples
        random.shuffle(self.samples)
        self.augment = augment
        ops = [transforms.Resize(IMG_SIZE)]
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                       saturation=0.15, hue=0.04),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, label = self.samples[idx]
        if isinstance(source, (str, Path)):
            pil = PILImage.open(source).convert("RGB")
        else:  # ndarray BGR → PIL RGB
            pil = PILImage.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        return self.transform(pil), torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL  –  Custom CNN (lightweight, matches SDP "custom CNN" requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationCNN(nn.Module):
    """
    Small custom CNN for binary image validation.
    Input: 3 × 150 × 150
    Architecture:
      3 Conv blocks (Conv→BN→ReLU→MaxPool)
      Global Average Pooling
      Dropout → FC128 → FC2
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 – 150→75
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2 – 75→37
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3 – 37→18
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4 – 18→9
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)   # → 256×1×1
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    print("\n" + "="*65)
    print("  PolypNet – Stage 1: Image Validation CNN")
    print("  Negatives: CIFAR-10 real photos  +  synthetic hard negatives")
    print("="*65)
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Samples : {args.samples} per class")

    # ── Positive samples (real colonoscopy) ───────────────────────────────────
    print("\n[1/4] Loading colonoscopy images (positive)…")
    all_pos = list(POLYPS_DIR.glob("*.jpg")) + list(POLYPS_DIR.glob("*.png"))
    if not all_pos:
        raise FileNotFoundError(f"No images in {POLYPS_DIR}")
    random.seed(42)
    pos_paths = random.sample(all_pos, min(args.samples, len(all_pos)))
    print(f"  ✅ {len(pos_paths)} valid colonoscopy images")

    # ── Negative samples ──────────────────────────────────────────────────────
    real_neg_samples = []   # (Path, 0)  – real non-colonoscopy images
    if args.neg_dir:
        neg_dir = Path(args.neg_dir)
        if neg_dir.is_dir():
            real_paths = (list(neg_dir.glob("*.jpg")) +
                          list(neg_dir.glob("*.jpeg")) +
                          list(neg_dir.glob("*.png")))
            random.shuffle(real_paths)
            n_real = min(len(real_paths), args.samples // 2)
            real_neg_samples = [(p, 0) for p in real_paths[:n_real]]
            print(f"\n[2/4] Real negatives (CIFAR-10): {len(real_neg_samples)} images from {neg_dir}")
        else:
            print(f"\n[2/4] ⚠  --neg-dir '{args.neg_dir}' not found; using synthetic only.")

    # Synthetic hard negatives fill the remaining quota
    n_synth = args.samples - len(real_neg_samples)
    print(f"  Generating {n_synth} synthetic hard negatives…")
    print("      Types: dark frame, overexposed, heavy blur, colour invert,")
    print("             greyscale, JPEG corruption, border-only, text doc,")
    print("             skin patch, mosaic scramble")
    neg_arrays = make_hard_negatives(all_pos, n_synth)
    synth_neg_samples = [(arr, 0) for arr in neg_arrays]   # (ndarray BGR, label=0)
    print(f"  ✅ {len(synth_neg_samples)} synthetic negatives generated")

    all_neg_samples = real_neg_samples + synth_neg_samples
    print(f"  Total negatives: {len(all_neg_samples)}  "
          f"({len(real_neg_samples)} real + {len(synth_neg_samples)} synthetic)")

    # ── 80/20 split ───────────────────────────────────────────────────────────
    n       = min(len(pos_paths), len(all_neg_samples))
    split   = int(0.8 * n)

    random.shuffle(all_neg_samples)
    pos_sample = [(p, 1) for p in pos_paths[:n]]
    neg_s      = all_neg_samples[:n]

    train_samples = pos_sample[:split] + neg_s[:split]
    val_samples   = pos_sample[split:] + neg_s[split:]

    train_ds = ValidationDataset(train_samples, augment=True)
    val_ds   = ValidationDataset(val_samples,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n  Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"  (Not trivially separable – expect ~82–93% accuracy)")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Building custom CNN…")
    model     = ValidationCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {args.epochs} epochs…\n")
    history    = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_v_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tl = tc = tt = 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"Ep {epoch:02d}/{args.epochs} [Train]",
                                  leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labels).sum().item()
            tt += imgs.size(0)

        # Validate
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                      desc=f"Ep {epoch:02d}/{args.epochs} [Val]  ",
                                      leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        scheduler.step()
        t_acc = tc / tt;  t_l = tl / tt
        v_acc = vc / vt;  v_l = vl / vt
        history["train_loss"].append(t_l);  history["train_acc"].append(t_acc)
        history["val_loss"].append(v_l);    history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:02d}/{args.epochs}  "
              f"Train Loss={t_l:.4f} Acc={t_acc*100:.1f}%  |  "
              f"Val Loss={v_l:.4f} Acc={v_acc*100:.1f}%")

        if v_acc > best_v_acc:
            best_v_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "architecture": "ValidationCNN (custom)",
            }, SAVE_DIR / "validation_model.pth")
            print(f"  💾 Best model saved  (val_acc={v_acc*100:.1f}%)")

    torch.save(model.state_dict(), SAVE_DIR / "validation_model_final.pth")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ep_range = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Image Validation CNN – Training Curves", fontsize=14,
                 fontweight="bold")
    ax1.plot(ep_range, history["train_loss"], label="Train Loss")
    ax1.plot(ep_range, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True)
    ax2.plot(ep_range, [a*100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(ep_range, [a*100 for a in history["val_acc"]],   label="Val Acc")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True); ax2.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()

    print("\n" + "="*65)
    print(f"  ✅ Training complete!")
    print(f"  Best val accuracy : {best_v_acc*100:.1f}%")
    print(f"  Model saved  →  {SAVE_DIR}/validation_model.pth")
    print(f"  Curves saved →  {OUTPUT_DIR}/training_curves.png")
    print("="*65)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 1 – Image Validation CNN")
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--samples", type=int,   default=3000)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--neg-dir", type=str,   default=None,
        help="Path to folder of REAL non-colonoscopy images (from prepare_negatives.py). "
             "If not set, uses synthetic hard negatives.")
    args = p.parse_args()
    args.neg_dir = args.neg_dir  # may be None
    train(args)
