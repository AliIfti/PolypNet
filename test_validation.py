"""
PolypNet – Validation Model Quick Test
=======================================
Tests the trained validation CNN on any image you provide.

Usage:
  # Test a single image
  python3 test_validation.py /path/to/image.jpg

  # Test an entire folder of images
  python3 test_validation.py /path/to/folder/

  # Batch test: colonoscopy images (should all be VALID)
  python3 test_validation.py PolypsSet/val2019/Image/1/ --limit 10

Examples:
  python3 test_validation.py some_selfie.jpg
  python3 test_validation.py PolypsSet/train2019/Image/10000.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage


# ─── Model definition (must match train_validation_model.py exactly) ──────────
class ValidationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.head(self.gap(self.features(x)).flatten(1))


# ─── Load model ───────────────────────────────────────────────────────────────
CKPT      = Path(__file__).parent / "checkpoints" / "validation" / "validation_model.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_TF    = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    if not CKPT.exists():
        print(f"❌ Checkpoint not found: {CKPT}")
        sys.exit(1)
    model = ValidationCNN()
    ckpt  = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


# ─── Predict one image ────────────────────────────────────────────────────────
def predict(model, image_path: str):
    try:
        pil    = PILImage.open(image_path).convert("RGB")
        tensor = IMG_TF(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        invalid_prob = float(probs[0])
        valid_prob   = float(probs[1])
        label        = "✅ VALID   (colonoscopy)" if valid_prob >= 0.5 else "❌ INVALID (non-colonoscopy)"
        return label, valid_prob, invalid_prob
    except Exception as e:
        return f"⚠  Error: {e}", 0.0, 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test PolypNet image validation CNN")
    parser.add_argument("input",
        help="Path to an image file or folder")
    parser.add_argument("--limit", type=int, default=0,
        help="Max images to test from a folder (0 = all)")
    args = parser.parse_args()

    model = load_model()
    print(f"\n  Model  : {CKPT.name}")
    print(f"  Device : {DEVICE}")

    path = Path(args.input)

    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted(list(path.glob("*.jpg")) +
                        list(path.glob("*.jpeg")) +
                        list(path.glob("*.png")) +
                        list(path.glob("*.bmp")))
        if args.limit > 0:
            images = images[:args.limit]
    else:
        print(f"❌ Not found: {path}")
        sys.exit(1)

    print(f"\n  Testing {len(images)} image(s)…\n")
    print(f"  {'Filename':<40} {'Result':<30} {'Valid%':>6}  {'Invalid%':>8}")
    print("  " + "─" * 90)

    valid_count = invalid_count = 0
    for img_path in images:
        label, vp, ip = predict(model, str(img_path))
        name = img_path.name[:38]
        print(f"  {name:<40} {label:<30} {vp*100:>5.1f}%  {ip*100:>7.1f}%")
        if vp >= 0.5:
            valid_count += 1
        else:
            invalid_count += 1

    print("  " + "─" * 90)
    print(f"\n  Summary: {valid_count} VALID  |  {invalid_count} INVALID  "
          f"(out of {len(images)} images)\n")


if __name__ == "__main__":
    main()
