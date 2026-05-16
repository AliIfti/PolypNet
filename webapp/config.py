"""
PolypNet Web Application – Configuration
"""

import os
from pathlib import Path

# Project root (one level up from webapp/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model Paths ──────────────────────────────────────────────────────────────
SEG_MODEL_PATH        = str(BASE_DIR / "checkpoints" / "segmentation" / "unet_bce_dice_best.pth")
VALIDATION_MODEL_PATH = str(BASE_DIR / "checkpoints" / "validation" / "validation_model.pth")  # MobileNetV2
YOLO_MODEL_PATH       = str(BASE_DIR / "yolo11n.pt")          # nano pre-trained
YOLO_BEST_PATH        = str(BASE_DIR / "runs" / "detect" / "runs" / "detect" / "polyp_yolo4" / "weights" / "best.pt")

# ── Upload / Output Dirs ──────────────────────────────────────────────────────
UPLOAD_FOLDER  = str(BASE_DIR / "webapp" / "static" / "uploads")
RESULTS_FOLDER = str(BASE_DIR / "webapp" / "static" / "results")

# ── Image dimensions ──────────────────────────────────────────────────────────
IMG_SIZE_CLASSIFY = (150, 150)
IMG_SIZE_SEG      = (256, 256)

# ── Inference thresholds ──────────────────────────────────────────────────────
YOLO_CONF      = 0.35
YOLO_IOU       = 0.45
SEG_THRESHOLD  = 0.5

# ── Allowed upload extensions ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# ── Flask secret key ──────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("POLYPNET_SECRET", "polypnet-dev-secret-key-2024")

# ── Max upload size (16 MB) ───────────────────────────────────────────────────
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
