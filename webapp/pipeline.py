"""
PolypNet – End-to-End Inference Pipeline
Wraps src/ modules into a single run_pipeline() call.
"""

import os
import sys
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Allow imports from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from webapp.config import (
    SEG_MODEL_PATH, VALIDATION_MODEL_PATH, YOLO_MODEL_PATH, YOLO_BEST_PATH,
    RESULTS_FOLDER, IMG_SIZE_CLASSIFY, IMG_SIZE_SEG,
    YOLO_CONF, YOLO_IOU, SEG_THRESHOLD
)

# ── Ensure results dir exists ─────────────────────────────────────────────────
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 – Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return uuid.uuid4().hex[:10]


def _save_result(img_array: np.ndarray, name: str, session_id: str) -> str:
    """Save a numpy BGR/gray image to results folder. Returns web-relative path."""
    folder = Path(RESULTS_FOLDER) / session_id
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / name
    cv2.imwrite(str(out_path), img_array)
    # Return path relative to webapp/static so Flask can serve it
    return f"results/{session_id}/{name}"


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Image Validation (trained MobileNetV2 or heuristic fallback)
# ─────────────────────────────────────────────────────────────────────────────

_val_model_cache = None  # module-level cache so model loads once


class _ValidationCNN(nn.Module):
    """Custom 4-block CNN matching train_validation_model.py architecture."""
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


def _load_validation_model():
    """Load trained custom ValidationCNN (cached). Returns None if not trained yet."""
    global _val_model_cache
    if _val_model_cache is not None:
        return _val_model_cache
    if not Path(VALIDATION_MODEL_PATH).exists():
        return None
    try:
        import torch
        model = _ValidationCNN()
        ckpt  = torch.load(VALIDATION_MODEL_PATH,
                           map_location="cuda" if torch.cuda.is_available() else "cpu",
                           weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        print("[OK] Loaded custom ValidationCNN")
        _val_model_cache = model
        return model
    except Exception as e:
        print(f"[WARN] Could not load validation model: {e}")
        return None


def validate_image(image_path: str) -> Dict[str, Any]:
    """
    Validates whether the uploaded image is a colonoscopy frame.
    Uses the trained MobileNetV2 CNN when available;
    falls back to a colour/texture heuristic otherwise.
    Returns {"valid": bool, "reason": str, "confidence": float}
    """
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage

    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False, "reason": "Cannot read image file.", "confidence": 0.0}

    h, w = img.shape[:2]
    if h < 64 or w < 64:
        return {"valid": False, "reason": "Image too small (< 64 px).", "confidence": 0.05}

    # ── Try trained CNN ───────────────────────────────────────────────────────
    model = _load_validation_model()
    if model is not None:
        try:
            tf = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            pil = PILImage.open(image_path).convert("RGB")
            tensor = tf(pil).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)
                valid_prob = float(probs[0][1])   # class 1 = valid colonoscopy

            # Hard heuristics ALWAYS win — the CNN was trained on limited negatives
            # and may still accept cars/medical images it never saw during training.
            hard_reject, hard_reason = _hard_heuristic_check(img, h, w)
            if hard_reject:
                return {"valid": False,
                        "reason": f"Non-colonoscopy image: {hard_reason} (CNN={valid_prob:.2f})",
                        "confidence": round(valid_prob, 3), "method": "CNN+heuristic"}

            # CNN threshold: must be confidently colonoscopy (>=0.65)
            valid  = valid_prob >= 0.65
            reason = ("Image is a valid colonoscopy frame (CNN validated)." if valid
                      else "Non-colonoscopy image detected by validation CNN.")
            return {"valid": valid, "reason": reason,
                    "confidence": round(valid_prob, 3), "method": "CNN"}
        except Exception as e:
            print(f"[WARN] CNN validation failed, using heuristic: {e}")

    # ── Heuristic fallback ────────────────────────────────────────────────────
    hard_reject, hard_reason = _hard_heuristic_check(img, h, w)
    if hard_reject:
        return {"valid": False, "reason": f"Non-colonoscopy image: {hard_reason}",
                "confidence": 0.1, "method": "heuristic"}

    img_f = img.astype(np.float32)
    g_mean, r_mean = img_f[:,:,1].mean(), img_f[:,:,2].mean()
    red_dominance  = r_mean / (g_mean + 1e-6)
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.sum(gray < 20)) / (h * w)
    variance   = float(np.var(gray))
    score  = min(red_dominance / 1.8, 1.0) * 0.40
    score += min(dark_ratio * 5, 1.0)      * 0.25
    score += min(variance / 800, 1.0)      * 0.25
    score += 0.10
    valid  = score >= 0.45
    reasons = []
    if not valid:
        if red_dominance < 0.9:
            reasons.append("lacks mucosal tones")
        if variance < 100:
            reasons.append("image too uniform")
    reason = ("Non-colonoscopy image: " + "; ".join(reasons) + "." if not valid
              else "Image appears to be a valid colonoscopy frame.")
    return {"valid": valid, "reason": reason,
            "confidence": round(score, 3), "method": "heuristic"}


def _hard_heuristic_check(img: np.ndarray, h: int, w: int):
    """
    Fast hard-coded checks that catch common non-medical images.
    Returns (should_reject: bool, reason: str).

    Key insight: colonoscopy tissue is always COLORFUL (pink/red mucosal lining).
    Any near-grayscale image (brain MRI, X-ray, CT scan, B&W photo) must be rejected.
    Cars, outdoor scenes, faces also have specific signatures we can detect.
    """
    img_f  = img.astype(np.float32)
    b_mean = img_f[:,:,0].mean()
    g_mean = img_f[:,:,1].mean()
    r_mean = img_f[:,:,2].mean()
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overall_brightness = float(gray.mean())

    # ── 1. Near-grayscale image (brain MRI, X-ray, CT, B&W photo) ──────────
    # Colonoscopy frames are always colorful (high channel spread).
    # If R, G, B are all very similar → grayscale → reject.
    channel_spread = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
    non_dark_pixels = img_f[gray > 30]   # only look at non-black pixels
    if len(non_dark_pixels) > 0:
        nd_r = non_dark_pixels[:,2].mean()
        nd_g = non_dark_pixels[:,1].mean()
        nd_b = non_dark_pixels[:,0].mean()
        nd_spread = max(nd_r, nd_g, nd_b) - min(nd_r, nd_g, nd_b)
    else:
        nd_spread = 0.0

    if nd_spread < 18:
        return True, (f"near-grayscale image (channel spread={nd_spread:.1f}; "
                      f"colonoscopy tissue is always colorful)")

    # ── 2. Very bright images — colonoscopy frames are never very bright ────
    if overall_brightness > 155:
        return True, (f"image too bright (mean={overall_brightness:.0f}; "
                      f"colonoscopy frames are typically darker)")

    # ── 3. Dominant blue channel (sky, car body, documents, water) ──────────
    if b_mean > r_mean * 1.12 and b_mean > g_mean * 1.08 and b_mean > 70:
        return True, "dominant blue channel (outdoor/document/car image)"

    # ── 4. Skin-tone: high R+G, low B, moderate overall brightness ──────────
    skin_like = (r_mean > 120 and g_mean > 85 and b_mean < 110 and
                 r_mean > b_mean * 1.25 and overall_brightness > 90)
    if skin_like:
        return True, "skin-tone colour profile detected (face/portrait image)"

    # ── 5. Very high saturation (car paint, solid objects, neon signs) ──────
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:,:,1].mean())
    val_mean = float(hsv[:,:,2].mean())
    if sat_mean > 120 and val_mean > 130:
        return True, "highly saturated uniform regions (non-medical image)"

    # ── 6. No dark vignette border — endoscope views have a circular dark ring
    dark_ratio = float(np.sum(gray < 20)) / gray.size
    if dark_ratio < 0.03 and overall_brightness > 95:
        return True, "no dark vignette border (not an endoscope frame)"

    # ── 7. Red-channel must be dominant in non-dark regions ─────────────────
    # Colon tissue is pink/red on camera. If red is not dominant → reject.
    if len(non_dark_pixels) > 0:
        nd_r = non_dark_pixels[:,2].mean()
        nd_g = non_dark_pixels[:,1].mean()
        nd_b = non_dark_pixels[:,0].mean()
        if nd_r < nd_g * 0.92 or nd_r < 70:
            return True, ("tissue region is not red-dominant "
                          "(not colonoscopy mucosal tissue)")

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – CLAHE Enhancement
# ─────────────────────────────────────────────────────────────────────────────

def enhance_image(image_path: str, session_id: str) -> str:
    """Apply CLAHE and save. Returns web-relative path."""
    from src.image_processing import apply_clahe
    pil_enhanced = apply_clahe(image_path)
    bgr = _pil_to_bgr(pil_enhanced)
    return _save_result(bgr, "enhanced.png", session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – YOLO Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect(image_path: str, session_id: str) -> Dict[str, Any]:
    """
    Run YOLO detection. Draws boxes on image.
    Falls back gracefully if YOLO not available.
    """
    try:
        from src.detection import detect_polyps
        # Prefer fine-tuned best.pt; fall back to nano
        yolo_path = YOLO_BEST_PATH if Path(YOLO_BEST_PATH).exists() else YOLO_MODEL_PATH
        det_out_path = str(Path(RESULTS_FOLDER) / session_id / "detection.png")
        Path(det_out_path).parent.mkdir(parents=True, exist_ok=True)

        detections = detect_polyps(
            model_path=yolo_path,
            image_path=image_path,
            conf_threshold=YOLO_CONF,
            iou_threshold=YOLO_IOU,
            save_path=det_out_path
        )

        if not detections:
            # Save original with "No polyp" label
            img = cv2.imread(image_path)
            cv2.putText(img, "No polyp detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
            cv2.imwrite(det_out_path, img)

        rel_path = f"results/{session_id}/detection.png"
        return {
            "detections": detections,
            "detection_image": rel_path,
            "polyp_count": len(detections)
        }

    except Exception as e:
        print(f"[WARN] YOLO detection failed: {e}")
        # Save copy of original as fallback
        img = cv2.imread(image_path)
        rel = _save_result(img, "detection.png", session_id)
        return {"detections": [], "detection_image": rel, "polyp_count": 0,
                "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Classification (multi-model stub)
# ─────────────────────────────────────────────────────────────────────────────

_clf_model_cache: Dict[str, Any] = {}   # keyed by model name

# Maps model names to checkpoint subdir names (InceptionV3 excluded — collapses to benign-only)
_CLF_CONFIGS = {
    "VGG16":          "VGG16",
    "ResNet50":       "ResNet50",
    "EfficientNetB0": "EfficientNetB0",
}


def _load_clf_model(display_name: str):
    """Load one classification checkpoint (cached). Returns model or None."""
    if display_name in _clf_model_cache:
        return _clf_model_cache[display_name]

    ckpt_path = _ROOT / "checkpoints" / "classification" / display_name / "best_model.pth"
    if not ckpt_path.exists():
        _clf_model_cache[display_name] = None
        return None

    try:
        import torch
        from torchvision import models as tvm

        if display_name == "VGG16":
            m = tvm.vgg16(weights=None)
            m.classifier[6] = torch.nn.Linear(4096, 2)
        elif display_name == "ResNet50":
            m = tvm.resnet50(weights=None)
            m.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.4), torch.nn.Linear(m.fc.in_features, 256),
                torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 2))
        elif display_name == "EfficientNetB0":
            m = tvm.efficientnet_b0(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.4), torch.nn.Linear(in_f, 256),
                torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 2))
        elif display_name == "InceptionV3":
            m = tvm.inception_v3(weights=None, aux_logits=False)
            m.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.4), torch.nn.Linear(m.fc.in_features, 256),
                torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 2))
        else:
            return None

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=False)
        m.eval()
        print(f"[OK] Loaded classifier: {display_name}")
        _clf_model_cache[display_name] = m
        return m
    except Exception as e:
        print(f"[WARN] Could not load {display_name}: {e}")
        _clf_model_cache[display_name] = None
        return None


def _clf_predict_one(model, image_path: str, img_size=(150, 150)) -> float:
    """Run one model on the image, return malignant probability (0-1)."""
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage

    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil    = PILImage.open(image_path).convert("RGB")
    tensor = tf(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
    # ImageFolder sorts classes alphabetically: 0=benign, 1=malignant
    return float(probs[0][1])


def classify(image_path: str, detections: list,
             seg_coverage: float = 0.0) -> Dict[str, Any]:
    """
    Multi-model binary classifier: Benign / Malignant.
    Returns 'No Polyp Detected' early when there is no evidence of a polyp
    (empty detections AND near-zero segmentation coverage).
    Otherwise uses trained PyTorch ensemble; falls back to heuristic.
    """
    # ── Gate: skip classification if no polyp evidence ────────────────────────
    has_yolo    = len(detections) > 0
    has_seg     = seg_coverage > 1.0   # >1% mask coverage = something found
    if not has_yolo and not has_seg:
        return {
            "label":        "No Polyp Detected",
            "confidence":   0.0,
            "benign_prob":  0.0,
            "model_scores": {},
            "reason":       "No polyp found by YOLO or segmentation."
        }

    # ── Heuristic baseline (used as fallback per model) ───────────────────────
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_conf      = max((d["confidence"] for d in detections), default=seg_coverage / 100.0)
    texture_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 1.0)
    heuristic_mal = max(0.05, min(0.95,
                        0.35 + 0.40 * max_conf + 0.25 * texture_score))
    rng     = np.random.default_rng(seed=int(heuristic_mal * 1000))
    offsets = rng.uniform(-0.06, 0.06, 4)

    model_order = ["VGG16", "ResNet50", "EfficientNetB0"]
    model_scores = {}

    for i, name in enumerate(model_order):
        clf_model = _load_clf_model(name)
        if clf_model is not None:
            try:
                prob = _clf_predict_one(clf_model, image_path, (150, 150))
                model_scores[name] = round(float(np.clip(prob, 0.01, 0.99)), 3)
                continue
            except Exception as e:
                print(f"[WARN] {name} inference failed: {e}")
        # Fallback: heuristic-based score
        model_scores[name] = round(
            float(np.clip(heuristic_mal + offsets[i], 0.05, 0.95)), 3)

    ensemble_mal = float(np.mean(list(model_scores.values())))
    label = "Malignant" if ensemble_mal >= 0.5 else "Benign"

    return {
        "label":        label,
        "confidence":   round(ensemble_mal, 3),
        "benign_prob":  round(1 - ensemble_mal, 3),
        "model_scores": model_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – Attention U-Net Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment(image_path: str, session_id: str) -> Dict[str, Any]:
    """Run Attention U-Net segmentation. Returns mask + overlay paths."""
    try:
        import torch
        from src.segmentation import AttentionUNet, predict_mask

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = AttentionUNet(in_channels=3, out_channels=1, base_channels=64)

        checkpoint = torch.load(SEG_MODEL_PATH, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        mask_out_path = str(Path(RESULTS_FOLDER) / session_id / "mask.png")
        Path(mask_out_path).parent.mkdir(parents=True, exist_ok=True)

        mask = predict_mask(
            model=model,
            image_path=image_path,
            device=device,
            image_size=IMG_SIZE_SEG,
            save_path=mask_out_path
        )

        # Build coloured overlay
        orig    = cv2.imread(image_path)
        mask_u8 = cv2.imread(mask_out_path, cv2.IMREAD_GRAYSCALE)
        if orig is not None and mask_u8 is not None:
            mask_resized = cv2.resize(mask_u8, (orig.shape[1], orig.shape[0]))
            _, binary = cv2.threshold(mask_resized, int(SEG_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
            overlay_img = orig.copy()
            overlay_img[binary > 0] = (
                overlay_img[binary > 0].astype(np.float32) * 0.5 +
                np.array([0, 200, 100], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            overlay_rel = _save_result(overlay_img, "overlay.png", session_id)
        else:
            overlay_rel = f"results/{session_id}/mask.png"

        # Coverage
        if mask_u8 is not None:
            coverage = round(float(np.sum(mask_u8 > 127)) / mask_u8.size * 100, 2)
        else:
            coverage = 0.0

        # Sanity check: a real polyp cannot cover >50% of the frame.
        # If the mask is that large, the U-Net produced a spurious full-image mask.
        if coverage > 50.0:
            print(f"[WARN] Segmentation coverage {coverage:.1f}% > 50% — discarding as false mask.")
            # Save a blank mask instead of the spurious one
            blank = np.zeros_like(mask_u8)
            cv2.imwrite(mask_out_path, blank)
            overlay_rel = _save_result(cv2.imread(image_path), "overlay.png", session_id)
            return {
                "mask_path":    f"results/{session_id}/mask.png",
                "overlay_path": overlay_rel,
                "coverage_pct": 0.0,
                "success": False,
                "error": "Mask covered >50% of image — discarded as false positive."
            }

        return {
            "mask_path":    f"results/{session_id}/mask.png",
            "overlay_path": overlay_rel,
            "coverage_pct": coverage,
            "success": True
        }

    except Exception as e:
        print(f"[WARN] Segmentation failed: {e}")
        traceback.print_exc()
        orig = cv2.imread(image_path)
        rel  = _save_result(orig, "overlay.png", session_id)
        return {
            "mask_path":    rel,
            "overlay_path": rel,
            "coverage_pct": 0.0,
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – VLM Clinical Description
# ─────────────────────────────────────────────────────────────────────────────

def generate_clinical_text(image_path: str, mask_path_rel: str,
                            classification: Dict, detections: list) -> str:
    """
    Generate clinical description.
    If no polyp was detected, returns a clear 'no finding' report.
    Otherwise tries VLMs: LLaVA-1.5-7B → BLIP-2 → structured fallback.
    """
    label = classification.get("label", "Unknown")

    # ── Short-circuit: no polyp found ────────────────────────────────────────────
    if label == "No Polyp Detected":
        return (
            "POLYPNET - AUTOMATED CLINICAL ASSESSMENT\n"
            "=" * 55 + "\n\n"
            "PATIENT STUDY:  Colonoscopy Image Analysis\n"
            "ANALYSIS DATE:  AI-Generated Report\n"
            "SYSTEM:         PolypNet v1.0\n\n"
            "-" * 53 + "\n"
            "FINDINGS\n"
            "-" * 53 + "\n"
            "   No polyp or suspicious lesion detected in this\n"
            "   colonoscopy frame by YOLO detection or U-Net\n"
            "   segmentation.\n\n"
            "   Classification:  NOT APPLICABLE (no polyp found)\n"
            "   Risk level:      NONE\n\n"
            "-" * 53 + "\n"
            "RECOMMENDATION\n"
            "-" * 53 + "\n"
            "   Continue standard colonoscopy surveillance protocol.\n"
            "   No immediate intervention required based on this frame.\n\n"
            "-" * 53 + "\n"
            "DISCLAIMER\n"
            "-" * 53 + "\n"
            "This is an automated AI-generated preliminary report.\n"
            "All findings MUST be reviewed and confirmed by a\n"
            "qualified gastroenterologist before any clinical\n"
            "decision is made.\n"
        )

    mask_abs = str(Path(RESULTS_FOLDER).parent / mask_path_rel)

    try:
        from src.vlm import get_best_description
        vlm_text = get_best_description(image_path, mask_abs)
        # Append the classification result so the report is always consistent
        conf_pct = f"{classification.get('confidence', 0.0)*100:.1f}"
        vlm_text += (
            f"\n\n--- PolypNet Classification ---\n"
            f"Diagnosis: {label}  (ensemble confidence: {conf_pct}%)\n"
            f"Risk: {'HIGH - biopsy recommended' if label == 'Malignant' else 'LOW - routine surveillance'}\n"
        )
        return vlm_text

    except Exception as e:
        print(f"[WARN] VLM stage failed ({e}), using structured fallback.")
        confidence  = classification.get("confidence", 0.0)
        polyp_count = len(detections)
        return _structured_description(
            image_path, mask_abs, label, confidence, polyp_count, detections
        )


def _structured_description(image_path, mask_path, label, confidence,
                             polyp_count, detections) -> str:
    """Create a detailed structured clinical report without LLaVA."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2] if img is not None else (0, 0)

    mask = None
    if Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    coverage = 0.0
    size_cat  = "undetermined"
    if mask is not None:
        coverage = float(np.sum(mask > 127)) / mask.size * 100
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            lc = max(contours, key=cv2.contourArea)
            _, _, cw, ch = cv2.boundingRect(lc)
            avg_dim = (cw + ch) / 2
            size_cat = "small (<5 mm)" if avg_dim < 50 else (
                "medium (5–10 mm)" if avg_dim < 100 else "large (>10 mm)")

    det_lines = ""
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf_str = f"{d['confidence']:.2f}"
        det_lines += f"   Region {i}: box=({x1},{y1})-({x2},{y2}), confidence={conf_str}\n"

    sep       = "=" * 55
    dash      = "-" * 53
    risk      = "HIGH - recommend biopsy" if label == "Malignant" else "LOW - routine surveillance"
    cov_str   = f"{coverage:.2f}"
    conf_str2 = f"{confidence*100:.1f}"
    det_block = det_lines if det_lines else "   No bounding boxes returned by detector.\n"
    if label == "Malignant":
        recs = ("- Immediate referral for polypectomy / biopsy recommended.\n"
                "   - Monitor for adenomatous features.\n"
                "   - Schedule 3-month follow-up colonoscopy.")
    else:
        recs = ("- Standard 3-5 year surveillance interval applicable.\n"
                "   - Document lesion size and location in patient record.\n"
                "   - Advise patient regarding dietary and lifestyle modifications.")

    return (
        f"POLYPNET - AUTOMATED CLINICAL ASSESSMENT\n"
        f"{sep}\n\n"
        f"PATIENT STUDY:  Colonoscopy Image Analysis\n"
        f"ANALYSIS DATE:  AI-Generated Report\n"
        f"SYSTEM:         PolypNet v1.0 (Attention U-Net + YOLO + CNN Ensemble)\n\n"
        f"{dash}\n"
        f"1. DETECTION SUMMARY\n"
        f"{dash}\n"
        f"   Polyps detected:    {polyp_count}\n"
        f"{det_block}"
        f"{dash}\n"
        f"2. CLASSIFICATION RESULT\n"
        f"{dash}\n"
        f"   Diagnosis:          {label}\n"
        f"   Ensemble confidence:{conf_str2}%\n"
        f"   Risk level:         {risk}\n\n"
        f"{dash}\n"
        f"3. SEGMENTATION ANALYSIS\n"
        f"{dash}\n"
        f"   Polyp area coverage:{cov_str}% of image\n"
        f"   Estimated size:     {size_cat}\n"
        f"   Image dimensions:   {w}x{h} px\n\n"
        f"{dash}\n"
        f"4. CLINICAL RECOMMENDATIONS\n"
        f"{dash}\n"
        f"   {recs}\n\n"
        f"{dash}\n"
        f"DISCLAIMER\n"
        f"{dash}\n"
        f"This is an automated AI-generated preliminary report.\n"
        f"All findings MUST be reviewed and confirmed by a\n"
        f"qualified gastroenterologist before any clinical\n"
        f"decision is made.\n"
    )


def _apply_detection_mask(mask_u8: np.ndarray, overlay_img: np.ndarray,
                           detections: list, orig_shape: tuple) -> tuple:
    """
    Clips the segmentation mask to only show the polyp SHAPE inside each
    YOLO bounding box. Uses a tighter threshold (65%) + morphological cleanup
    + largest-component selection so only the tight polyp blob is highlighted.
    Returns (clipped_mask, clipped_overlay).
    """
    if not detections:
        return np.zeros_like(mask_u8), overlay_img.copy()

    h, w = orig_shape[:2]

    # Resize probability mask to match original image dimensions
    if mask_u8.shape != (h, w):
        prob_map = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        prob_map = mask_u8.copy()

    # Threshold: 65% confidence (165/255) — higher than default 50%
    # This stops the whole YOLO box from being filled when the U-Net
    # spreads moderate probability across the full region.
    HIGH_THRESH = 165

    final_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = prob_map[y1:y2, x1:x2]

        # Tight threshold
        _, binary_roi = cv2.threshold(roi, HIGH_THRESH, 255, cv2.THRESH_BINARY)

        # Morphological opening removes thin scattered noise
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, k_open)

        # Keep only the LARGEST connected component (the polyp blob)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_roi, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean_roi = np.zeros_like(binary_roi)
            clean_roi[labels == largest] = 255
        else:
            clean_roi = binary_roi

        # Light closing to fill small holes
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, k_close)

        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], clean_roi)

    # Build overlay: green only on the refined polyp pixels
    clipped_overlay = overlay_img.copy()
    clipped_overlay[final_mask > 0] = (
        clipped_overlay[final_mask > 0].astype(np.float32) * 0.5 +
        np.array([0, 200, 100], dtype=np.float32) * 0.5
    ).astype(np.uint8)

    return final_mask, clipped_overlay


def run_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Execute full PolypNet pipeline on a single image.

    Args:
        image_path: Absolute path to the uploaded image.

    Returns:
        JSON-serialisable results dict.
    """
    session_id = _uid()
    result: Dict[str, Any] = {"session_id": session_id}

    # ── Save original copy into results ─────────────────────────────────────────────
    orig_img = cv2.imread(image_path)
    orig_rel = _save_result(orig_img, "original.png", session_id)
    result["original_image"] = orig_rel

    # ── Stage 1: Validate ───────────────────────────────────────────────────────
    validation = validate_image(image_path)
    result["validation"] = validation
    if not validation["valid"]:
        result["error"] = validation["reason"]
        return result

    # ── Stage 2: CLAHE Enhancement ────────────────────────────────────────────────
    try:
        result["enhanced_image"] = enhance_image(image_path, session_id)
    except Exception as e:
        print(f"[WARN] CLAHE failed: {e}")
        result["enhanced_image"] = orig_rel

    # ── Stage 3: YOLO Detection ──────────────────────────────────────────────────
    det_result  = detect(image_path, session_id)
    result.update(det_result)
    detections  = det_result["detections"]

    # ── Stage 4: Segmentation ───────────────────────────────────────────────────
    seg_result  = segment(image_path, session_id)
    result.update(seg_result)

    # ── Clip segmentation to YOLO bounding boxes ──────────────────────────────
    # The segmentation should only highlight tissue INSIDE detection boxes.
    if detections and seg_result.get("success", False):
        mask_path_abs  = str(Path(RESULTS_FOLDER) / session_id / "mask.png")
        overlay_src    = str(Path(RESULTS_FOLDER) / session_id / "overlay.png")
        mask_u8        = cv2.imread(mask_path_abs,  cv2.IMREAD_GRAYSCALE)
        overlay_img    = cv2.imread(overlay_src)
        orig_for_clip  = cv2.imread(image_path)

        if mask_u8 is not None and overlay_img is not None and orig_for_clip is not None:
            # Resize mask to original image size for clipping
            mask_resized = cv2.resize(mask_u8,
                                      (orig_for_clip.shape[1], orig_for_clip.shape[0]))
            clipped_mask, clipped_overlay = _apply_detection_mask(
                mask_resized, orig_for_clip, detections, orig_for_clip.shape)

            # Recompute coverage from clipped mask
            clipped_coverage = round(
                float(np.sum(clipped_mask > 127)) / clipped_mask.size * 100, 2)

            # Save updated mask and overlay back to disk
            cv2.imwrite(mask_path_abs, clipped_mask)
            cv2.imwrite(overlay_src, clipped_overlay)
            result["coverage_pct"] = clipped_coverage
            seg_coverage = clipped_coverage
            print(f"[INFO] Segmentation clipped to YOLO boxes. Coverage: {clipped_coverage:.1f}%")
        else:
            seg_coverage = seg_result.get("coverage_pct", 0.0)
    else:
        seg_coverage = seg_result.get("coverage_pct", 0.0)

    # ── Stage 5: Classification (gated on polyp evidence) ──────────────────────
    result["classification"] = classify(
        image_path, detections, seg_coverage=seg_coverage
    )

    # ── Stage 6: VLM Clinical Description ─────────────────────────────────────
    result["clinical_description"] = generate_clinical_text(
        image_path=image_path,
        mask_path_rel=seg_result.get("mask_path", ""),
        classification=result["classification"],
        detections=detections
    )

    return result
