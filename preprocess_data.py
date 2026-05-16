"""
PolypNet – Complete Data Preprocessing Pipeline
================================================
Steps performed:
  1. Audit dataset and report counts
  2. Build classification dataset (benign / malignant labels from class folders)
      → resizes to 150×150, normalises 0-1, saves as organised directory
  3. Validate every image (remove corrupt / too-small files)
  4. Data augmentation (rotation, flip, zoom, brightness)
  5. Print summary report

Run:
    python3 preprocess_data.py            # full run
    python3 preprocess_data.py --audit    # audit only, no disk writes
    python3 preprocess_data.py --no-aug   # skip augmentation
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parent
POLYPS_SET    = BASE / "PolypsSet"
CLF_DATASET   = BASE / "clf_dataset"        # classification ready dataset
SEG_DATASET   = BASE / "seg_dataset"        # already exists

# ─── Class → label mapping ────────────────────────────────────────────────────
# From dataset_metadata.json:
#   Classes 1-10  → Benign  (hyperplastic, adenoma, serrated, inflammatory …)
#   Classes 11-17 → Malignant (colorectal cancer T1-T4, adenocarcinoma …)
BENIGN_CLASSES    = {1,2,3,4,5,6,7,8,9,10}
MALIGNANT_CLASSES = {11,12,13,14,15,16,17,18,19,20,21,22,23,24}

CLASS_NAMES = {
    1:"Hyperplastic Polyp",       2:"Tubular Adenoma",
    3:"Tubulovillous Adenoma",    4:"Villous Adenoma",
    5:"Sessile Serrated Adenoma", 6:"Traditional Serrated Adenoma",
    7:"Inflammatory Polyp",       8:"Hamartomatous Polyp",
    9:"Lipoma",                   10:"Carcinoid Tumor",
    11:"Early Colorectal Cancer", 12:"Colorectal Cancer T2",
    13:"Advanced Colorectal Cancer",14:"Metastatic Colorectal Cancer",
    15:"Adenocarcinoma",          16:"Mucinous Adenocarcinoma",
    17:"Signet Ring Cell Carcinoma"
}

# ─── Image sizes ──────────────────────────────────────────────────────────────
CLF_SIZE = (150, 150)    # for validation + classification CNNs
SEG_SIZE = (256, 256)    # for U-Net segmentation

# ─── Augmentation params ──────────────────────────────────────────────────────
AUG_PER_IMAGE = 3        # generate 3 augmented copies per image


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def audit_dataset() -> Dict:
    """Print a full audit of the PolypsSet directory."""
    print("\n" + "="*65)
    print("  POLYPNET – DATASET AUDIT")
    print("="*65)

    stats = {}

    for split in ["train2019", "val2019", "test2019"]:
        split_dir = POLYPS_SET / split / "Image"
        if not split_dir.exists():
            print(f"\n⚠  {split}/Image not found – skipping")
            continue

        # Flat layout (train) vs class-subfolder layout (val/test)
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs:
            total = 0
            print(f"\n📁 {split}/ (multi-class):")
            benign_count = malignant_count = 0
            for d in sorted(subdirs, key=lambda x: int(x.name)):
                cid   = int(d.name)
                count = len(list(d.glob("*.*")))
                label = "Benign" if cid in BENIGN_CLASSES else "Malignant"
                cname = CLASS_NAMES.get(cid, f"Class {cid}")
                print(f"   [{cid:>2}] {cname:<35} {label:<10} {count:>4} images")
                total += count
                if cid in BENIGN_CLASSES:    benign_count    += count
                elif cid in MALIGNANT_CLASSES: malignant_count += count
            print(f"   ─── Total: {total}  |  Benign: {benign_count}  |  Malignant: {malignant_count}")
            stats[split] = {"total": total, "benign": benign_count, "malignant": malignant_count}
        else:
            imgs = list(split_dir.glob("*.*"))
            print(f"\n📁 {split}/ (flat, no class labels):  {len(imgs)} images")
            stats[split] = {"total": len(imgs)}

    # Segmentation dataset
    seg_train_img = list((SEG_DATASET / "train" / "images").glob("*")) if (SEG_DATASET / "train" / "images").exists() else []
    seg_val_img   = list((SEG_DATASET / "val"   / "images").glob("*")) if (SEG_DATASET / "val"   / "images").exists() else []
    print(f"\n📁 seg_dataset/ (for U-Net training):")
    print(f"   train/images:  {len(seg_train_img)}")
    print(f"   val/images:    {len(seg_val_img)}")

    print("\n" + "="*65)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMAGE VALIDATION (clean corrupt / tiny files)
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_image(path: Path, min_size: int = 32) -> bool:
    """Check image can be read and meets minimum dimensions."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= min_size and w >= min_size
    except Exception:
        return False


def clean_images(image_dir: Path, min_size: int = 32) -> Tuple[int, int]:
    """Remove unreadable or too-small images. Returns (kept, removed)."""
    all_imgs = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png")) + \
               list(image_dir.rglob("*.jpeg")) + list(image_dir.rglob("*.bmp"))
    kept = removed = 0
    for p in all_imgs:
        if is_valid_image(p, min_size):
            kept += 1
        else:
            p.unlink(missing_ok=True)
            removed += 1
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD CLASSIFICATION DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def resize_and_save(src: Path, dst: Path, size: Tuple[int,int]) -> bool:
    """Read → resize → normalise display (save as uint8) → write."""
    try:
        img = cv2.imread(str(src))
        if img is None:
            return False
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img_resized)
        return True
    except Exception:
        return False


def build_classification_dataset(splits: List[str] = None,
                                  dry_run: bool = False) -> Dict:
    """
    Build clf_dataset/ with benign/malignant subfolders.
    Each image is resized to CLF_SIZE (150×150) and normalised.

    Directory layout:
        clf_dataset/
            train/ benign/ *.jpg
                   malignant/ *.jpg
            val/   benign/ *.jpg
                   malignant/ *.jpg
            test/  benign/ *.jpg
                   malignant/ *.jpg
    """
    if splits is None:
        splits = ["val2019", "test2019"]   # only these have class labels

    totals: Dict[str, Dict[str, int]] = {}

    for split in splits:
        src_img_root = POLYPS_SET / split / "Image"
        dst_split    = "train" if "train" in split else split.replace("2019","")

        benign_dir    = CLF_DATASET / dst_split / "benign"
        malignant_dir = CLF_DATASET / dst_split / "malignant"

        if not dry_run:
            benign_dir.mkdir(parents=True, exist_ok=True)
            malignant_dir.mkdir(parents=True, exist_ok=True)

        subdirs = sorted([d for d in src_img_root.iterdir() if d.is_dir()],
                         key=lambda x: int(x.name))
        if not subdirs:
            print(f"  ⚠  {split} has no class subfolders – skipping")
            continue

        print(f"\n[INFO] Building clf_dataset/{dst_split}/ from {split}/")
        b_count = m_count = skip_count = 0

        for class_dir in tqdm(subdirs, desc=f"  {split}"):
            cid  = int(class_dir.name)
            if cid in BENIGN_CLASSES:
                dst_dir = benign_dir
                label   = "benign"
            elif cid in MALIGNANT_CLASSES:
                dst_dir = malignant_dir
                label   = "malignant"
            else:
                skip_count += 1
                continue

            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
                    continue
                dst_path = dst_dir / img_path.name
                if not dry_run:
                    ok = resize_and_save(img_path, dst_path, CLF_SIZE)
                    if not ok:
                        skip_count += 1
                        continue
                if label == "benign":    b_count += 1
                else:                    m_count += 1

        totals[dst_split] = {"benign": b_count, "malignant": m_count}
        print(f"  ✅ {dst_split}: {b_count} benign, {m_count} malignant images")

    return totals


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random combination of: rotation, flip, zoom, brightness."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 1. Random rotation ±30°
    angle = random.uniform(-30, 30)
    pil = pil.rotate(angle, expand=False, fillcolor=(0,0,0))

    # 2. Random horizontal/vertical flip
    if random.random() > 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.7:
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random zoom (crop then resize back)
    zoom_factor = random.uniform(0.8, 1.0)
    w, h = pil.size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    left  = random.randint(0, w - new_w)
    top   = random.randint(0, h - new_h)
    pil   = pil.crop((left, top, left + new_w, top + new_h))
    pil   = pil.resize((w, h), Image.BILINEAR)

    # 4. Random brightness adjustment
    enhancer = ImageEnhance.Brightness(pil)
    pil = enhancer.enhance(random.uniform(0.6, 1.4))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def augment_classification_dataset(aug_per_image: int = AUG_PER_IMAGE):
    """Generate augmented copies for the training split of clf_dataset."""
    train_dir = CLF_DATASET / "train"
    if not train_dir.exists():
        print("  ⚠  clf_dataset/train/ not found – run build first")
        return

    for label in ["benign", "malignant"]:
        src_dir = train_dir / label
        if not src_dir.exists():
            continue
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        print(f"\n[INFO] Augmenting clf_dataset/train/{label}/: {len(images)} originals → +{aug_per_image}x")
        for img_path in tqdm(images, desc=f"  Aug {label}"):
            orig = cv2.imread(str(img_path))
            if orig is None:
                continue
            for i in range(aug_per_image):
                aug    = augment_image(orig)
                stem   = img_path.stem
                out    = src_dir / f"{stem}_aug{i}.jpg"
                cv2.imwrite(str(out), aug)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VERIFY SEGMENTATION DATASET (already built by generate_masks.py)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_seg_dataset():
    """Quick integrity check on existing seg_dataset."""
    print("\n[INFO] Verifying seg_dataset/ ...")
    issues = 0
    for split in ["train", "val"]:
        img_dir  = SEG_DATASET / split / "images"
        mask_dir = SEG_DATASET / split / "masks"
        if not img_dir.exists():
            print(f"  ⚠  {split}/images missing")
            continue
        imgs  = sorted(img_dir.glob("*.*"))
        masks = sorted(mask_dir.glob("*.*"))
        print(f"  {split}: {len(imgs)} images, {len(masks)} masks")
        if len(imgs) != len(masks):
            print(f"  ⚠  Count mismatch! Check generate_masks.py")
            issues += 1
        # Spot-check 10 random pairs
        for img_path in random.sample(imgs, min(10, len(imgs))):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                issues += 1
    if issues == 0:
        print("  ✅ seg_dataset looks good!")
    else:
        print(f"  ⚠  {issues} issue(s) found in seg_dataset")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PolypNet Data Preprocessing Pipeline")
    parser.add_argument("--audit",   action="store_true", help="Audit only, no disk writes")
    parser.add_argument("--no-aug",  action="store_true", help="Skip data augmentation")
    parser.add_argument("--dry-run", action="store_true", help="Count files without copying")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  POLYPNET – DATA PREPROCESSING PIPELINE")
    print("="*65)

    # ── Step 1: Audit ──────────────────────────────────────────────────────
    print("\n── STEP 1: DATASET AUDIT ──")
    audit_dataset()

    if args.audit:
        print("\n[Audit-only mode] Exiting without modifications.")
        return

    # ── Step 2: Build Classification Dataset ──────────────────────────────
    print("\n── STEP 2: BUILD CLASSIFICATION DATASET (150×150) ──")
    print("[INFO] Mapping classes 1-10 → Benign, 11-17 → Malignant")
    totals = build_classification_dataset(
        splits=["val2019", "test2019"],
        dry_run=args.dry_run
    )

    # ── Step 3: Validate Images ────────────────────────────────────────────
    if not args.dry_run:
        print("\n── STEP 3: VALIDATE & CLEAN IMAGES ──")
        for split in ["train", "val", "test"]:
            d = CLF_DATASET / split
            if d.exists():
                kept, removed = clean_images(d)
                print(f"  clf_dataset/{split}: kept={kept}, removed_corrupt={removed}")

    # ── Step 4: Data Augmentation ──────────────────────────────────────────
    if not args.no_aug and not args.dry_run:
        print("\n── STEP 4: DATA AUGMENTATION (train split) ──")
        augment_classification_dataset(AUG_PER_IMAGE)

    # ── Step 5: Verify Segmentation Dataset ───────────────────────────────
    print("\n── STEP 5: VERIFY SEGMENTATION DATASET (256×256) ──")
    verify_seg_dataset()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)

    if not args.dry_run:
        for split, counts in totals.items():
            total = counts.get("benign",0) + counts.get("malignant",0)
            if not args.no_aug and split == "train":
                aug_total = total * (1 + AUG_PER_IMAGE)
                print(f"  clf_dataset/{split}: {total} originals → {aug_total} after augmentation")
            else:
                print(f"  clf_dataset/{split}: {counts.get('benign',0)} benign + {counts.get('malignant',0)} malignant = {total}")
        seg_count = len(list((SEG_DATASET/"train"/"images").glob("*"))) if (SEG_DATASET/"train"/"images").exists() else 0
        print(f"  seg_dataset/train: {seg_count} image-mask pairs (256×256 ready)")

    print("\n✅ Preprocessing complete!\n")
    print("  Next steps:")
    print("  1. Train classification model:  python3 train_classifiers.py")
    print("  2. Train validation model:      python3 train_classifiers.py --model validation")
    print("  3. Run web app:                 python3 webapp/app.py")


if __name__ == "__main__":
    main()
