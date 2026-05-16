"""
PolypNet – Download Real Negative Images for Validation Training
================================================================
Downloads two datasets and saves them as JPEG files in negatives/real/

  1. CIFAR-10  (~170 MB) – natural photos (cars, birds, cats, ships…)
  2. LFW       (~200 MB) – real human face photos (handles the selfie case)

Run ONCE:
  python3 prepare_negatives.py

Output:
  negatives/real/   ← ~4000 real non-colonoscopy images at 150×150

Time: ~5-8 minutes (downloads ~370 MB total)
"""

from pathlib import Path
import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
from PIL import Image as PILImage
from tqdm import tqdm

OUT_DIR  = Path(__file__).parent / "negatives" / "real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESIZE   = T.Resize((150, 150))
N_CIFAR  = 2000   # natural scene images
N_LFW    = 2000   # real face photos


def save_cifar(data_dir: Path):
    """Save N_CIFAR images from CIFAR-10."""
    out_count = len(list(OUT_DIR.glob("cifar_*.jpg")))
    if out_count >= N_CIFAR:
        print(f"  ✅ CIFAR-10: already have {out_count} images – skipping.")
        return

    print("\n  [1/2] Downloading CIFAR-10 (~170 MB)…")
    ds = dsets.CIFAR10(root=str(data_dir / "cifar10"), train=True, download=True)

    print(f"  Saving {N_CIFAR} CIFAR-10 images…")
    for i, (img, _) in enumerate(tqdm(ds, total=N_CIFAR, desc="  CIFAR-10")):
        if i >= N_CIFAR:
            break
        RESIZE(img).save(OUT_DIR / f"cifar_{i:05d}.jpg", quality=95)
    print(f"  ✅ Saved {N_CIFAR} CIFAR-10 images")


def save_lfw(data_dir: Path):
    """Save N_LFW face images from LFW People dataset."""
    out_count = len(list(OUT_DIR.glob("lfw_*.jpg")))
    if out_count >= N_LFW:
        print(f"  ✅ LFW faces: already have {out_count} images – skipping.")
        return

    print("\n  [2/2] Downloading LFW face photos (~200 MB)…")
    try:
        ds = dsets.LFWPeople(
            root=str(data_dir / "lfw"),
            split="train",
            image_set="original",   # full 250×250 colour faces
            download=True,
        )
    except Exception as e:
        print(f"  ⚠  LFW download failed ({e}); trying DeepFunneled variant…")
        ds = dsets.LFWPeople(
            root=str(data_dir / "lfw"),
            split="train",
            image_set="deepfunneled",
            download=True,
        )

    n_save = min(N_LFW, len(ds))
    print(f"  Saving {n_save} LFW face images…")
    for i in tqdm(range(n_save), desc="  LFW faces"):
        img, _ = ds[i]
        RESIZE(img).save(OUT_DIR / f"lfw_{i:05d}.jpg", quality=95)
    print(f"  ✅ Saved {n_save} LFW face images")


def main():
    print("\n" + "="*60)
    print("  PolypNet – Preparing Real Non-Colonoscopy Negatives")
    print("  Sources: CIFAR-10 (scenes)  +  LFW (human faces)")
    print("="*60)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    save_cifar(data_dir)
    save_lfw(data_dir)

    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\n  ✅ Total negatives in negatives/real/: {total} images")
    print(f"\n  Now retrain the validation model:")
    print(f"    python3 train_validation_model.py --neg-dir negatives/real/ \\")
    print(f"            --epochs 25 --batch 32 --samples 3000\n")


if __name__ == "__main__":
    main()
