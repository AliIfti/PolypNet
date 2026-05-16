"""
PolypNet – Stage 3: Classification Model Training
===================================================
Trains four CNN architectures (VGG16, ResNet50, EfficientNetB0, InceptionV3)
for binary Benign / Malignant classification using transfer learning.

Dataset used:  clf_dataset/
  train/  benign/  malignant/
  val/    benign/  malignant/
  test/   benign/  malignant/

Output per model:
  checkpoints/classification/<ModelName>/best_model.pth
  checkpoints/classification/<ModelName>/final_model.pth
  output/classification/<ModelName>/training_curves.png

Usage:
  # Train all four models
  python3 train_classifiers.py

  # Train a single model
  python3 train_classifiers.py --model efficientnet

  # Quick test with fewer epochs / smaller batch
  python3 train_classifiers.py --epochs 10 --batch 16
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent
CLF_DIR   = BASE / "clf_dataset"
CKPT_DIR  = BASE / "checkpoints" / "classification"
OUT_DIR   = BASE / "output"      / "classification"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (150, 150)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES  = ["benign", "malignant"]   # sorted = alphabetical; PT ImageFolder does this


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_loaders(batch_size: int, inception: bool = False):
    if inception:
        # InceptionV3 requires 299×299 minimum
        train_tf = transforms.Compose([
            transforms.Resize((310, 310)),
            transforms.RandomCrop((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_dir = CLF_DIR / "train"
    val_dir   = CLF_DIR / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"clf_dataset/train/ not found.\n"
            f"Run: python3 preprocess_data.py  (then wait for train-split copy to finish)"
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")
    print(f"  Classes: {train_ds.classes}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS  (all pretrained on ImageNet, head replaced for 2-class)
# ═══════════════════════════════════════════════════════════════════════════════

def _head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )


def build_vgg16() -> nn.Module:
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Freeze all conv layers, fine-tune classifier only
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[6] = nn.Linear(4096, 2)
    return m


def build_resnet50() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze early layers
    for name, p in m.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            p.requires_grad = False
    m.fc = _head(m.fc.in_features)
    return m


def build_efficientnet() -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feat = m.classifier[1].in_features
    m.classifier = _head(in_feat)
    return m


def build_inception() -> nn.Module:
    # InceptionV3 requires 299×299; we'll use a resize in its transform
    m = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,
                             aux_logits=True)
    m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, 2)
    m.fc           = _head(m.fc.in_features)
    return m


MODELS = {
    "vgg16":       (build_vgg16,       "VGG16"),
    "resnet50":    (build_resnet50,    "ResNet50"),
    "efficientnet":(build_efficientnet,"EfficientNetB0"),
    "inception":   (build_inception,   "InceptionV3"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_model(model_key: str, model: nn.Module,
                    train_loader, val_loader,
                    epochs: int, lr: float,
                    is_inception: bool = False):
    _, display_name = MODELS[model_key]

    ckpt_folder = CKPT_DIR / display_name
    out_folder  = OUT_DIR  / display_name
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    out_folder.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    early_stop_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training {display_name}  |  Device: {DEVICE}  |  Epochs: {epochs}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_loss = t_correct = t_total = 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                                  leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if is_inception:
                out, aux = model(imgs)
                loss = criterion(out, labels) + 0.4 * criterion(aux, labels)
            else:
                out  = model(imgs)
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                      desc=f"Epoch {epoch:02d}/{epochs} [Val]  ",
                                      leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        t_l   = t_loss / t_total
        v_l   = v_loss / v_total

        scheduler.step(v_acc)
        history["train_loss"].append(t_l)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_l)
        history["val_acc"].append(v_acc)

        print(f"  [{display_name}] Epoch {epoch:02d}/{epochs}  "
              f"Train Loss={t_l:.4f} Acc={t_acc*100:.1f}%  |  "
              f"Val Loss={v_l:.4f} Acc={v_acc*100:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "classes": CLASSES,
            }, ckpt_folder / "best_model.pth")
            print(f"  💾 Best saved ({v_acc*100:.1f}%)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 15:
                print(f"  Early stopping at epoch {epoch} - no improvement for 15 epochs")
                break

    # Save final
    torch.save(model.state_dict(), ckpt_folder / "final_model.pth")

    # ── Plot ──────────────────────────────────────────────────────────────
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{display_name} – Training Curves", fontsize=14, fontweight="bold")
    ax1.plot(ep, history["train_loss"], label="Train Loss")
    ax1.plot(ep, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(ep, [a*100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(ep, [a*100 for a in history["val_acc"]],   label="Val Acc")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    ax2.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_folder / "training_curves.png", dpi=150)
    plt.close()

    print(f"\n  ✅ {display_name} done — best val acc: {best_val_acc*100:.1f}%")
    print(f"     Saved: {ckpt_folder}/best_model.pth")
    return best_val_acc


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PolypNet Classification Trainer")
    parser.add_argument("--model",  type=str, default="all",
                        choices=["all", "vgg16", "resnet50",
                                 "efficientnet", "inception"],
                        help="Which model to train (default: all)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per model (default: 20)")
    parser.add_argument("--batch",  type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",     type=float, default=1e-4,
                        help="Learning rate (default: 0.0001)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PolypNet – Stage 3: Classification Training")
    print("="*60)
    print(f"  Device  : {DEVICE}")
    print(f"  Model(s): {args.model}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")

    train_loader, val_loader = get_loaders(args.batch)

    to_train = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}

    for key in to_train:
        build_fn, display_name = MODELS[key]
        model = build_fn()
        is_inception = (key == "inception")
        # Use 299×299 loaders for InceptionV3, standard loaders for others
        if is_inception:
            t_loader, v_loader = get_loaders(args.batch, inception=True)
        else:
            t_loader, v_loader = train_loader, val_loader
        acc = train_one_model(
            key, model, t_loader, v_loader,
            args.epochs, args.lr, is_inception
        )
        results[display_name] = acc

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name:<20}: {acc*100:.1f}% val accuracy")

    print("\n  Models saved in:  checkpoints/classification/")
    print("  Curves saved in:  output/classification/")
    print("\n  Next: python3 webapp/app.py")


if __name__ == "__main__":
    main()
