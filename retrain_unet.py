"""
PolypNet – Retrain Attention U-Net with Combined BCE + Dice Loss
================================================================
Loss:   0.5 * BCE + 0.5 * Dice
Dice:   1 - (2*sum(pred*target)+1) / (sum(pred)+sum(target)+1)

Training:
  - 150 epochs max, early stopping patience 15 (on val Dice)
  - Save best checkpoint by highest validation Dice
  - Print metrics every 10 epochs

Post-training:
  - Plot & save training curves (loss + Dice)
  - Threshold sweep 0.50–0.70
  - Report Dice, IoU, Precision, Recall at each threshold
  - Save everything to results/unet_retrain_eval.txt

Usage:
    python3 retrain_unet.py
"""

import sys, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.segmentation import AttentionUNet, PolypDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SEG_DATASET   = BASE / "seg_dataset"
SAVE_DIR      = BASE / "checkpoints" / "segmentation"
RESULTS_DIR   = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH   = RESULTS_DIR / "unet_retrain_eval.txt"
CURVES_PATH   = RESULTS_DIR / "unet_retrain_curves.png"

# ── Hyperparameters (same as original) ────────────────────────────────────────
IMAGE_SIZE    = (256, 256)
BATCH_SIZE    = 8
LR            = 1e-4
MAX_EPOCHS    = 150
PATIENCE      = 15            # early stopping on val Dice
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Report logging ────────────────────────────────────────────────────────────
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Dice Loss as specified:
       dice_loss = 1 - (2*sum(pred*target)+1) / (sum(pred)+sum(target)+1)
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat   = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined Loss = 0.5 * BCE + 0.5 * Dice"""
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (soft / differentiable versions not needed here — we use hard metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Return Dice, IoU, Precision, Recall for a batch."""
    pred_bin   = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    pred_flat   = pred_bin.contiguous().view(-1)
    target_flat = target_bin.contiguous().view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    return dice, iou, precision, recall


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    log("=" * 65)
    log("  PolypNet – Retrain Attention U-Net (BCE + Dice)")
    log(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = PolypDataset(
        str(SEG_DATASET / "train" / "images"),
        str(SEG_DATASET / "train" / "masks"),
        image_size=IMAGE_SIZE, augment=True,
    )
    val_ds = PolypDataset(
        str(SEG_DATASET / "val" / "images"),
        str(SEG_DATASET / "val" / "masks"),
        image_size=IMAGE_SIZE, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    log(f"\n  Train images : {len(train_ds)}")
    log(f"  Val images   : {len(val_ds)}")
    log(f"  Batch size   : {BATCH_SIZE}")
    log(f"  Image size   : {IMAGE_SIZE}")
    log(f"  LR           : {LR}")
    log(f"  Max epochs   : {MAX_EPOCHS}")
    log(f"  Patience     : {PATIENCE}")
    log(f"  Device       : {DEVICE}")
    log(f"  Loss         : 0.5*BCE + 0.5*Dice")

    # ── Model / Optimizer / Loss ──────────────────────────────────────────
    device = torch.device(DEVICE)
    model  = AttentionUNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    criterion = CombinedLoss()

    best_val_dice  = 0.0
    patience_ctr   = 0

    # History
    hist_train_loss = []
    hist_val_loss   = []
    hist_val_dice   = []
    hist_val_iou    = []

    log(f"\n🚀  Training started …\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        hist_train_loss.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum  = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss_sum += criterion(outputs, masks).item()
                d, i, _, _ = compute_metrics(outputs, masks)
                val_dice_sum += d
                val_iou_sum  += i

        val_loss = val_loss_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_iou  = val_iou_sum  / len(val_loader)

        hist_val_loss.append(val_loss)
        hist_val_dice.append(val_dice)
        hist_val_iou.append(val_iou)

        scheduler.step(val_dice)

        # ── Early stopping / checkpoint ───────────────────────────────────
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_ctr  = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou":  val_iou,
            }, SAVE_DIR / "best_model.pth")
            marker = "  ✅ saved"
        else:
            patience_ctr += 1
            marker = ""

        # ── Print every 10 epochs ─────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1 or patience_ctr >= PATIENCE:
            log(f"Epoch {epoch:>3}/{MAX_EPOCHS}  |  "
                f"Train Loss: {train_loss:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  |  "
                f"Val Dice: {val_dice:.4f}  |  "
                f"Val IoU: {val_iou:.4f}{marker}")

        if patience_ctr >= PATIENCE:
            log(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    log(f"\n✅  Training complete.  Best Val Dice = {best_val_dice:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Post-training: Training curves
    # ═══════════════════════════════════════════════════════════════════════
    log(f"\n📊  Saving training curves → {CURVES_PATH}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(hist_train_loss) + 1)

    ax1.plot(epochs_range, hist_train_loss, label="Train Loss", linewidth=1.5)
    ax1.plot(epochs_range, hist_val_loss,   label="Val Loss",   linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, hist_val_dice, label="Val Dice", linewidth=1.5, color="green")
    ax2.plot(epochs_range, hist_val_iou,  label="Val IoU",  linewidth=1.5, color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Dice & IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(CURVES_PATH), dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Post-training: Threshold sweep on validation set
    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 65)
    log("  THRESHOLD SWEEP (0.50 → 0.70)")
    log("=" * 65)

    # Reload best model
    ckpt = torch.load(str(SAVE_DIR / "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    thresholds = np.arange(0.50, 0.71, 0.01)
    best_thresh = 0.5
    best_sweep_dice = 0.0
    best_metrics = {}

    log(f"\n  {'Thresh':>7}  {'Dice':>8}  {'IoU':>8}  {'Precision':>10}  {'Recall':>8}")
    log("  " + "─" * 50)

    for thresh in thresholds:
        all_dice, all_iou, all_prec, all_rec = [], [], [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                d, i, p, r = compute_metrics(outputs, masks, threshold=thresh)
                all_dice.append(d)
                all_iou.append(i)
                all_prec.append(p)
                all_rec.append(r)

        m_dice = np.mean(all_dice)
        m_iou  = np.mean(all_iou)
        m_prec = np.mean(all_prec)
        m_rec  = np.mean(all_rec)

        log(f"  {thresh:>7.2f}  {m_dice:>8.4f}  {m_iou:>8.4f}  {m_prec:>10.4f}  {m_rec:>8.4f}")

        if m_dice > best_sweep_dice:
            best_sweep_dice = m_dice
            best_thresh = thresh
            best_metrics = {
                "dice": m_dice, "iou": m_iou,
                "precision": m_prec, "recall": m_rec,
            }

    log("\n" + "─" * 65)
    log(f"  🏆  Optimal threshold : {best_thresh:.2f}")
    log(f"       Dice             : {best_metrics['dice']:.4f}")
    log(f"       IoU              : {best_metrics['iou']:.4f}")
    log(f"       Precision        : {best_metrics['precision']:.4f}")
    log(f"       Recall           : {best_metrics['recall']:.4f}")
    log("─" * 65)

    # ═══════════════════════════════════════════════════════════════════════
    # Save full report
    # ═══════════════════════════════════════════════════════════════════════
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"\n✅  Full report saved → {REPORT_PATH}")


if __name__ == "__main__":
    train()
