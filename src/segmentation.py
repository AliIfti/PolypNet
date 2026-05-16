"""Attention U-Net for polyp segmentation.

Why Attention U-Net for this dataset:
1. Medical Image Segmentation: U-Net architecture is the gold standard for medical imaging
2. Attention Mechanism: Helps focus on polyp regions while suppressing irrelevant background
3. Skip Connections: Preserves fine-grained details crucial for accurate polyp boundaries
4. Small Dataset Friendly: Works well with limited medical imaging datasets
5. Multi-scale Features: Captures polyps of varying sizes effectively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import cv2
from tqdm import tqdm


class AttentionBlock(nn.Module):
    """Attention gate for focusing on relevant features."""
    
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        Args:
            f_g: Number of feature channels from gating signal (decoder)
            f_l: Number of feature channels from skip connection (encoder)
            f_int: Number of intermediate feature channels
        """
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
        """
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for polyp segmentation.
    
    Architecture:
    - Encoder: 4 levels with max pooling
    - Bottleneck: Deepest feature extraction
    - Decoder: 4 levels with attention gates and upsampling
    - Output: Sigmoid activation for binary segmentation
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(f_g=base_channels * 8, f_l=base_channels * 8, f_int=base_channels * 4)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=base_channels * 4, f_l=base_channels * 4, f_int=base_channels * 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=base_channels * 2, f_l=base_channels * 2, f_int=base_channels)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=base_channels, f_l=base_channels, f_int=base_channels // 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out_conv(d1)
        return torch.sigmoid(out)


class PolypDataset(Dataset):
    """Dataset for loading polyp images and segmentation masks."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False
    ):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            image_size: Target size for images (H, W)
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find all images
        self.image_files = sorted(list(self.image_dir.rglob('*.png')) + 
                                  list(self.image_dir.rglob('*.jpg')))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Find corresponding mask
        rel_path = img_path.relative_to(self.image_dir)
        mask_path = self.mask_dir / rel_path
        
        if not mask_path.exists():
            mask_path = self.mask_dir / rel_path.with_suffix('.png')
        
        mask = Image.open(mask_path).convert('L')
        
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        
        # Normalize image
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss for better segmentation."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU)."""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice coefficient."""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    
    return dice.item()


def train_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'checkpoints'
) -> None:
    """
    Train Attention U-Net for polyp segmentation.
    
    Args:
        model: Attention U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training Attention U-Net on {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Loss: Dice + BCE (0.5 each)")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
            train_dice += calculate_dice(outputs, masks)
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
                val_dice += calculate_dice(outputs, masks)
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice
            }, save_path / 'best_model.pth')
            print(f"  ✅ Best model saved (Val Loss: {val_loss:.4f})")
    
    print(f"\n✅ Training complete! Best model saved to {save_path / 'best_model.pth'}")


def predict_mask(
    model: nn.Module,
    image_path: str,
    device: str = 'cuda',
    image_size: Tuple[int, int] = (256, 256),
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Trained Attention U-Net model
        image_path: Path to input image
        device: Device to run on
        image_size: Size to resize image to
        save_path: Optional path to save predicted mask
        
    Returns:
        Predicted mask as numpy array
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(image_size, Image.BILINEAR)
    
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = (image_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    
    # Predict
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        mask = output.squeeze().cpu().numpy()
    
    # Resize back to original size
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Save if requested
    if save_path:
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(save_path, mask_uint8)
    
    return mask
