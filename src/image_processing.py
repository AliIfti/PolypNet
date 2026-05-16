"""Image enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


def apply_clahe(
    image_path: str,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Apply CLAHE enhancement to improve contrast and visibility of polyps.
    
    CLAHE works in LAB color space to enhance the luminance channel while
    preserving color information, which is crucial for medical imaging.
    
    Args:
        image_path: Path to input image
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid: Size of grid for histogram equalization
        output_path: Optional path to save enhanced image
        
    Returns:
        PIL Image object of enhanced image
    """
    # Read image
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Convert to LAB color space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    merged = cv2.merge((l_enhanced, a, b))
    
    # Convert back to BGR then RGB
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(enhanced_rgb)
    
    # Save if output path provided
    if output_path:
        pil_image.save(output_path)
    
    return pil_image


def enhance_dataset(
    src_dir: str,
    dst_dir: str,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8)
) -> int:
    """
    Apply CLAHE enhancement to all images in a dataset while preserving structure.
    
    Args:
        src_dir: Source directory containing images
        dst_dir: Destination directory for enhanced images
        clip_limit: CLAHE clip limit parameter
        tile_grid: CLAHE tile grid size
        
    Returns:
        Number of images processed
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")
    
    print(f"\n[INFO] Enhancing images with CLAHE...")
    print(f"   Clip Limit: {clip_limit}")
    print(f"   Tile Grid: {tile_grid}")
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [
        f for f in src_path.rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    processed = 0
    for img_path in image_files:
        # Preserve directory structure
        rel_path = img_path.relative_to(src_path)
        out_path = dst_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            apply_clahe(
                str(img_path),
                clip_limit=clip_limit,
                tile_grid=tile_grid,
                output_path=str(out_path)
            )
            processed += 1
            
            if processed % 100 == 0:
                print(f"   Processed {processed}/{len(image_files)} images...")
        except Exception as e:
            print(f"[WARN] Error processing {img_path}: {e}")
    
    print(f"[OK] Enhanced {processed} images saved to {dst_path}")
    return processed


def compare_enhancement(original_path: str, enhanced_path: str, output_path: str) -> None:
    """
    Create a side-by-side comparison of original and enhanced images.
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to enhanced image
        output_path: Path to save comparison image
    """
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)
    
    if original is None or enhanced is None:
        raise ValueError("Unable to read one or both images")
    
    # Resize if needed to match heights
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    if h1 != h2:
        enhanced = cv2.resize(enhanced, (w1, h1))
    
    # Concatenate horizontally
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'CLAHE Enhanced', (w1 + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"[OK] Comparison saved to {output_path}")
