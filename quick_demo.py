"""
Quick Demo Script - Test the pipeline on sample data

This script demonstrates the pipeline capabilities without full training.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_tools import summarize_dataset, parse_xml_annotation
from src.image_processing import apply_clahe, compare_enhancement
from src.segmentation import AttentionUNet, predict_mask
from src.vlm import LlavaGenerator


def demo_dataset_inspection():
    """Demo: Inspect dataset structure."""
    print("\n" + "="*80)
    print("DEMO 1: DATASET INSPECTION")
    print("="*80)
    
    dataset_root = "PolypsSet"
    
    if not Path(dataset_root).exists():
        print(f"[FAIL] Dataset not found at: {dataset_root}")
        print("Please ensure the dataset is in the correct location.")
        return False
    
    stats = summarize_dataset(dataset_root, samples_per_dir=2)
    return True


def demo_clahe_enhancement():
    """Demo: CLAHE enhancement on a sample image."""
    print("\n" + "="*80)
    print("DEMO 2: CLAHE ENHANCEMENT")
    print("="*80)
    
    # Find a sample image
    dataset_root = Path("PolypsSet")
    sample_images = list((dataset_root / "train2019" / "Image").rglob("*.png"))
    
    if not sample_images:
        sample_images = list((dataset_root / "train2019" / "Image").rglob("*.jpg"))
    
    if not sample_images:
        print("[FAIL] No sample images found")
        return False
    
    # Take first image
    sample_img = sample_images[0]
    print(f"\nProcessing: {sample_img.name}")
    
    # Create output directory
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply CLAHE
    enhanced_path = output_dir / f"enhanced_{sample_img.name}"
    enhanced = apply_clahe(str(sample_img), output_path=str(enhanced_path))
    
    print(f"[OK] Enhanced image saved to: {enhanced_path}")
    
    # Create comparison
    comparison_path = output_dir / "comparison.png"
    compare_enhancement(str(sample_img), str(enhanced_path), str(comparison_path))
    
    print(f"[OK] Comparison saved to: {comparison_path}")
    
    return True


def demo_xml_parsing():
    """Demo: Parse XML annotations."""
    print("\n" + "="*80)
    print("DEMO 3: XML ANNOTATION PARSING")
    print("="*80)
    
    # Find a sample XML annotation
    dataset_root = Path("PolypsSet")
    sample_xmls = list((dataset_root / "test2019" / "Annotation").rglob("*.xml"))
    
    if not sample_xmls:
        # Try val2019
        sample_xmls = list((dataset_root / "val2019" / "Annotation").rglob("*.xml"))
    
    if not sample_xmls:
        print("[WARN] No sample XML annotations found")
        return False
    
    sample_xml = sample_xmls[0]
    print(f"\nProcessing: {sample_xml}")
    
    # Parse XML
    bboxes = parse_xml_annotation(str(sample_xml))
    
    print(f"\n[OK] Parsed {len(bboxes)} object(s) from XML")
    
    for i, bbox in enumerate(bboxes, 1):
        class_id, cx, cy, w, h, obj_name = bbox
        print(f"\nObject {i}:")
        print(f"  Class ID: {class_id}")
        print(f"  Object name: {obj_name}")
        print(f"  Center (YOLO): ({cx:.3f}, {cy:.3f})")
        print(f"  Size (YOLO): {w:.3f} x {h:.3f}")
    
    # Visualize if corresponding image exists
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find corresponding image
    # sample_xml is like: PolypsSet/PolypsSet/test2019/Annotation/1/1.xml
    # We need to find: PolypsSet/PolypsSet/test2019/Image/1/1.jpg
    
    # Get the split (test2019, val2019, etc.)
    xml_parts = sample_xml.parts
    split_idx = None
    for i, part in enumerate(xml_parts):
        if 'test2019' in part or 'val2019' in part or 'train2019' in part:
            split_idx = i
            break
    
    if split_idx is not None:
        # Construct image directory path
        split_name = xml_parts[split_idx]
        rel_path = sample_xml.relative_to(dataset_root / split_name / "Annotation")
        img_dir = dataset_root / split_name / "Image"
        
        possible_img_paths = [
            img_dir / rel_path.with_suffix('.png'),
            img_dir / rel_path.with_suffix('.jpg'),
            img_dir / rel_path.with_suffix('.jpeg'),
        ]
        
        img_path = None
        for p in possible_img_paths:
            if p.exists():
                img_path = p
                break
    else:
        img_path = None
    
    if img_path:
        # Load image
        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]
        
        # Draw bboxes
        for bbox in bboxes:
            class_id, cx, cy, w, h, obj_name = bbox
            
            # Convert from YOLO format to pixel coordinates
            x1 = int((cx - w/2) * w_img)
            y1 = int((cy - h/2) * h_img)
            x2 = int((cx + w/2) * w_img)
            y2 = int((cy + h/2) * h_img)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{obj_name} (C{class_id})", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        bbox_vis_path = output_dir / "xml_bbox_visualization.png"
        cv2.imwrite(str(bbox_vis_path), img)
        
        print(f"\n[OK] Bounding box visualization saved to: {bbox_vis_path}")
    
    return True


def demo_model_inference():
    """Demo: Model instantiation and dummy inference."""
    print("\n" + "="*80)
    print("DEMO 4: MODEL INFERENCE (Untrained)")
    print("="*80)
    
    print("\nInstantiating Attention U-Net...")
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[OK] Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test inference on dummy data
    print("\nRunning dummy inference...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"[OK] Inference successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return True


def demo_clinical_description():
    """Demo: Generate clinical description (fallback mode)."""
    print("\n" + "="*80)
    print("DEMO 5: CLINICAL DESCRIPTION GENERATION")
    print("="*80)
    
    # Find sample image and XML
    dataset_root = Path("PolypsSet")
    sample_images = list((dataset_root / "test2019" / "Image").rglob("*.jpg"))[:1]
    if not sample_images:
        sample_images = list((dataset_root / "test2019" / "Image").rglob("*.png"))[:1]
    if not sample_images:
        sample_images = list((dataset_root / "val2019" / "Image").rglob("*.jpg"))[:1]
    if not sample_images:
        sample_images = list((dataset_root / "val2019" / "Image").rglob("*.png"))[:1]
    
    # For demo, create a dummy mask (since we don't have actual mask images)
    if not sample_images:
        print("[WARN] No sample images found, using fallback")
        return False
    
    sample_img = sample_images[0]
    # Create a dummy mask for demo purposes
    img = cv2.imread(str(sample_img))
    sample_mask_path = Path("output/demo/dummy_mask.png")
    sample_mask_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = dummy_mask.shape
    dummy_mask[h//4:3*h//4, w//4:3*w//4] = 255
    cv2.imwrite(str(sample_mask_path), dummy_mask)
    sample_mask = sample_mask_path
    
    print(f"\nImage: {sample_img.name}")
    print(f"Mask: {sample_mask.name}")
    
    # Initialize LLaVA generator (will use fallback if model not available)
    llava = LlavaGenerator()
    
    # Generate description
    print("\nGenerating clinical description...")
    description = llava._generate_fallback_description(str(sample_img), str(sample_mask))
    
    # Save description
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    desc_path = output_dir / "clinical_description.txt"
    with open(desc_path, 'w') as f:
        f.write(f"Image: {sample_img}\n")
        f.write(f"Mask: {sample_mask}\n")
        f.write(f"\n{'='*80}\n")
        f.write(description)
    
    print(f"\n{'='*80}")
    print(description)
    print('='*80)
    
    print(f"\n[OK] Description saved to: {desc_path}")
    
    # Create overlay
    overlay_path = output_dir / "overlay.png"
    llava.create_overlay(str(sample_img), str(sample_mask), str(overlay_path))
    print(f"[OK] Overlay image saved to: {overlay_path}")
    
    return True


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("POLYP DETECTION PIPELINE - QUICK DEMO")
    print("="*80)
    print("\nThis demo will showcase the pipeline capabilities without training.")
    print("All outputs will be saved to: output/demo/")
    
    demos = [
        ("Dataset Inspection", demo_dataset_inspection),
        ("CLAHE Enhancement", demo_clahe_enhancement),
        ("XML Annotation Parsing", demo_xml_parsing),
        ("Model Inference", demo_model_inference),
      ##("Clinical Description", demo_clinical_description)
    ]
    
    results = {}
    
    for name, demo_func in demos:
        try:
            results[name] = demo_func()
        except Exception as e:
            print(f"\n[FAIL] Demo failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    for name, success in results.items():
        status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
        print(f"{status}: {name}")
    
    if all(results.values()):
        print("\n[OK] All demos completed successfully!")
        print("\nNext steps:")
        print("1. Review outputs in: output/demo/")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run verification: python verify_setup.py")
        print("4. Start training: python main_pipeline.py --mode train_seg --epochs 100")
    else:
        print("\n[WARN]  Some demos failed. Please check the errors above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

