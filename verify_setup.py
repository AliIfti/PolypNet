"""
Verification Script for Polyp Detection Pipeline

This script verifies:
1. Dataset structure and accessibility
2. CLAHE enhancement functionality
3. YOLO data preparation
4. Model instantiation
5. Dependencies installation
"""

import sys
from pathlib import Path
import numpy as np


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'torch': 'torch',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }
    
    optional = {
        'ultralytics': 'ultralytics (for YOLO)',
        'transformers': 'transformers (for LLaVA)'
    }
    
    missing = []
    
    print("\nRequired Dependencies:")
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package}")
            missing.append(package)
    
    print("\nOptional Dependencies:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [WARN] {package} (optional)")
    
    if missing:
        print(f"\n[FAIL] Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All required dependencies installed!")
        return True


def check_dataset(dataset_root: str):
    """Verify dataset structure."""
    print("\n" + "="*80)
    print("CHECKING DATASET STRUCTURE")
    print("="*80)
    
    root = Path(dataset_root)
    
    if not root.exists():
        print(f"[FAIL] Dataset root not found: {root}")
        return False
    
    required_splits = ['train2019', 'test2019', 'val2019']
    all_good = True
    
    for split in required_splits:
        split_path = root / split
        if not split_path.exists():
            print(f"❌ Missing split: {split}")
            all_good = False
            continue
        
        image_dir = split_path / 'Image'
        mask_dir = split_path / 'Annotation'
        
        if not image_dir.exists():
            print(f"[FAIL] Missing Image directory in {split}")
            all_good = False
        else:
            image_count = len(list(image_dir.rglob('*.png')) + list(image_dir.rglob('*.jpg')))
            print(f"[OK] {split}/Image: {image_count} images")
        
        if not mask_dir.exists():
            print(f"[FAIL] Missing Annotation directory in {split}")
            all_good = False
        else:
            xml_count = len(list(mask_dir.rglob('*.xml')))
            print(f"[OK] {split}/Annotation: {xml_count} XML annotations")
    
    return all_good


def test_clahe():
    """Test CLAHE enhancement."""
    print("\n" + "="*80)
    print("TESTING CLAHE ENHANCEMENT")
    print("="*80)
    
    try:
        from src.image_processing import apply_clahe
        import cv2
        
        # Create a test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite('test_image.png', test_img)
        
        # Apply CLAHE
        enhanced = apply_clahe('test_image.png')
        
        # Check if output is different
        enhanced_np = np.array(enhanced)
        if not np.array_equal(test_img, enhanced_np):
            print("[OK] CLAHE enhancement working correctly")
            result = True
        else:
            print("[WARN] CLAHE output identical to input (unexpected)")
            result = False
        
        # Cleanup
        Path('test_image.png').unlink()
        
        return result
        
    except Exception as e:
        print(f"[FAIL] CLAHE test failed: {e}")
        return False


def test_xml_parsing():
    """Test XML annotation parsing."""
    print("\n" + "="*80)
    print("TESTING XML ANNOTATION PARSING")
    print("="*80)
    
    try:
        from src.dataset_tools import parse_xml_annotation
        import xml.etree.ElementTree as ET
        
        # Create a test XML annotation
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>1</folder>
    <filename>test.png</filename>
    <path>/test/1/test.png</path>
    <source>
        <database>Test</database>
    </source>
    <size>
        <width>384</width>
        <height>288</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>adenomatous</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>"""
        
        # Write test XML file
        with open('test_annotation.xml', 'w') as f:
            f.write(xml_content)
        
        # Parse XML
        bboxes = parse_xml_annotation('test_annotation.xml', class_id=1)
        
        if len(bboxes) > 0:
            print(f"[OK] Parsed {len(bboxes)} object(s) from XML")
            bbox = bboxes[0]
            print(f"   Class ID: {bbox[0]}")
            print(f"   BBox (YOLO format): cx={bbox[1]:.3f}, cy={bbox[2]:.3f}, w={bbox[3]:.3f}, h={bbox[4]:.3f}")
            print(f"   Object name: {bbox[5]}")
            result = True
        else:
            print("[FAIL] No objects parsed from XML")
            result = False
        
        # Cleanup
        Path('test_annotation.xml').unlink()
        
        return result
        
    except Exception as e:
        print(f"[FAIL] XML parsing test failed: {e}")
        # Cleanup on error
        if Path('test_annotation.xml').exists():
            Path('test_annotation.xml').unlink()
        return False



def test_model_instantiation():
    """Test if models can be instantiated."""
    print("\n" + "="*80)
    print("TESTING MODEL INSTANTIATION")
    print("="*80)
    
    try:
        from src.segmentation import AttentionUNet
        import torch
        
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=32)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        
        if output.shape == (1, 1, 256, 256):
            print(f"[OK] Attention U-Net instantiated successfully")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            return True
        else:
            print(f"[FAIL] Unexpected output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Model instantiation failed: {e}")
        return False


def test_yolo_availability():
    """Test if YOLO is available."""
    print("\n" + "="*80)
    print("TESTING YOLO AVAILABILITY")
    print("="*80)
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO is available")
        print("   You can train YOLOv11 models")
        return True
    except ImportError:
        print("[WARN] Ultralytics not installed")
        print("   Install with: pip install ultralytics")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("POLYP DETECTION PIPELINE - VERIFICATION SCRIPT")
    print("="*80)
    
    results = {}
    
    # Test 1: Dependencies
    results['dependencies'] = check_dependencies()
    
    # Test 2: Dataset
    dataset_root = "PolypsSet"
    results['dataset'] = check_dataset(dataset_root)
    
    # Test 3: CLAHE
    results['clahe'] = test_clahe()
    
    # Test 4: XML Parsing
    results['xml_parsing'] = test_xml_parsing()
    
    # Test 5: Model
    results['model'] = test_model_instantiation()
    
    # Test 6: YOLO
    results['yolo'] = test_yolo_availability()
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_critical_passed = all([
        results['dependencies'],
        results['clahe'],
        results['xml_parsing'],
        results['model']
    ])
    
    if all_critical_passed:
        print("\n[OK] All critical tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Inspect dataset: python main_pipeline.py --mode inspect")
        print("2. Test enhancement: python main_pipeline.py --mode enhance")
        print("3. Prepare YOLO data: python main_pipeline.py --mode prepare_yolo")
        return 0
    else:
        print("\n[FAIL] Some critical tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
