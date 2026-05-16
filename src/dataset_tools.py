"""Dataset inspection and YOLO data preparation utilities with XML parsing."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import yaml


def summarize_dataset(root_dir: str, samples_per_dir: int = 3) -> Dict[str, any]:
    """
    Print and return the folder hierarchy with sample files per leaf.
    
    Args:
        root_dir: Path to dataset root (e.g., PolypsSet)
        samples_per_dir: Number of sample files to display per directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")
    
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'splits': {}
    }
    
    print(f"\n{'='*80}")
    print(f"DATASET STRUCTURE: {root_path.name}")
    print(f"{'='*80}\n")
    
    for current_root, dirs, files in os.walk(root_path):
        rel = Path(current_root).relative_to(root_path)
        indent_level = len(rel.parts)
        indent = "  " * indent_level
        folder_name = rel.name if rel.name else root_path.name
        
        # Count images and annotations
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        xml_files = [f for f in files if f.lower().endswith('.xml')]
        
        if image_files or xml_files:
            file_count = len(image_files) if image_files else len(xml_files)
            file_type = "images" if image_files else "annotations"
            print(f"{indent}[DIR] {folder_name}/ ({file_count} {file_type})")
            file_indent = "  " * (indent_level + 1)
            
            sample_files = image_files[:samples_per_dir] if image_files else xml_files[:samples_per_dir]
            for sample in sample_files:
                print(f"{file_indent}|- {sample}")
            
            if file_count > samples_per_dir:
                print(f"{file_indent}|- ... and {file_count - samples_per_dir} more")
            
            # Update stats
            if 'Image' in str(rel):
                stats['total_images'] += len(image_files)
            elif 'Annotation' in str(rel):
                stats['total_annotations'] += len(xml_files)
        else:
            print(f"{indent}[DIR] {folder_name}/")
    
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"{'='*80}\n")
    
    return stats


def parse_xml_annotation(xml_path: str, class_id: Optional[int] = None) -> List[Tuple[int, float, float, float, float, str]]:
    """
    Parse XML annotation file to extract bounding boxes and class information.
    
    Args:
        xml_path: Path to XML annotation file
        class_id: Override class ID (if None, uses folder number from path)
        
    Returns:
        List of tuples: (class_id, center_x, center_y, width, height, object_name)
        Coordinates are normalized to [0, 1]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            raise ValueError(f"No size information in {xml_path}")
        
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # If class_id not provided, try to extract from folder in path
        if class_id is None:
            # Extract from path: .../Annotation/1/file.xml -> class_id = 1
            path_parts = Path(xml_path).parts
            for i, part in enumerate(path_parts):
                if part == 'Annotation' and i + 1 < len(path_parts):
                    try:
                        class_id = int(path_parts[i + 1])
                        break
                    except ValueError:
                        pass
            
            if class_id is None:
                class_id = 0  # Default to 0 if can't determine
        
        # Parse all objects
        bboxes = []
        for obj in root.findall('object'):
            # Get object name (polyp type)
            name_elem = obj.find('name')
            object_name = name_elem.text if name_elem is not None else "unknown"
            
            # Get bounding box
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = ((xmin + xmax) / 2) / img_width
            center_y = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            bboxes.append((class_id, center_x, center_y, width, height, object_name))
        
        return bboxes
        
    except Exception as e:
        print(f"[WARN] Error parsing {xml_path}: {e}")
        return []


def prepare_yolo_data(dataset_root: str, output_dir: str, class_names: List[str] = None, multi_class: bool = True) -> str:
    """
    Prepare YOLO format dataset from XML annotations.
    
    For test2019 and val2019, the folder structure contains class information:
    - Annotation/1/, Annotation/3/, etc. represent different polyp/cancer types
    - The folder number is used as the class ID
    
    Args:
        dataset_root: Root directory containing train2019, test2019, val2019
        output_dir: Output directory for YOLO formatted data
        class_names: List of class names (if None, will auto-generate from folder structure)
        multi_class: If True, use folder numbers as class IDs (for test/val)
        
    Returns:
        Path to the generated dataset.yaml file
    """
    from src.class_mapping import POLYP_CLASSES
    
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    # Create YOLO directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("\n[INFO] Preparing YOLO dataset from XML annotations...")
    if multi_class:
        print("[INFO] Multi-class mode enabled - using folder structure for class labels")
    
    # Collect all class IDs and object names found in dataset
    found_classes = set()
    object_names = {}  # class_id -> set of object names
    
    # Process each split
    split_mapping = {
        'train2019': 'train',
        'val2019': 'val',
        'test2019': 'test'
    }
    
    class_distribution = {}
    
    for source_split, target_split in split_mapping.items():
        source_path = dataset_root / source_split
        if not source_path.exists():
            print(f"[WARN] {source_split} not found, skipping...")
            continue
        
        print(f"\n[INFO] Processing {source_split} -> {target_split}...")
        
        # Find all images and annotations
        image_dir = source_path / 'Image'
        annot_dir = source_path / 'Annotation'
        
        if not image_dir.exists() or not annot_dir.exists():
            print(f"[WARN] Image or Annotation folder missing in {source_split}")
            continue
        
        # Get all XML annotation files recursively
        xml_files = list(annot_dir.rglob('*.xml'))
        
        processed = 0
        skipped = 0
        
        for xml_path in xml_files:
            # Find corresponding image
            rel_path = xml_path.relative_to(annot_dir)
            img_name = xml_path.stem  # filename without extension
            
            # Try to find image with same name
            possible_img_paths = [
                image_dir / rel_path.with_suffix('.png'),
                image_dir / rel_path.with_suffix('.jpg'),
                image_dir / rel_path.with_suffix('.jpeg'),
            ]
            
            img_path = None
            for p in possible_img_paths:
                if p.exists():
                    img_path = p
                    break
            
            if img_path is None:
                skipped += 1
                continue
            
            # Determine class ID from folder structure
            class_id = 0
            if multi_class and len(rel_path.parts) > 1:
                try:
                    # First part should be the class folder number
                    class_id = int(rel_path.parts[0])
                    found_classes.add(class_id)
                except ValueError:
                    class_id = 0
            
            # Parse XML annotation
            bboxes = parse_xml_annotation(str(xml_path), class_id=class_id)
            
            if not bboxes:
                skipped += 1
                continue
            
            # Track object names for this class
            for bbox in bboxes:
                obj_name = bbox[5]  # object name is the 6th element
                if class_id not in object_names:
                    object_names[class_id] = set()
                object_names[class_id].add(obj_name)
            
            # Copy image with unique name to avoid conflicts
            if len(rel_path.parts) > 1:
                # Include class folder in filename to ensure uniqueness
                target_img_name = f"{rel_path.parts[0]}_{img_path.name}"
            else:
                target_img_name = img_path.name
            
            target_img = output_dir / target_split / 'images' / target_img_name
            import shutil
            shutil.copy2(img_path, target_img)
            
            # Generate YOLO label
            target_label = output_dir / target_split / 'labels' / f"{target_img_name.rsplit('.', 1)[0]}.txt"
            
            with open(target_label, 'w') as f:
                for bbox in bboxes:
                    # bbox format: (class_id, center_x, center_y, width, height, object_name)
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            
            # Track class distribution
            class_key = f"{target_split}_class_{class_id}"
            class_distribution[class_key] = class_distribution.get(class_key, 0) + 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed} annotations...")
        
        print(f"[OK] {target_split}: {processed} annotations processed, {skipped} skipped")
    
    # Generate class names
    if class_names is None:
        if multi_class and found_classes:
            max_class = max(found_classes)
            class_names = []
            for i in range(max_class + 1):
                if i in POLYP_CLASSES:
                    class_names.append(POLYP_CLASSES[i])
                elif i in object_names and object_names[i]:
                    # Use the object name from XML if available
                    class_names.append(list(object_names[i])[0])
                else:
                    class_names.append(f"Class_{i}")
        else:
            class_names = ['Polyp']
    
    # Print class distribution and object names
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION AND OBJECT TYPES")
    print(f"{'='*80}")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        split_classes = {k: v for k, v in class_distribution.items() if k.startswith(split)}
        if split_classes:
            for class_key, count in sorted(split_classes.items()):
                class_id = int(class_key.split('_')[-1])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                obj_types = ", ".join(object_names.get(class_id, {"unknown"}))
                print(f"  Class {class_id} ({class_name}): {count} images")
                print(f"    Object types in XML: {obj_types}")
        else:
            print(f"  No data")
    print(f"{'='*80}\n")
    
    # Create dataset.yaml
    yaml_path = output_dir / 'dataset.yaml'
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"[OK] YOLO dataset prepared at: {output_dir}")
    print(f"[OK] Configuration saved to: {yaml_path}")
    print(f"[OK] Number of classes: {len(class_names)}")
    print(f"[OK] Classes found in dataset: {sorted(found_classes)}")
    
    return str(yaml_path)
