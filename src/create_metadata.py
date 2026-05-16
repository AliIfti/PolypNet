"""
Utility to create dataset metadata and class information documentation.
"""

import json
from pathlib import Path
from typing import Dict
from src.class_mapping import POLYP_CLASSES, LOCATION_CLASSES


def create_dataset_metadata(dataset_root: str, output_file: str = "dataset_metadata.json") -> None:
    """
    Create a metadata file documenting the dataset structure and class information.
    
    Args:
        dataset_root: Root directory of the dataset
        output_file: Output JSON file path
    """
    metadata = {
        "dataset_name": "PolypsSet",
        "description": "Polyp and colorectal cancer detection dataset",
        "structure": {
            "train2019": {
                "description": "Training split - single class (all polyps)",
                "structure": "Image/ and Annotation/ folders without subfolders",
                "class_id": 0,
                "class_name": "Polyp (All Types)"
            },
            "test2019": {
                "description": "Test split - multi-class (polyp/cancer types)",
                "structure": "Image/[class_id]/ and Annotation/[class_id]/ subfolders",
                "classes": "Numbered folders (1, 3, 4, 5, ..., 24) represent different types"
            },
            "val2019": {
                "description": "Validation split - multi-class (polyp/cancer types)",
                "structure": "Image/[class_id]/ and Annotation/[class_id]/ subfolders",
                "classes": "Numbered folders (1-17) represent different types"
            }
        },
        "class_mapping": {
            "type": "polyp_and_cancer_types",
            "classes": POLYP_CLASSES
        },
        "location_mapping": {
            "type": "anatomical_location",
            "classes": LOCATION_CLASSES
        },
        "notes": [
            "Class IDs are extracted from folder structure in test/val splits",
            "Training data uses class_id=0 (generic polyp)",
            "Test/val data uses folder numbers as class IDs",
            "Some class IDs may be missing (e.g., no folder '2' in test2019)"
        ]
    }
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Dataset metadata saved to: {output_path}")
    
    # Also create a human-readable text version
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POLYPSSET DATASET - CLASS INFORMATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET STRUCTURE:\n")
        f.write("-"*80 + "\n")
        f.write("train2019/: Single class training data (all polyps labeled as class 0)\n")
        f.write("test2019/:  Multi-class test data (folders 1, 3, 4, ..., 24)\n")
        f.write("val2019/:   Multi-class validation data (folders 1-17)\n\n")
        
        f.write("POLYP/CANCER TYPE CLASSES:\n")
        f.write("-"*80 + "\n")
        for class_id, class_name in sorted(POLYP_CLASSES.items()):
            f.write(f"Class {class_id:2d}: {class_name}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANATOMICAL LOCATION CLASSES (if applicable):\n")
        f.write("="*80 + "\n")
        for loc_id, loc_name in sorted(LOCATION_CLASSES.items()):
            f.write(f"Location {loc_id:2d}: {loc_name}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTES:\n")
        f.write("-"*80 + "\n")
        f.write("- Folder numbers in test/val correspond to class IDs\n")
        f.write("- Not all class IDs may be present in the dataset\n")
        f.write("- Training data is single-class for initial polyp detection\n")
        f.write("- Test/val data enables multi-class classification\n")
        f.write("="*80 + "\n")
    
    print(f"[OK] Human-readable class info saved to: {txt_path}")


def analyze_dataset_classes(dataset_root: str) -> Dict:
    """
    Analyze the dataset to find which classes are actually present.
    
    Args:
        dataset_root: Root directory of the dataset
        
    Returns:
        Dictionary with class distribution information
    """
    from pathlib import Path
    
    dataset_path = Path(dataset_root)
    analysis = {
        'train': {'classes': set(), 'total_images': 0},
        'test': {'classes': set(), 'total_images': 0},
        'val': {'classes': set(), 'total_images': 0}
    }
    
    # Analyze each split
    for split_name, split_key in [('train2019', 'train'), ('test2019', 'test'), ('val2019', 'val')]:
        split_path = dataset_path / split_name / 'Image'
        
        if not split_path.exists():
            continue
        
        # Check for class folders
        for item in split_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                class_id = int(item.name)
                analysis[split_key]['classes'].add(class_id)
                
                # Count images in this class
                image_count = len(list(item.glob('*.png'))) + len(list(item.glob('*.jpg')))
                analysis[split_key]['total_images'] += image_count
            elif item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Direct images (train split)
                analysis[split_key]['classes'].add(0)
                analysis[split_key]['total_images'] += 1
    
    # Print analysis
    print("\n" + "="*80)
    print("DATASET CLASS ANALYSIS")
    print("="*80)
    
    for split in ['train', 'test', 'val']:
        print(f"\n{split.upper()}:")
        print(f"  Total images: {analysis[split]['total_images']}")
        print(f"  Classes present: {sorted(analysis[split]['classes'])}")
        print(f"  Number of classes: {len(analysis[split]['classes'])}")
    
    print("="*80 + "\n")
    
    return analysis


if __name__ == "__main__":
    # Create metadata files
    create_dataset_metadata("PolypsSet")
    
    # Analyze dataset
    analyze_dataset_classes("PolypsSet")
