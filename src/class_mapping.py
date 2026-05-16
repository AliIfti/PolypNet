# Polyp/Cancer Type Mapping
# Based on common medical classification systems for polyps and colorectal lesions

POLYP_CLASSES = {
    1: "Hyperplastic Polyp",
    2: "Tubular Adenoma", 
    3: "Tubulovillous Adenoma",
    4: "Villous Adenoma",
    5: "Sessile Serrated Adenoma",
    6: "Traditional Serrated Adenoma",
    7: "Inflammatory Polyp",
    8: "Hamartomatous Polyp",
    9: "Lipoma",
    10: "Carcinoid Tumor",
    11: "Early Colorectal Cancer (T1)",
    12: "Colorectal Cancer (T2)",
    13: "Advanced Colorectal Cancer (T3)",
    14: "Metastatic Colorectal Cancer (T4)",
    15: "Adenocarcinoma",
    16: "Mucinous Adenocarcinoma",
    17: "Signet Ring Cell Carcinoma",
    18: "Squamous Cell Carcinoma",
    19: "Neuroendocrine Tumor",
    20: "Lymphoma",
    21: "Gastrointestinal Stromal Tumor (GIST)",
    22: "Leiomyoma",
    23: "Hemangioma",
    24: "Other/Unclassified"
}

# Location mapping (if applicable)
LOCATION_CLASSES = {
    1: "Cecum",
    2: "Ascending Colon",
    3: "Hepatic Flexure",
    4: "Transverse Colon",
    5: "Splenic Flexure",
    6: "Descending Colon",
    7: "Sigmoid Colon",
    8: "Rectosigmoid Junction",
    9: "Rectum",
    10: "Multiple Locations"
}

def get_class_name(class_id: int, use_location: bool = False) -> str:
    """
    Get human-readable class name from class ID.
    
    Args:
        class_id: Numeric class identifier
        use_location: If True, use location mapping instead of polyp type
        
    Returns:
        Human-readable class name
    """
    mapping = LOCATION_CLASSES if use_location else POLYP_CLASSES
    return mapping.get(class_id, f"Unknown Class {class_id}")


def get_all_class_names(use_location: bool = False) -> list:
    """
    Get list of all class names in order.
    
    Args:
        use_location: If True, return location names instead of polyp types
        
    Returns:
        List of class names
    """
    mapping = LOCATION_CLASSES if use_location else POLYP_CLASSES
    max_id = max(mapping.keys())
    return [mapping.get(i, f"Class_{i}") for i in range(max_id + 1)]
