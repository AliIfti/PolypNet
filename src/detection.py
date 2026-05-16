"""YOLOv11 object detection for polyp localization."""

from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np


def train_yolo(
    data_yaml: str,
    model_size: str = 'n',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = 'runs',
    name: str = 'polyp_yolo'
) -> str:
    """
    Train YOLOv11 model for polyp detection.
    
    Args:
        data_yaml: Path to dataset.yaml file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory
        name: Experiment name
        
    Returns:
        Path to best model weights
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        )
    
    # Initialize model
    model_name = f'yolo11{model_size}.pt'
    print(f"\n🚀 Training YOLOv11-{model_size} for polyp detection...")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {imgsz}")
    print(f"   Batch Size: {batch}")
    
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,
        save=True,
        device='0' if __import__('torch').cuda.is_available() else 'cpu',  # Auto-detect GPU
        plots=True
    )
    
    best_modelYOLO_WEIGHTS = Path.cwd() / "runs" / "detect" / "runs" / "detect" / "polyp_yolo4" / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"✅ Best model saved to: {best_modelYOLO_WEIGHTS}")
    
    return str(best_modelYOLO_WEIGHTS)


def detect_polyps(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.45,
    save_path: Optional[str] = None
) -> List[Dict]:
    """
    Detect polyps in an image using trained YOLO model.
    
    Args:
        model_path: Path to trained YOLO weights
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_path: Optional path to save annotated image
        
    Returns:
        List of detections, each containing:
        - bbox: [x1, y1, x2, y2]
        - confidence: float
        - class_id: int
        - class_name: str
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        )
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
    )
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            detections.append({
                'bbox': box.tolist(),
                'confidence': conf,
                'class_id': cls,
                'class_name': result.names[cls]
            })
    
    # Save annotated image if requested
    if save_path and detections:
        img = cv2.imread(image_path)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            label = f"{det['class_name']} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 0), 1)
        
        cv2.imwrite(save_path, img)
        print(f"✅ Annotated image saved to {save_path}")
    
    return detections


def evaluate_yolo(model_path: str, data_yaml: str) -> Dict:
    """
    Evaluate YOLO model on test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset.yaml
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        )
    
    print("\n📊 Evaluating YOLO model...")
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split='test')
    
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr)
    }
    
    print("\n📊 Evaluation Results:")
    print(f"   mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    return metrics
