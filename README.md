```markdown
# PolypNet
### AI-Powered Colorectal Polyp Detection, Segmentation & Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
PolypNet is an end-to-end six-stage deep learning pipeline for automated colorectal polyp detection, segmentation and malignancy classification from colonoscopy images. It integrates YOLOv11, Attention U-Net, a three-model CNN ensemble and LLaVA-1.5 Vision Language Model for automated clinical report generation.

## Results

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv11 Detection | mAP@0.50 | 84.82% |
| YOLOv11 Detection | Precision | 93.77% |
| YOLOv11 Detection | Recall | 73.00% |
| YOLOv11 Detection | F1 Score | 82.09% |
| Attention U-Net | Dice Score | 73.41% |
| Attention U-Net | IoU | 64.08% |
| Attention U-Net | Precision | 84.13% |
| CNN Ensemble | Overall Accuracy | 82.10% |
| CNN Ensemble | Malignant Recall | 70.57% |
| Validation CNN | Test Accuracy | 100% |

## Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Image Validation | Custom CNN + 7 heuristic checks. Rejects non-colonoscopy images. |
| 2 | CLAHE Enhancement | Contrast Limited Adaptive Histogram Equalization |
| 3 | YOLOv11 Detection | Real-time polyp localization with bounding boxes |
| 4 | CNN Classification | 3-model ensemble — VGG16, ResNet50, EfficientNetB0 |
| 5 | Attention U-Net | Pixel-level polyp segmentation with YOLO-gated masking |
| 6 | LLaVA-1.5 VLM | 7B parameter Vision Language Model for clinical report generation |

## Installation

```bash
git clone https://github.com/AliIfti/PolypNet.git
cd PolypNet
pip install -r requirements.txt
```

## Model Weights

Download pretrained model weights from Google Drive:

**[Download checkpoints.zip (3.0 GB)](https://drive.google.com/file/d/1pIT8MWJ7Nq5H-cTldy3QIXAIj6pCah1v/view)**

After downloading, extract in the project root:

```bash
unzip checkpoints_backup.zip
```

Expected structure:
```
checkpoints/
├── segmentation/
│   └── unet_bce_dice_best.pth
├── classification/
│   ├── VGG16/best_model.pth
│   ├── ResNet50/best_model.pth
│   └── EfficientNetB0/best_model.pth
└── validation/
    └── validation_model.pth
```

## Setup VLM (Optional)

To enable LLaVA-1.5 clinical report generation:

```bash
python3 setup_vlm.py --model llava
```

Requires 8GB+ VRAM. Falls back to structured template if unavailable.

## Run Web Application

```bash
cd webapp
python3 app.py
```

Open in browser: http://127.0.0.1:5000

## Dataset

- 75,000 annotated colonoscopy images
- Sources: Kvasir-SEG, CVC-ClinicDB, ETIS-LaribPolypDB
- Split: 70% train / 15% val / 15% test
- Augmentation: mosaic, flipping, rotation, brightness jitter

## Training

```bash
# Retrain segmentation
python3 retrain_unet.py

# Retrain classification
python3 train_classifiers.py

# Retrain validation CNN
python3 train_validation_model.py
```

## Project Structure

```
PolypNet/
├── src/
│   └── vlm.py                  # LLaVA-1.5 VLM integration
├── webapp/
│   ├── app.py                  # Flask application
│   ├── pipeline.py             # Six-stage pipeline
│   ├── config.py               # Configuration
│   ├── templates/              # HTML frontend
│   └── static/                 # CSS and JS
├── setup_vlm.py                # LLaVA-1.5 download script
├── train_classifiers.py        # CNN ensemble training
├── train_validation_model.py   # Validation CNN training
├── retrain_unet.py             # Attention U-Net training
├── requirements.txt            # Dependencies
└── README.md
```

## Team

| Name | Student ID |
|------|------------|
| Usman Asif | 2022612 |
| Ahmad Fraz | 2022065 |
| Ali Iktikhar | 2022341 |

**Supervisor:** Dr. Zoya  
**Co-Supervisor:** Dr. Muhammad Hanif  
**Institution:** Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)  
**Degree:** Bachelor of Data Science (2022-2026)
```
