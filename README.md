# PolypNet — AI-Powered Colorectal Polyp Detection & Classification

End-to-end six-stage deep learning pipeline for colorectal polyp
detection, segmentation and malignancy classification from colonoscopy images.

## Pipeline Stages
1. Image Validation CNN — 100% test accuracy
2. CLAHE Enhancement
3. YOLOv11 Detection — 84.82% mAP, 93.77% Precision
4. CNN Ensemble Classification — 82.10% accuracy (VGG16, ResNet50, EfficientNetB0)
5. Attention U-Net Segmentation — 73.41% Dice, 64.08% IoU
6. LLaVA-1.5 Clinical Report Generation (7B parameter VLM)

## Model Weights
(https://drive.google.com/file/d/1pIT8MWJ7Nq5H-cTldy3QIXAIj6pCah1v/view?usp=sharing)
Place downloaded checkpoints/ folder in: ~/Desktop/fyp/fyp/

## Setup
pip install -r requirements.txt
python3 setup_vlm.py

## Run
cd webapp && python3 app.py

## Team
Usman Asif | Ahmad Fraz | Ali Iktikhar
Supervisor: Dr. Zoya | Co-Supervisor: Dr. Muhammad Hanif
GIKI — BS Data Science 2022-2026
