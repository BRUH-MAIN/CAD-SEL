# CAD-SEL: A dual-modal colonoscopy dataset of subepithelial lesion

A deep learning-based computer-aided diagnosis system for detecting and classifying neuroendocrine tumors (NETs) in endoscopic images.

## Project Overview

CAD-SEL is a comprehensive medical AI system designed to assist in the diagnosis of neuroendocrine tumors using both white-light endoscopy (WL) and endoscopic ultrasound (EUS) images. The system supports multiple state-of-the-art object detection models and provides detailed performance analysis.


## Supported Models

### Faster R-CNN Series
- `fasterrcnn_resnet`: ResNet50-FPN backbone
- `fasterrcnn_resnet50_fpn_v2`: ResNet50-FPN V2
- `fasterrcnn_mobilenet`: MobileNetV3-Large-FPN (lightweight)
- `fasterrcnn_mobilenet_320`: 320px input version
- `fasterrcnn_mobilenet_320_v2`: 320px V2 version

### Other Detection Models
- **SSD Series**: SSD300-VGG16, SSDLite320-MobileNetV3
- **RetinaNet Series**: RetinaNet-ResNet50-FPN and V2
- **FCOS**: FCOS-ResNet50-FPN

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CAD-SEL.git
cd CAD-SEL

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train a Faster R-CNN model on WL dataset
python train.py \
    --model_type fasterrcnn_mobilenet_320 \
    --images_dir data/Images/Internal/NET-WL/ \
    --labels_dir data/Labels/Internal/NET-WL/ \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --output_dir checkpoints/output_model
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --model_type fasterrcnn_mobilenet_320 \
    --images_dir data/Images/Internal/NET-WL/ \
    --labels_dir data/Labels/Internal/NET-WL/ \
    --output_dir evaluation_results \
    --conf_threshold 0.5
```

## Project Structure

```
CAD-SEL/
├── models/
│   └── model.py              # Model architectures and utilities
├── dataloader.py             # Dataset and data loading
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── dataset_statistics.py     # Dataset analysis tools
├── check_data_correspondence.py  # Data validation
├── rename_categories.py      # Data preprocessing
├── run_eval.sh              # Batch evaluation script
└── README.md                # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision 0.9+
- OpenCV
- NumPy
- Matplotlib
- pandas
- tqdm

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
