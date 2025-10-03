# Multi-QR Code Recognition Hackathon

This repository contains a two-stage solution for multi-QR code detection and decoding using YOLOv8 for detection and OpenCV's QRCodeDetector for decoding, with automatic classification of QR types (batch, manufacturer, distributor, regulator).

## 📁 Repository Structure

```
multiqr-hackathon/
├── README.md                           # Setup & usage instructions
├── requirements.txt                    # Python dependencies
├── train.py                           # Model training script
├── infer.py                           # Stage 1: Detection inference
├── infer2.py                          # Stage 2: Decoding inference
├── outputs/
│   ├── submission_detection_1.json    # Stage 1 output
│   └── submission_decoding_2.json     # Stage 2 output
└── src/
    ├── models/                        # Model weights and outputs
    ├── datasets/                      # Dataset files
    ├── utils/                         # Utility functions
    └── __init__.py
```

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sudeepa110/multiqr-hackathon.git
cd multiqr-hackathon
```

### 2. Environment Setup

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

Place your dataset folder (eg.`QR_DS`) inside the `src/datasets/` directory, then update the `data.yaml` file with correct paths,i.e copy the paths of test_images,train_images,valid_ images placed in src/datasets/QR_DS and paste it in the data.yaml file:


**Example `data.yaml`:**

```yaml
# Training images path
train: /path/to/your/project/src/datasets/QR_DS/train_images/images

# Validation images path
val: /path/to/your/project/src/datasets/QR_DS/valid_images/images

# Test images path
test: /path/to/your/project/src/datasets/QR_DS/test_images/images

# Number of classes
nc: 1

# Class names
names: ['qr_codes']
```

> **Note:** Replace `/path/to/your/project/` with your actual project directory path.

## ⚙️ Configuration

All scripts contain configurable variables at the top for easy customization.

### Training Configuration (`train.py`)

```python
DATA_YAML = "src/datasets/data.yaml"    # Dataset configuration path
EPOCHS = 100                             # Training epochs
BATCH_SIZE = 16                          # Batch size
IMG_SIZE = 640                           # Training image size
PATIENCE = 20                            # Early stopping patience
PRETRAINED_MODEL = "yolov8n.pt"         # Pretrained model
PROJECT_NAME = "src/models"              # Model save directory
MODEL_NAME = "qr_detector"               # Model subfolder name
```


## 🎯 Usage

### Training the Model

Train the YOLOv8 model for QR code detection:

```bash
python train.py
```

**Output:** Trained model weights saved to `src/models/qr_detector/weights/best.pt`

### Stage 1: QR Code Detection

Detects QR codes in images and generates bounding box predictions.

**Steps:**

1. **Update paths in `infer.py`:**
   - `MODEL_PATH`: Path to `best.pt` from `src/models/qr_detector/weights/`
   - `INPUT_DIR`: Path to `test_images` from `src/datasets/QR_DS/`
   - `OUTPUT_DIR`: Path to `outputs` folder

### Stage 1 Configuration (`infer.py`) 
   - You find it at the end of the code in infer.py

```python
MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
SAVE_IMAGES = True
```
2. **Run detection:**
   ```bash
   python infer.py
   ```

3. **Output:**
   - Annotated images: `src/models/predict_stage1_annotated/`
   - JSON results: `outputs/submission_detection_1.json`

### Stage 2: QR Code Decoding

Decodes detected QR codes and classifies them by type.

**Steps:**

1. **Update paths in `infer2.py`:**
   - `MODEL_PATH`: Path to `best.pt` from `src/models/qr_detector/weights/`
   - `INPUT_DIR`: Path to `test_images` from `src/datasets/QR_DS/`
   - `OUTPUT_DIR`: Path to `outputs` folder

### Stage 2 Configuration (`infer2.py`)
 - You find it at the end of the code in infer2.py

```python
MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
SAVE_IMAGES = True
```


2. **Run decoding:**
   ```bash
   python infer2.py
   ```

3. **Output:**
   - Annotated images: `src/models/predict_stage2_annotated/`
   - JSON results: `outputs/submission_decoding_2.json`

## 📋 Requirements

- Python 
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

See `requirements.txt` for the complete list of dependencies.

## 📝 Notes

- Ensure all paths in configuration files use absolute paths or are relative to the project root
- The `data.yaml` file must be updated with correct paths before training
- Model weights are saved automatically during training with early stopping enabled
- Both stages can save annotated images for visual inspection by setting `SAVE_IMAGES = True`

## 🏆 Submission Files

- **Stage 1:** `outputs/submission_detection_1.json` - QR code bounding box predictions
- **Stage 2:** `outputs/submission_decoding_2.json` - Decoded QR codes with type classification