# Setup Guide for DocuExtract - Odia OCR

This guide will help you set up the project for development and training.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows (WSL2)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space for dataset
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for training)

### Software Dependencies
- Git
- Python 3.8+
- pip (Python package manager)
- CUDA 11.8+ (if using GPU)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/docuextract-odia-ocr.git
cd docuextract-odia-ocr
```

### 2. Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 📦 Dataset Setup

### Option 1: Download Pre-processed Dataset (Recommended)

The preprocessed dataset is hosted separately due to size constraints.

```bash
# Create data directory
mkdir -p data/processed/splits

# Download from Hugging Face (or your hosting service)
# Instructions coming soon...
```

### Option 2: Build Dataset from Scratch

If you want to rebuild the entire dataset:

#### Step 1: Download Raw Datasets

Download the following datasets:

1. **Odia Handwritten Character Dataset**
   - Source: [Link to dataset]
   - Place in: `data/raw/Odia_handwritten/`

2. **Odia Number Dataset**
   - Source: [Link to dataset]
   - Place in: `data/raw/Odia_Number/`

3. **IndicSTR Odia Scene Text**
   - Source: https://indicstr.ai4bharat.org/
   - Filter Odia images and place in: `data/raw/indicstr_odia/`

4. **Mozhi Text Corpus**
   - Source: AI4Bharat IndicCorpV2
   - Use: `datasets` library (automated in scripts)

#### Step 2: Run Preprocessing Pipeline

```bash
cd src/data_prep

# 1. Create annotations
python create_annotations.py \
  --handwritten-dir ../../data/raw/Odia_handwritten \
  --output ../../data/annotations/

# 2. Balance dataset
python balance_dataset.py \
  --input ../../data/annotations/handwritten_characters_full.json \
  --output ../../data/annotations/balanced_annotations.json

# 3. Generate synthetic data
python generate_missing_characters.py \
  --missing-classes 4,5,6,7,8,9,36,37,38,39,40,41,42,43,44,45,46 \
  --samples-per-class 3000 \
  --output ../../data/synthetic/missing_characters/

# 4. Create splits
python create_splits.py \
  --input ../../data/annotations/complete_dataset.json \
  --output-dir ../../data/processed/splits/
```

---

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# Data paths
DATA_DIR=/path/to/your/data
CHECKPOINT_DIR=/path/to/checkpoints

# Training settings
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=50

# Weights & Biases (optional)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=docuextract-odia-ocr
```

### GPU Setup

**Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Install CUDA-compatible PyTorch (if needed):**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧪 Verify Setup

### Test Data Loading

```python
import json
from pathlib import Path

# Check if splits exist
splits_dir = Path("data/processed/splits")
if (splits_dir / "train.json").exists():
    with open(splits_dir / "train.json", 'r') as f:
        train_data = json.load(f)
    print(f"✓ Training data: {len(train_data['annotations']):,} images")
else:
    print("✗ Training split not found. Please download or build dataset.")
```

### Test Image Loading

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample image
sample_path = "data/samples/10010.jpg"
img = Image.open(sample_path)

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='gray')
plt.title("Sample Odia Character")
plt.axis('off')
plt.show()

print(f"✓ Image loaded: {img.size}")
```

---

## 📊 Project Structure Verification

After setup, your directory should look like:

```
docuextract-odia-ocr/
├── src/
│   └── data_prep/           # ✓ 4 Python scripts
├── data/
│   ├── annotations/         # ✓ metadata.json, class_distribution.json
│   ├── samples/             # ✓ ~10 sample images
│   └── processed/splits/    # ✓ train.json, val.json, test.json (if downloaded)
├── fonts/                   # ✓ 4 Odia TTF fonts
├── docs/                    # ✓ 5 markdown files
├── requirements.txt         # ✓
├── README.md               # ✓
└── LICENSE                 # ✓
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'cv2'
```bash
pip install opencv-python
```

#### 2. NumPy version conflict
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

#### 3. CUDA out of memory (during training)
- Reduce batch size in training config
- Use gradient accumulation
- Enable mixed precision training (fp16)

#### 4. Permission denied errors
```bash
chmod +x src/data_prep/*.py
```

---

## 🚦 Next Steps

After successful setup:

1. **Explore the data**:
   ```bash
   jupyter notebook notebooks/
   ```

2. **Run data preprocessing** (if building from scratch):
   ```bash
   cd src/data_prep
   python create_annotations.py
   ```

3. **Start training** (Phase 2):
   ```bash
   # Coming soon: training scripts
   python src/training/train_deepseek.py
   ```

---

## 📚 Additional Resources

- **Documentation**: See `docs/` folder
- **Project Plan**: `PROJECT_PLAN.md`
- **Dataset Info**: `docs/DATASETS_README.md`
- **Phase 1 Summary**: `docs/PHASE1_COMPLETE_SUMMARY.md`

---

## 💬 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [documentation](docs/)
3. Open an issue on GitHub
4. Contact: [your.email@example.com]

---

**Last Updated**: January 2026
