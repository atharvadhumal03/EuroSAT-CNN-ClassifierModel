# 🛰️ EuroSAT CNN Classifier

A CNN-based land use classifier for the EuroSAT dataset, achieving **87.93% test accuracy** in satellite image classification across 10 distinct land cover categories using PyTorch.

---

## 📋 Project Overview

This project implements a custom **Convolutional Neural Network (CNN)** to classify satellite images from the EuroSAT dataset. The model identifies land use patterns from **Sentinel-2 satellite imagery**, categorizing images into 10 different classes of land cover with high precision and recall.

**Technical Highlights:**
- End-to-end deep learning pipeline with PyTorch
- Custom CNN architecture optimized for 64×64 RGB satellite imagery
- Proper train/validation/test methodology preventing data leakage
- Data augmentation pipeline for improved generalization
- Comprehensive evaluation with per-class metrics

---

## 🗂️ Dataset

**EuroSAT: Sentinel-2 Satellite Image Classification Dataset**

| Property | Value |
|----------|-------|
| Total Images | 27,000 labeled samples |
| Image Dimensions | 64×64×3 (RGB) |
| Data Source | Sentinel-2 Satellite |
| Classes | 10 land use categories |
| Data Split | 70% Train / 15% Val / 15% Test |

**Class Distribution:**
```
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, 
Pasture, PermanentCrop, Residential, River, SeaLake
```

---

## 🏗️ Model Architecture

**Custom CNN Design:**
```
Input (3×64×64)
    ↓
Conv2D(3→32) → ReLU → MaxPool2D(2×2)
    ↓
Conv2D(32→64) → ReLU → MaxPool2D(2×2)
    ↓
Conv2D(64→128) → ReLU → MaxPool2D(2×2)
    ↓
Conv2D(128→256) → ReLU → MaxPool2D(2×2)
    ↓
Flatten(256×4×4 = 4096)
    ↓
FC(4096→512) → ReLU → Dropout(0.6)
    ↓
FC(512→10) → Softmax
    ↓
Output (10 classes)
```

**Architecture Specifications:**
- **Convolutional Layers:** 4 layers with increasing filters (32→64→128→256)
- **Kernel Size:** 3×3 with padding=1
- **Pooling:** 2×2 Max Pooling after each conv layer
- **Activation:** ReLU
- **Regularization:** Dropout (p=0.6)
- **Output:** 10-way softmax classification

---

## 📊 Results

### 🎯 Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **87.93%** |
| Training Accuracy | 89.00% |
| Validation Accuracy | 87.00% |
| Macro Avg F1-Score | 0.87 |
| Weighted Avg F1-Score | 0.88 |

### 📈 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 🌊 SeaLake | 0.98 | 0.96 | **0.97** | 437 |
| 🏘️ Residential | 0.97 | 0.94 | **0.96** | 452 |
| 🌲 Forest | 0.94 | 0.97 | **0.96** | 452 |
| 🏭 Industrial | 0.91 | 0.97 | **0.94** | 387 |
| 🌾 AnnualCrop | 0.90 | 0.87 | **0.88** | 449 |
| 🌊 River | 0.91 | 0.85 | **0.88** | 360 |
| 🛣️ Highway | 0.82 | 0.86 | **0.84** | 401 |
| 🌿 Pasture | 0.89 | 0.76 | **0.82** | 303 |
| 🌱 HerbaceousVegetation | 0.69 | 0.94 | **0.79** | 435 |
| 🌳 PermanentCrop | 0.85 | 0.58 | **0.69** | 374 |

**Key Observations:**
- ✅ Water bodies (SeaLake, River) and urban areas (Residential, Industrial) show excellent performance
- ✅ Natural landscapes (Forest) are well-recognized with high F1-scores
- ⚠️ Agricultural classes (PermanentCrop, HerbaceousVegetation) show lower performance due to visual similarity

---

## ✨ Key Features

### 🔧 Technical Implementation
- **Data Augmentation Pipeline:**
  - RandomHorizontalFlip (p=0.5)
  - RandomVerticalFlip (p=0.5)
  - RandomRotation (±30°)
  - ColorJitter (brightness, contrast, saturation)

- **Training Strategy:**
  - Proper train/validation/test split (no data leakage)
  - Real-time progress tracking with tqdm
  - Batch processing (batch_size=64)
  - Adam optimizer with CrossEntropyLoss

- **Hardware Acceleration:**
  - Apple Silicon MPS support
  - NVIDIA CUDA compatibility
  - CPU fallback

---

## 🛠️ Requirements
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
torchinfo>=1.8.0
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/eurosat-cnn-classifier.git
cd eurosat-cnn-classifier
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv myVenv
source myVenv/bin/activate  # Windows: myVenv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Notebook

1. **Start Jupyter:**
```bash
jupyter notebook notebook.ipynb
```

2. **Execute cells sequentially** - The notebook includes:
   - 📦 Data loading and preprocessing
   - 🏗️ Model architecture definition  
   - 🎯 Training loop with validation
   - 📊 Evaluation and visualization
   - 📈 Classification report generation

**Note:** Dataset downloads automatically on first run (~2GB).

---

## 📁 Project Structure
```
eurosat-cnn-classifier/
│
├── 📂 data/                  # EuroSAT dataset (auto-downloaded)
├── 📂 myVenv/               # Virtual environment
├── 📓 notebook.ipynb        # Main implementation notebook
├── 📄 requirements.txt      # Python dependencies
├── 📖 README.md            # Project documentation
└── 🚫 .gitignore           # Git ignore rules
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |
| **Learning Rate** | Default (1e-3) |
| **Batch Size** | 64 |
| **Epochs** | 10 |
| **Dropout Rate** | 0.6 |
| **Device** | MPS / CUDA / CPU |

---

## 📓 Notebook Sections

1. **🔧 Setup & Imports** - Loading required libraries and checking device availability
2. **📊 Data Loading** - EuroSAT dataset with transforms, augmentation, and splits
3. **🏗️ Model Architecture** - Custom CNN class definition with forward pass
4. **🎯 Training Loop** - Epoch-wise training with validation monitoring
5. **📈 Evaluation** - Test set evaluation and performance metrics
6. **📉 Visualizations** - Loss/accuracy curves and training dynamics
7. **📋 Classification Report** - Detailed per-class performance analysis

---

## 📊 Visualizations

The notebook generates comprehensive visualizations:

- 📉 **Training Curves:** Loss and accuracy over epochs
- 🎯 **Validation Monitoring:** Real-time performance tracking
- 📊 **Classification Report:** Precision, recall, F1-scores per class
- 🖼️ **Sample Predictions:** Visual verification of model outputs

---

## 🔮 Future Improvements

- [ ] **Architecture Enhancements:** Experiment with ResNet, VGG, EfficientNet
- [ ] **Training Optimizations:** Learning rate scheduling, early stopping
- [ ] **Transfer Learning:** Leverage pretrained ImageNet models
- [ ] **Hyperparameter Tuning:** Grid search for optimal configuration
- [ ] **Ensemble Methods:** Combine multiple models for improved accuracy
- [ ] **Model Compression:** Quantization and pruning for deployment
- [ ] **Explainability:** Grad-CAM visualization for interpretability

---

## 🙏 Acknowledgments

- **Dataset:** [EuroSAT Dataset](https://github.com/phelber/EuroSAT) by Helber et al., 2019
- **Framework:** Built with [PyTorch](https://pytorch.org/)

---

## 👤 Author

**Atharva Dhumal**  
Graduate Student, Northeastern University  
 🎓 Project: Satellite Image Classification with CNNs

---

## 📄 License
feel free to use this project for learning and research purposes.

---

## 📚 References
```
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). 
EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
```

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**


</div>