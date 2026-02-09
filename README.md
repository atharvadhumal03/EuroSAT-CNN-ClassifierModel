# EuroSAT CNN Classifier

A CNN-based land use classifier for the EuroSAT dataset, achieving high-accuracy satellite image classification across 10 distinct land cover categories using PyTorch.

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) to classify satellite images from the EuroSAT dataset. The model identifies land use patterns from Sentinel-2 satellite imagery, categorizing images into 10 different classes of land cover.

## Dataset

**EuroSAT Dataset**
- 27,000 labeled Sentinel-2 satellite images
- 10 land use classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Image dimensions: 64×64 RGB
- Split: 70% training / 15% validation / 15% test

## Model Architecture

**Custom CNN Architecture:**
- 4 Convolutional layers (3→32→64→128→256 channels)
- ReLU activation functions
- Max pooling layers (2×2) after each convolution
- 2 Fully connected layers (4096→512→10)
- Dropout (p=0.6) for regularization
- Total parameters: ~16.8M

## Results

**Performance Metrics:**
- **Test Accuracy:** 87.93%
- **Training Accuracy:** 89%
- **Validation Accuracy:** 87%

**Per-Class Performance (F1-Scores):**
- SeaLake: 0.97
- Residential: 0.96
- Forest: 0.96
- Industrial: 0.94
- AnnualCrop: 0.88
- River: 0.88
- Highway: 0.84
- Pasture: 0.82
- HerbaceousVegetation: 0.79
- PermanentCrop: 0.69

## Key Features

- **Data Augmentation:** RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
- **Proper train/val/test split** with no data leakage
- **Progress tracking** using tqdm
- **Comprehensive evaluation** with classification reports and confusion matrices
- **GPU acceleration** support (MPS for Apple Silicon, CUDA for NVIDIA)

## Requirements
```txt
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm
torchinfo
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eurosat-cnn-classifier.git
cd eurosat-cnn-classifier
```

2. Create and activate virtual environment:
```bash
python -m venv myVenv
source myVenv/bin/activate  # On Windows: myVenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Training the model:**
```python
python train.py
```

The script will:
- Automatically download the EuroSAT dataset
- Train the model for 10 epochs
- Display training progress with tqdm
- Save performance metrics and visualizations

**Evaluating the model:**
```python
python evaluate.py
```

## Project Structure
```
eurosat-cnn-classifier/
├── data/                  # Dataset directory (auto-downloaded)
├── models/               # Model checkpoints
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore file
```

## Training Details

- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 64
- **Epochs:** 10
- **Device:** MPS (Apple Silicon) / CUDA / CPU

## Visualizations

The training process generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix
- Per-class performance metrics

## Future Improvements

- [ ] Experiment with deeper architectures (ResNet, VGG)
- [ ] Implement learning rate scheduling
- [ ] Add early stopping
- [ ] Try transfer learning with pretrained models
- [ ] Ensemble multiple models
- [ ] Optimize hyperparameters

## Acknowledgments

- EuroSAT dataset: [Helber et al., 2019](https://github.com/phelber/EuroSAT)
- Built with PyTorch

## Author

Atharva Dhumal - Northeastern University
