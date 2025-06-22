# Alzheimer's Disease Detection and Image Generation

This project provides an advanced suite of deep learning models for Alzheimer's disease detection from brain MRI scans and synthetic MRI image generation. It combines state-of-the-art classification models with various generative adversarial networks (GANs) to provide a comprehensive tool for both analysis and research.

## Features

### 1. MRI Classification
- **Multiple Model Support**: 
  - ResNet101: Deep residual network with 101 layers
  - ResNeXt101_32x8d: Residual network with grouped convolutions
  - DenseNet161: Densely connected convolutional network
  - Ensemble: Combines predictions from all models for higher accuracy

- **Classification Categories**:
  - NonDemented: Normal cognitive function
  - VeryMildDemented: Very mild cognitive decline
  - MildDemented: Mild cognitive decline
  - ModerateDemented: Moderate cognitive decline

### 2. Synthetic MRI Generation
- **CGAN (Conditional GAN)**: Generate synthetic MRI images conditioned on Alzheimer's stage
- **CycleGAN**: Transform between normal and Alzheimer's affected brain images
- **HighResGAN**: Generate detailed high-resolution brain MRI images

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alzimer2

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
alzimer2/
├── models/                      # Pre-trained model weights
│   ├── best_resnet101.pth       # ResNet101 classification model
│   ├── best_resnext101_32x8d.pth # ResNeXt101 classification model
│   ├── best_densenet161.pth     # DenseNet161 classification model
│   ├── cgan_generator.pth       # CGAN model
│   ├── cyclegan_G_AB.pth        # CycleGAN: Normal to Alzheimer's
│   ├── cyclegan_G_BA.pth        # CycleGAN: Alzheimer's to Normal
│   └── highres_gan_generator.pth # HighResGAN model
├── segmented_images/            # Training/test images
│   └── train/                   # Training data organized by category
│       ├── NonDemented/
│       ├── VeryMildDemented/
│       ├── MildDemented/
│       └── ModerateDemented/
├── generated_images/            # Output directory for generated images
├── uploaded_images/             # Storage for user-uploaded images
├── app.py                       # Streamlit web application
├── test.py                      # Classification model definitions & training
├── train_resnext.py             # ResNeXt model specific training
├── train_densenet.py            # DenseNet model specific training
├── gan_models.py                # GAN model definitions
└── README.md                    # This file
```

## Usage

### Running the Web Application
```bash
streamlit run app.py
```
This will start a web server and open the application in your default browser.

### Application Modes

1. **Classification Mode**
   - Select a classification model (ResNet101, ResNeXt101, DenseNet161, or Ensemble)
   - Upload a brain MRI scan
   - View the classification result with probability distribution
   - When using Ensemble mode, individual model predictions are also displayed

2. **Image Generation Mode**
   - Select a GAN model (CGAN, CycleGAN, or HighResGAN)
   - For CGAN: Select an Alzheimer's stage to generate
   - For CycleGAN: Upload an image to transform between normal and Alzheimer's state
   - For HighResGAN: Optionally condition on an Alzheimer's stage
   - Generated images are saved to the `generated_images` directory

3. **About Mode**
   - Information about the application and models

### Training Custom Models

#### Classification Models
```bash
# Train ResNeXt model
python train_resnext.py

# Train DenseNet model  
python train_densenet.py

# Train all models (including ensemble)
python test.py
```

#### Force CPU Usage
To force CPU usage (helpful for systems with limited GPU memory):
```bash
$env:FORCE_CPU=1; python train_resnext.py
```

## Model Details

### Classification Models
- **ResNet101**: Deep residual network that helps solve the vanishing gradient problem
- **ResNeXt101_32x8d**: Enhanced ResNet with grouped convolutions for better accuracy
- **DenseNet161**: Each layer connects to every other layer, reducing parameters while maintaining high accuracy
- **Ensemble**: Combines predictions from all models to achieve higher accuracy and robustness

### GAN Models
- **CGAN**: Generates MRI images based on a specified Alzheimer's stage
- **CycleGAN**: Uses cycle consistency to transform images between domains (normal ↔ Alzheimer's)
- **HighResGAN**: Specialized GAN for generating high-resolution detailed MRI images

## Dataset
The models were trained on segmented brain MRI images from the OASIS and ADNI datasets. The dataset includes:
- 9,600 NonDemented images
- 8,960 VeryMildDemented images
- 8,960 MildDemented images
- 6,464 ModerateDemented images

## Requirements
- Python 3.7+
- PyTorch 1.8+
- Streamlit 1.0+
- scikit-learn
- matplotlib
- seaborn
- numpy
- Pillow
- CUDA-capable GPU (recommended)

## License
[Specify your license here]

## Acknowledgments
- [OASIS Dataset](https://www.oasis-brains.org/)
- [ADNI Dataset](http://adni.loni.usc.edu/)
- PyTorch team for providing pre-trained model architectures 