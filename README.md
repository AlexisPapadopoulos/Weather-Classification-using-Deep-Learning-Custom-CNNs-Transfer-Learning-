# Weather Image Classification using Neural Networks

## Overview

This project explores **neural networks for image classification**, focusing on recognizing **weather conditions from images**. Multiple custom Convolutional Neural Networks (CNNs) and transfer learning models were implemented, trained, and evaluated to compare performance.

The goal is to classify images into four weather categories:

* Snow
* Rain
* Sandstorm
* Fog / Smog

The project was developed as part of an academic assignment.

---

## Dataset

* **Source:** Kaggle – Weather Dataset
* **Total images:** 1472
* **Selected classes:** Snow, Rain, Sandstorm, Fog/Smog

### Dataset Splitting

* Train/Test split: **80/20**

  * Train: 1176 images
  * Test: 296 images
* Internal split of training data:

  * Training: 846 images
  * Validation: 212 images
  * Internal test: 118 images

### Preprocessing

* Dataset balanced across all classes (368 images per class)
* Pixel normalization to **[0, 1]**
* Image resizing:

  * Custom CNNs: **64×64×1 (grayscale)**
  * Transfer learning models: **224×224×3 (RGB)**

---

## Models Implemented

### Custom CNN Models

#### CNN Model 1

* 4 convolutional blocks (16 → 128 filters)
* Batch Normalization, MaxPooling, Dropout (0.2)
* Dense layers: 32 → 64 → Softmax
* **Best validation accuracy:** 73.58%

#### CNN Model 2

* 5 convolutional blocks (64 → 256 filters)
* Dropout range: 0.2–0.3
* Dense layers: 64 → 128 → Softmax
* **Best validation accuracy:** 74.06%

#### CNN Model 3

* 6 convolutional blocks (32 → 256 → 64)
* Higher dropout (up to 0.5)
* **Best validation accuracy:** 68.87%

#### CNN Model 4

* Mixed convolutional architecture
* Larger training duration
* **Best validation accuracy:** 78.30%

---

### Custom ResNet Models

#### Model 5 – Custom ResNet

* Residual blocks: 64 → 128 → 256
* Global Average Pooling
* Dense blocks: 256 → 128 → 64
* **Best validation accuracy:** **90.57%**

#### Model 8 – Custom ResNet + Data Augmentation

* Same architecture as Model 5
* Data augmentation applied during training
* **Best validation accuracy:** 80.77%

Data augmentation techniques:

* Small rotations (±3%)
* Horizontal and vertical shifts (±3%)

---

### Transfer Learning Models

#### Model 6 – VGG16

* Pretrained VGG16 (frozen, no top layers)
* Global Average Pooling
* Dense layer (512 units)
* **Best validation accuracy:** 83.96%

#### Model 7 – ResNet50

* Pretrained ResNet50 (frozen)
* Global Average Pooling
* Dense layer (512 units)
* **Best validation accuracy:** 83.02%

#### Model 9 – VGG16 + Data Augmentation

* VGG16 with data augmentation
* **Best validation accuracy:** 81.13%

Data augmentation:

* Horizontal flip
* Rotation (±5%)
* Translation (±5%)

#### Model 10 – ResNet50 + Data Augmentation

* ResNet50 with data augmentation
* **Best validation accuracy:** 85.85%

Data augmentation:

* Horizontal flip
* Rotation (±8%)
* Zoom (±8%)

---

## Training Details

* Optimizers used: **Adam**, **AdamW**
* Learning rate scheduling
* Early stopping applied where appropriate
* Batch sizes ranged from **8 to 32**
* Epochs ranged from **20 to 200**

---

## Results Summary

* Best overall performance achieved by **Custom ResNet Model (Model 5)** with **90.57% validation accuracy**
* Transfer learning models showed strong and stable performance
* Data augmentation improved generalization in some models but not consistently across all architectures

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Kaggle Datasets

---

## Author

**Alexios Papadopoulos**

---

## Acknowledgments

* Kaggle for the weather image dataset
* Pretrained models: VGG16 and ResNet50

---

Thank you for reading.
