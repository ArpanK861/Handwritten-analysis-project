# Multi-Architecture Image Classification: 3-Class Custom Dataset

This project implements a comprehensive deep learning pipeline for classifying a custom 3-class image dataset. It evaluates and compares the performance of traditional Convolutional Neural Networks (CNNs) against modern Vision Transformers (ViT).

## 📊 Overview
The project is implemented in Google Colab and explores model performance under both augmented and non-augmented conditions to identify the most robust architecture for this specific dataset.

## 🏗️ Model Architectures

### 1. LeNet-5
A classic CNN architecture utilizing:
* Two sets of `Conv2D` and `AveragePooling2D` layers.
* Three `Dense` layers (120, 84, and 3 units).
* ReLU activation for hidden layers and Softmax for the output.

### 2. Vision Transformer (ViT)
A state-of-the-art transformer-based approach:
* **Patch Embedding**: Divides 256x256 images into 16x16 patches projected into a 128-dimension space.
* **Transformer Encoder**: Utilizes 6 encoder blocks with `MultiHeadAttention` (8 heads) and `LayerNormalization`.
* **Class Token**: Implements a learnable `ClassTokenLayer` for final feature extraction.



### 3. Custom 3-Layer CNN
A deeper sequential model designed for high-level feature extraction:
* Three `Conv2D` layers with increasing filter counts (64, 128, 128).
* `MaxPooling2D` with strides of 2 for dimensionality reduction.
* Two fully connected layers (200 and 50 units) before the classification head.



## 🚀 Key Features
* **Data Augmentation**: Uses `ImageDataGenerator` for rotation (30°), shifts, shears, zooms, and horizontal flips.
* **Automated Checkpoints**: Employs `ModelCheckpoint` to save the best model iteration based on `val_accuracy`.
* **Comprehensive Metrics**: Evaluates models using Accuracy, AUC, Precision, and Recall.
* **Visualization**: Generates training history curves and heatmapped Confusion Matrices for detailed error analysis.

## 📈 Performance Comparison
The **3-Layer CNN** showed the strongest results in this experimental setup.

| Model | Validation Acc (%) | Testing Acc (%) | Precision (%) | Recall (%) |
| :--- | :--- | :--- | :--- | :--- |
| **3-Layer CNN** | 95.57 | 92.50 | 95 | 95 |
| **LeNet-5** | 92.00 | 73.08 | 73 | 73 |
| **Vision Transformer** | 81.11 | 80.01 | 80 | 94 |
| **MobileNet V2** | 88.00 | 84.00 | 75 | 96 |



## 📂 Setup and Usage
1.  **Google Drive**: Ensure your dataset is stored at `/MyDrive/CustomDS/3Class/`.
2.  **Environment**: Run the notebook in Google Colab with a GPU-enabled runtime for the Vision Transformer.
3.  **Execution**: Run cells sequentially to load data, train the chosen architecture, and view comparison analytics.

---
*Developed for comparative analysis of modern image classification architectures.*
