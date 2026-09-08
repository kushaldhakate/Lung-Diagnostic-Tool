# 🫁 LungAI — AI-Based Lung Disease Classification Tool

LungAI is a deep learning-based medical image classification application developed
as a 3-member team project. The system uses **PyTorch** and **EfficientNetV2-M**
to classify lung-related medical images into three categories:

- Normal
- Pneumonia
- Lung Cancer

The application supports multiple medical imaging modalities, including:

- Chest X-rays
- CT scans
- Histopathological images

A **Gradio-based interface** allows users to upload an image and view the
model's predicted class and confidence scores.

> ⚠️ This project is intended for educational and research purposes only and
> is not intended for clinical diagnosis or medical decision-making.

---

## 📌 Project Overview

The goal of LungAI is to explore the application of deep learning to automated
classification of lung-related medical images.

The project combines image preprocessing, deep learning-based classification,
and an interactive user interface into a single application.

### Classification Classes

| Class | Description |
|-------|-------------|
| Normal | Normal lung image |
| Pneumonia | Image associated with pneumonia |
| Lung Cancer | Image associated with lung cancer |

---

## 👨‍💻 My Contribution

This was developed as a collaborative project by a team of **3 members**.

My primary responsibilities were **data preprocessing and model training**.

### Data Preprocessing

- Prepared and organized the medical image datasets for training and evaluation.
- Applied image preprocessing and transformations required by the model.
- Resized input images to **224 × 224 pixels**.
- Converted images into tensors and applied normalization.
- Prepared the data pipeline for efficient model training.

### Model Development & Training

- Implemented the image classification pipeline using **PyTorch**.
- Used **EfficientNetV2-M** as the deep learning architecture.
- Configured the model for classification into three disease categories.
- Trained and evaluated the model on the prepared datasets.
- Analyzed model predictions and classification performance.

---

## ✨ Features

- 🧠 Deep learning-based image classification
- 🏗️ EfficientNetV2-M architecture
- 🖼️ Support for X-ray, CT, and histopathological images
- 📊 Confidence score prediction
- 📈 Probability distribution across classification classes
- 🖥️ Interactive Gradio interface
- ⚡ GPU/CPU inference support
- 🔍 Visualized prediction output

---

## 🧠 Model Architecture

The project uses **EfficientNetV2-M** for image classification.

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNetV2-M |
| Framework | PyTorch |
| Number of Classes | 3 |
| Input Size | 224 × 224 |
| Output Classes | Normal, Pneumonia, Lung Cancer |
| Model Weights | `best_model.pth` |

---

## 📊 Model Performance

The model achieved approximately **94–95% test accuracy** during evaluation,
with performance varying slightly depending on the evaluation run.

The model was evaluated across multiple medical image types:

- Chest X-rays
- CT scans
- Histopathological images

The classification performance was evaluated using metrics including:

- Accuracy
- Precision
- Recall
- F1-score

> Performance figures are approximate and may vary depending on the dataset
> split, preprocessing configuration, and evaluation run.

---

## 🔄 How It Works

```text
Medical Image Upload
        ↓
Image Preprocessing
        ↓
Resize to 224 × 224
        ↓
Tensor Conversion & Normalization
        ↓
EfficientNetV2-M
        ↓
Class Probability Prediction
        ↓
Predicted Class + Confidence Score
