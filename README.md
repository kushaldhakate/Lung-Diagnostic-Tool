# 🫁 LungAI — AI-Based Lung Disease Diagnostic Tool

An AI-powered medical image diagnostic system built using **PyTorch**, **EfficientNetV2-M**, and **Gradio** for detecting lung diseases from medical images such as:

* Chest X-rays
* CT Scans
* Histopathological Images

The model classifies images into:

* Normal
* Pneumonia
* Lung Cancer

---

## 🚀 Features

* Deep Learning based diagnosis using EfficientNetV2-M
* Interactive Gradio frontend
* Annotated disease region visualization
* Confidence score prediction
* Clinical findings display
* Probability distribution for all classes
* Supports multiple medical imaging modalities
* GPU/CPU support

---

## 🧠 Model Information

| Parameter    | Value            |
| ------------ | ---------------- |
| Architecture | EfficientNetV2-M |
| Framework    | PyTorch          |
| Classes      | 3                |
| Input Size   | 224 × 224        |
| Model File   | `best_model.pth` |

### Classes

1. Normal
2. Pneumonia
3. Lung Cancer

---

# 📂 Project Structure

```text
LungAI/
│
├── Lung_Diagnostic_App.py
├── best_model.pth
├── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/LungAI.git
cd LungAI
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Create a `requirements.txt` file with:

```text
torch
torchvision
timm
gradio
numpy
pillow
```

---

# ▶️ Running the Application

Place the model file:

```text
best_model.pth
```

inside the project folder.

Then run:

```bash
python app.py
```

---

# 🌐 Application Interface

After running the application, Gradio will generate a local URL:

```text
http://127.0.0.1:7860
```

Open it in your browser.

---

# 🖼️ Workflow

```text
Medical Image Upload
        ↓
Image Preprocessing
        ↓
EfficientNetV2-M Model
        ↓
Disease Prediction
        ↓
Confidence Score Generation
        ↓
Annotated Output + Clinical Findings
```

---

# 🔍 Image Processing Pipeline

The uploaded image undergoes:

1. Image resizing (224×224)
2. Tensor conversion
3. Normalization
4. Deep learning inference
5. Softmax probability prediction
6. Region annotation

---

# 📊 Output Provided

The system provides:

* Predicted disease class
* Confidence percentage
* Probability scores for all classes
* Annotated disease regions
* Clinical observations
* Severity indication
* Suggested urgency level

---

# 🛠️ Technologies Used

* Python
* PyTorch
* torchvision
* timm
* Gradio
* NumPy
* Pillow

---

# 🧪 Sample Prediction

| Input            | Prediction  |
| ---------------- | ----------- |
| Chest X-ray      | Pneumonia   |
| CT Scan          | Lung Cancer |
| Normal Lung Scan | Normal      |

---

# ⚠️ Disclaimer

This project is developed for:

* Educational purposes
* Research purposes

It is **NOT** intended to replace professional medical diagnosis or clinical decision-making.

All predictions should be reviewed by qualified healthcare professionals.

---
