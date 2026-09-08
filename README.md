# 🫁 LungAI — AI-Based Lung Disease Classification Tool

LungAI is a deep learning-based medical image classification application
developed as a **3-member team project**. The system uses **PyTorch** and
**EfficientNet-B3** to classify lung-related medical images into three
categories:

- Normal
- Pneumonia
- Lung Cancer

The application supports multiple medical imaging modalities:

- Chest X-rays
- CT scans
- Histopathological images

A **Gradio-based interface** allows users to upload a medical image and view
the predicted class, confidence score, probability distribution, and
visualized prediction output.

> ⚠️ **Disclaimer:** This project is intended for educational and research
> purposes only. It is not a medical device and should not be used for
> clinical diagnosis or treatment decisions.

---

## 📌 Project Overview

The objective of LungAI is to explore the use of deep learning for
classification of lung-related medical images.

The system combines:

1. Medical image validation
2. Image modality detection
3. Image preprocessing
4. Deep learning-based classification
5. Prediction confidence and probability visualization
6. Interactive Gradio interface

### Classification Classes

| Class | Description |
|-------|-------------|
| Normal | Normal lung image |
| Pneumonia | Image classified as pneumonia |
| Lung Cancer | Image classified as lung cancer |

---

## 👨‍💻 My Contribution

This project was developed collaboratively by a **3-member team**.

My primary responsibilities were **data preprocessing and model training**.

### Data Preprocessing

- Prepared and organized the image datasets for model training and evaluation.
- Implemented the image preprocessing pipeline required for model inference.
- Resized input images to **300 × 300 pixels**.
- Converted images into PyTorch tensors.
- Applied image normalization using standard ImageNet normalization values.
- Prepared the data pipeline for deep learning model training and evaluation.

### Model Development & Training

- Worked on the deep learning classification pipeline using **PyTorch**.
- Used **EfficientNet-B3** as the image classification architecture.
- Configured the model for 3-class classification.
- Trained and evaluated the model on the prepared datasets.
- Analyzed classification performance and model predictions.

---

## ✨ Features

- 🧠 Deep learning-based image classification
- 🏗️ EfficientNet-B3 architecture
- 🖼️ Support for Chest X-ray, CT Scan, and Histopathological images
- 🔍 Automatic medical image validation
- 🔄 Automatic image modality detection
- 📊 Prediction confidence score
- 📈 Probability distribution across all classes
- 🖥️ Interactive Gradio interface
- 🖼️ Annotated prediction visualization
- ⚡ GPU/CPU inference support
- 📋 Model and prediction information display

---

## 🧠 Model Architecture

The project uses **EfficientNet-B3** for image classification.

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B3 |
| Framework | PyTorch |
| Number of Classes | 3 |
| Input Size | 300 × 300 |
| Output Classes | Normal, Pneumonia, Lung Cancer |
| Model Weights | `best_model.pth` |

The model architecture is created using the `timm` library and configured
with three output classes.

---

## 📊 Model Performance

The model was evaluated on a test dataset containing approximately
**1,466 medical images**.

Based on the project evaluation, the model achieved approximately:

**~94–95% test accuracy**

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

> Performance may vary depending on the dataset split, preprocessing,
> model configuration, and evaluation conditions.

---

## 🔄 How It Works

```text
Medical Image Upload
        ↓
Medical Image Validation
        ↓
Image Modality Detection
        ↓
Image Preprocessing
        ↓
Resize to 300 × 300
        ↓
Tensor Conversion & Normalization
        ↓
EfficientNet-B3
        ↓
Class Probability Prediction
        ↓
Predicted Class + Confidence
        ↓
Visualization & Results
```

---

## 🔍 Image Processing Pipeline

The uploaded image goes through the following processing pipeline:

1. The uploaded image is converted to RGB format.
2. The application checks whether the image matches expected medical-image
   characteristics.
3. The image modality is automatically estimated as Chest X-ray, CT Scan,
   or Histopathological.
4. The image is resized to **300 × 300 pixels**.
5. The image is converted into a PyTorch tensor.
6. ImageNet-style normalization is applied.
7. The processed image is passed to the trained EfficientNet-B3 model.
8. Class probabilities are calculated.
9. The predicted class and confidence score are displayed.

---

## 🖥️ Application Interface

The application is built using **Gradio** and provides an interactive
interface for image analysis.

Users can upload supported medical images and view:

- Predicted class
- Confidence score
- Probability distribution
- Detected image modality
- Annotated visualization
- Model information

### Application Screenshot

Add your actual screenshot here after taking one:

```markdown
![LungAI Application](images/interface.png)
```

### Prediction Result

Add a screenshot of an actual prediction:

```markdown
![LungAI Prediction Result](images/prediction.png)
```

---

## 🖼️ Supported Image Types

| Image Type | Supported |
|------------|-----------|
| Chest X-ray | ✅ |
| CT Scan | ✅ |
| Histopathological Image | ✅ |

The application uses image characteristics to automatically estimate the
uploaded image modality.

---

## 📍 Prediction Visualization

The application provides an annotated version of the uploaded image along
with the classification result.

The visualization highlights predefined regions associated with the
predicted class for demonstration purposes.

> The highlighted regions are visualization overlays and should not be
> interpreted as clinically validated lesion localization or a medical
> heatmap.

---

## 📂 Project Structure

```text
Lung-Diagnostic-Tool/
│
├── Lung_Diagnostic_App.py
├── best_model.pth
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kushaldhakate/Lung-Diagnostic-Tool.git
cd Lung-Diagnostic-Tool
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

The project uses:

```text
torch
torchvision
timm
gradio
numpy
Pillow
```

---

## ▶️ Running the Application

Make sure `best_model.pth` is present in the project directory.

Run:

```bash
python Lung_Diagnostic_App.py
```

The Gradio application will start on port `7860`.

The application can be accessed locally at:

```text
http://127.0.0.1:7860
```

The application is configured to support sharing through Gradio when
running the script.

---

## 💻 Hardware Support

The application automatically uses:

- **CUDA GPU** when available
- **CPU** otherwise

The model is loaded onto the available device during application startup.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Application and machine learning development |
| PyTorch | Deep learning model development and inference |
| EfficientNet-B3 | Image classification architecture |
| torchvision | Image preprocessing and transformations |
| timm | EfficientNet model implementation |
| Gradio | Interactive web interface |
| NumPy | Numerical and image array operations |
| Pillow | Image processing and annotation |

---

## 🚀 Future Improvements

- Improve model generalization using larger and more diverse datasets.
- Perform additional hyperparameter tuning.
- Add more lung disease categories.
- Improve model explainability using techniques such as Grad-CAM.
- Add more comprehensive evaluation and error analysis.
- Deploy the application as a cloud-based service.
- Improve modality-specific preprocessing.

---

## 🤝 Team Project

LungAI was developed as a collaborative project by a **3-member team**.

The project involved multiple areas including:

- Data preprocessing
- Model training and evaluation
- Application development
- Model integration
- Testing

My primary contribution was **data preprocessing and model training**.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

It is not a medical device and should not be used for diagnosis, treatment,
or clinical decision-making.

The predictions generated by the model may be incorrect and should not be
considered a substitute for evaluation by a qualified healthcare
professional.

The visualization and clinical information displayed by the application
are intended for demonstration and educational purposes and are not
clinically validated.

---
