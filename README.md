# 🧠 NeuroScan AI — Brain Tumor Detection & Diagnostics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

**NeuroScan AI** is an end-to-end deep learning medical diagnostics platform powered by a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras**. It accurately classifies brain Magnetic Resonance Imaging (MRI) scans into **Tumorous** and **Healthy (Non-Tumorous)** tissue, accompanied by an interactive real-time web application.

---

## 📸 Screenshots & Diagnostic Visuals

### 1. NeuroScan AI Web Application
An interactive, glassmorphism dark-mode web application providing real-time MRI scanning, confidence breakdown, and sample testing:

![NeuroScan AI Web Application](web_app_screenshot.png)

### 2. Comprehensive Model Evaluation & Prediction Dashboard
Performance metrics, confusion matrix heatmap, training history curves, and sample test predictions:

![Brain Tumor Detection Output Dashboard](brain_tumor_detection_output.png)

---

## 🎯 Key Features

- **High Tumor Sensitivity (87.5% Recall)**: Minimizes false negatives—critical in clinical AI diagnostics.
- **Real-Time Interactive Web Dashboard**:
  - Drag-and-drop or upload any brain MRI image (JPG, PNG, DICOM-converted).
  - 1-click testing with pre-loaded Kaggle MRI test samples.
  - Animated dual probability bars showing exact Tumor vs. Healthy confidence scores.
- **Production-Ready Deployment**: Includes `Dockerfile`, `.dockerignore`, and `requirements.txt` for 1-click deployment on Render, Railway, Hugging Face Spaces, or AWS.
- **Fast In-Memory Inference**: Optimized pipeline processes scans in `<100ms`.

---

## 🏗️ Neural Network Architecture

The model processes standard $224 \times 224 \times 3$ axial brain MRI scans through a hierarchical convolutional feature extraction pipeline:

![Neural Network Architecture](convnet_architecture.jpg)

### Architecture Pipeline:
1. **Data Augmentation Layer**: Random horizontal flip, subtle rotation ($0.08$), and zoom ($0.10$) to prevent overfitting on limited medical samples.
2. **Rescaling Layer**: Normalizes raw pixel intensities from $[0, 255]$ into $[0.0, 1.0]$.
3. **Convolutional Block 1**: 32 filters ($3 \times 3$, ReLU) + MaxPooling2D.
4. **Convolutional Block 2**: 64 filters ($3 \times 3$, ReLU) + MaxPooling2D.
5. **Convolutional Block 3**: 128 filters ($3 \times 3$, ReLU) + MaxPooling2D.
6. **Convolutional Block 4**: 256 filters ($3 \times 3$, ReLU) + MaxPooling2D.
7. **Dense Classifier**: Flatten $\rightarrow$ Dense (128 units, ReLU) $\rightarrow$ Dropout ($0.50$) $\rightarrow$ Dense (1 unit, Sigmoid).

---

## 🔬 How Detection Works

The final output neuron uses a **Sigmoid activation function** producing a continuous probability score $P \in [0.0, 1.0]$:

```
[Brain MRI Image]
       │
       ▼ (Resize to 224x224 & Normalize)
[4-Block CNN Feature Extractor]
       │
       ▼ (Dense 128 + Dropout 0.5)
[Sigmoid Activation Neuron]
       │
       ├── Score >= 0.50  ➔  TUMOR DETECTED       (Confidence = P * 100%)
       └── Score <  0.50  ➔  NO TUMOR (HEALTHY)   (Confidence = (1 - P) * 100%)
```

---

## 📊 Model Evaluation & Metrics

Evaluated on an untouched test dataset of 40 MRI scans (24 tumorous, 16 healthy):

| Metric | Score | Clinical Importance |
| :--- | :---: | :--- |
| **Test Accuracy** | **77.50%** | Overall diagnostic correctness |
| **Recall / Sensitivity** | **87.50%** | **Catches 21 of 24 tumors** (low false-negative rate) |
| **Precision** | **77.78%** | High reliability when flagging positive tumors |
| **F1-Score** | **82.35%** | Harmonic balance between precision and recall |

### Confusion Matrix
```
                 Predicted: Healthy    Predicted: Tumor
Actual Healthy:          10                    6
Actual Tumor:             3                   21
```

---

## 💻 How the Web Page Was Built

The web interface ([`app.py`](app.py)) is engineered to be **lightweight, fast, and dependency-minimal**:
- **Backend**: Built with Python's native `http.server`, eliminating heavyweight framework overhead. It keeps `brain_tumor_best.keras` resident in memory for instantaneous inference.
- **Frontend**: Clean Vanilla HTML5, modern CSS3 (custom CSS variables, glassmorphism, responsive grid), and Vanilla JavaScript.
- **Real-Time API**:
  - `GET /api/samples`: Fetches sample MRI metadata and base64 thumbnails.
  - `GET /api/predict_sample?id=<id>`: Executes immediate inference on preloaded samples.
  - `POST /api/predict_upload`: Accepts base64 encoded user uploads for instant diagnostics.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/manoj-lokireddy/NeuroScan-AI.git
cd NeuroScan-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Interactive Web Application
```bash
python app.py
```
Open **`http://localhost:5050`** in your browser.

### 4. Run CLI Prediction on a Single Scan
```bash
# Tumorous sample:
python predict.py yes/Y1.jpg

# Healthy sample:
python predict.py "no/1 no.jpeg"
```

### 5. Evaluate Model on Test Dataset
```bash
python evaluate.py
```

---

## 🐳 Docker Deployment

Build and run anywhere with Docker:

```bash
# Build Docker image
docker build -t neuroscan-ai .

# Run container
docker run -p 5050:5050 neuroscan-ai
```
Visit `http://localhost:5050`.

---

## 📁 Repository Structure

```text
├── app.py                           # Interactive Web Application server
├── predict.py                       # CLI inference script for single MRI scans
├── evaluate.py                      # Test dataset evaluation script
├── train.py                         # CNN training script with data augmentation
├── generate_visual_output.py        # Generates comprehensive evaluation board
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container deployment definition
├── Procfile                         # PaaS deployment process configuration
├── models/
│   ├── brain_tumor_best.keras       # Best trained CNN model weights
│   ├── accuracy_curve.png           # Training vs validation accuracy plot
│   └── loss_curve.png              # Training vs validation loss plot
├── dataset/                         # Train, validation, and test splits
├── yes/                             # Tumorous Brain MRI sample scans
├── no/                              # Non-tumorous Brain MRI sample scans
├── web_app_screenshot.png           # Web application UI preview
└── brain_tumor_detection_output.png # Diagnostic evaluation chart
```

---

## 📜 License
Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for more information.
