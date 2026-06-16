# Rheumatoid Arthritis Severity Classification from Hand X-Ray Images

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Library](https://img.shields.io/badge/library-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/library-OpenCV-green.svg)](https://opencv.org/)

An end-to-end medical deep learning project for classifying the severity of Rheumatoid Arthritis (RA) from hand X-ray images. The system features a custom hybrid **CNN-LSTM** architecture and provides a user-friendly, web-based interface built with **Flask** for real-time diagnostic inference.

---

## 📌 Project Overview
Rheumatoid Arthritis (RA) is a chronic inflammatory disorder affecting joints. Early and accurate detection of cartilage wear, joint space narrowing, and bone changes is critical. This project implements a hybrid deep learning model to categorize X-ray scans of hands into **five severity classes**:
1. **Normal** - Joint structure appears healthy with no signs of arthritis.
2. **Doubtful** - Possible early-stage arthritis with minor joint space narrowing.
3. **Mild** - Slight cartilage wear and minor bone spurs.
4. **Moderate** - Noticeable joint space reduction and structural bone changes.
5. **Severe** - Significant cartilage loss, bone damage, and joint deformity.

---

## ⚙️ Model Architecture & Workflow

The classification engine relies on a hybrid **CNN-LSTM** network:

```
[Input X-Ray Image] ──> [OpenCV Preprocessing]
                               │
                               ▼
                    [VGG16 Feature Extractor] (Pre-trained on ImageNet)
                               │
                               ▼
                    [LSTM Sequence Classifier] (256 -> 128 -> 64 units)
                               │
                               ▼
                    [Dense Softmax Layer] (5 output classes)
                               │
                               ▼
                     [Predicted Diagnosis] (Severity & Confidence)
```

1. **Feature Extraction (CNN):** A pre-trained **VGG16** (without the top dense layers) serves as a frozen feature extractor, mapping spatial image patterns into high-dimensional feature vectors.
2. **Sequence Processing (LSTM):** Features are reshaped and fed into a multi-layered **LSTM** network (256, 128, and 64 units) to model structural/hierarchical joint representations.
3. **Classification:** Dense layers with Dropout (for regularization) and Batch Normalization process the LSTM outputs, producing the final severity classification using a Softmax activation.

---

## 📈 Model Performance & Metrics

The model achieves state-of-the-art results on the test partition:

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | **97.90%** |
| **Test Accuracy** | **91.80%** |
| **Weighted Precision** | **91.99%** |
| **Weighted Recall** | **91.80%** |
| **Weighted F1-Score** | **91.83%** |

*Optimization strategies used: `ReduceLROnPlateau` learning rate decay, Dropout (0.5/0.3/0.2) to prevent overfitting, and Batch Normalization.*

---

## 💻 Web Application Features

The interactive web interface is designed with a Flask backend to facilitate seamless integration:
* **Real-time Image Upload:** Upload custom hand X-rays in common formats (`.jpg`, `.png`, `.webp`).
* **Instant Preprocessing:** Automatic scaling, pixel normalization (division by 255.0), and RGB conversion via OpenCV.
* **Live Inference:** Instant calculation of predicted severity stage alongside classification confidence percentages.

---

## 📂 Project Structure

```directory
Rheumatoid-arthritis/
│
├── app/
│   ├── static/                    # Asset files (CSS, JS, images)
│   │   ├── uploaded_images/       # Temp directory for uploaded images
│   │   └── style.css              # Custom UI styling
│   │
│   ├── templates/
│   │   └── result.html            # Main web page GUI
│   │
│   ├── app.py                     # Flask server and inference routing
│   ├── cnn_lstm_model.keras       # Trained classification model weights
│   └── model-cnn+lstm.ipynb       # Jupyter notebook of the training pipeline
│
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── requirements.txt               # List of dependencies
```

---

## 🚀 Setup & Execution Instructions

Follow these steps to run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Abhitkumar89/BTP.git
cd Rheumatoid-arthritis
```

### 2. Create and Activate a Virtual Environment (Recommended)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web Application
```bash
python app/app.py
```

### 5. Access the Web GUI
Open your web browser and navigate to:
```url
http://127.0.0.1:5000/
```
*(Upload your X-ray image and hit Submit to run diagnostics!)*
