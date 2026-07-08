# MediScan AI – Intelligent Disease Diagnosis System
# Disease Prediction using Deep Learning

A multi-disease diagnostic web app that uses deep learning to classify medical images across five conditions. Upload a scan, pick the relevant model, and get a prediction with a confidence score — served through a simple Flask interface.

## Overview

This project trains separate convolutional neural networks for five different medical imaging tasks and exposes them through a single web app. Instead of one general-purpose model, each disease has its own purpose-built classifier, since the imaging modality and visual cues differ significantly between, say, a brain MRI and a chest X-ray.

| Disease | Imaging Type | Classes | Input Size |
|---|---|---|---|
| Alzheimer's / Dementia | Brain MRI (OASIS dataset) | Non Demented, Very Mild, Mild, Moderate Dementia | 128×128 |
| Brain Tumor | Brain MRI | Glioma, Meningioma, Pituitary, No Tumor | 224×224 |
| Diabetic Retinopathy | Retinal Fundus Images | DR, No DR | 224×224 |
| Kidney Conditions | CT Scan | Cyst, Stone, Tumor, Normal | 224×224 |
| Respiratory Disease | Chest X-ray | Bacterial Pneumonia, Viral Pneumonia, COVID-19, Tuberculosis, Normal | 128×128 |

## How it works

1. The user selects a disease model and uploads an image through the web form.
2. The image is preprocessed (resized, normalized) to match the input shape the selected model was trained on.
3. The corresponding `.h5` Keras model runs inference and returns the predicted class along with a confidence percentage.
4. The result and the uploaded image are displayed back to the user.

## Model architectures

Each model was trained independently in its own notebook (see `Disease Predication Code/`):

- **Alzheimer's / Dementia** (`oasis.ipynb`) — a custom CNN built from scratch with Conv2D, BatchNormalization, MaxPooling, and Dropout layers.
- **Brain Tumor** (`brain-tumor-mri-dataset1.ipynb`) — transfer learning on **DenseNet169** with a custom classification head.
- **Diabetic Retinopathy** (`diagnosis-of-diabetic-retinopathy1.ipynb`) — transfer learning on **VGG19** (with an InceptionV3 variant also explored) with custom dense layers on top.
- **Kidney CT Scan** (`kidney-ct-scan.ipynb`) — transfer learning on **VGG16** with a custom classification head.
- **Respiratory Disease** (`respiratory-disease.ipynb`) — a custom CNN built from scratch, similar in structure to the dementia model.

All transfer-learning models freeze the pretrained convolutional base and train only the new dense layers on top, using data augmentation (rotation, shifts, zoom, horizontal flip) to improve generalization on relatively small medical imaging datasets.

### Explainability

`lime-code.ipynb` applies **LIME (Local Interpretable Model-agnostic Explanations)** to visualize which regions of an input image most influenced a model's prediction — useful for sanity-checking that the models are attending to clinically relevant areas rather than artifacts.

## Tech stack

- **Backend:** Flask
- **Deep learning:** TensorFlow / Keras
- **Image processing:** Pillow, TensorFlow image ops
- **Frontend:** HTML, Bootstrap 5
- **Model training/experimentation:** Jupyter notebooks (developed on Kaggle)
- **Explainability:** LIME

## Project structure

```
Disease-Prediction_DL/
├── app.py                          # Flask app: routes, model loading, inference
├── Disease Predication Code/       # Training notebooks (one per disease)
│   ├── oasis.ipynb                 # Alzheimer's / Dementia classifier
│   ├── brain-tumor-mri-dataset1.ipynb
│   ├── diagnosis-of-diabetic-retinopathy1.ipynb
│   ├── kidney-ct-scan.ipynb
│   ├── respiratory-disease.ipynb
│   └── lime-code.ipynb             # Model explainability with LIME
├── templates/
│   └── index.html                  # Upload form + results page
├── static/
│   ├── css/style.css
│   └── uploads/                    # Uploaded images saved here at runtime
└── README.md
```

## Getting started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/Disease-Prediction_DL.git
cd Disease-Prediction_DL
pip install flask tensorflow numpy pillow
```

### Model files

The trained `.h5` model files are not included in this repository due to size. Train them yourself using the notebooks in `Disease Predication Code/`, or download pretrained weights if you've hosted them separately (e.g. via Git LFS, Google Drive, or a release asset), then place them in a `model/` folder.

> **Note:** `app.py` currently points to absolute local paths (e.g. `C:\Users\...\model\BrainTumor (2).h5`). Update these to relative paths before running, for example:
> ```python
> "model": safe_load_model("model/BrainTumor.h5"),
> ```

### Run the app

```bash
python app.py
```

The app will be available at `http://localhost:5002`.

## Usage

1. Open the app in your browser.
2. Select a disease model from the dropdown (Alzheimer's, Brain Tumor, Diabetic Retinopathy, Kidney CT Scan, or Respiratory Disease).
3. Upload a matching medical image.
4. Click **Predict** to see the classification result and confidence score.

## Datasets

Models were trained on publicly available Kaggle datasets:

- OASIS Alzheimer's MRI dataset
- Brain Tumor MRI dataset
- Diagnosis of Diabetic Retinopathy dataset
- CT Kidney dataset (Normal, Cyst, Tumor, Stone)
- Lung Disease (4 types) chest X-ray dataset

## Disclaimer

This project is intended for educational and research purposes only. It is **not a medical device** and should not be used for actual clinical diagnosis or to inform real treatment decisions. Always consult a qualified healthcare professional for medical advice.

## License

Specify your license here (e.g. MIT, Apache 2.0).
