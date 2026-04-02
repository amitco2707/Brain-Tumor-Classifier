# Brain Tumor Classifier

A deep learning project that classifies brain MRI scans as **tumor** or **no tumor** using a fine-tuned ResNet-18 model. Includes a full training pipeline and an interactive web app built with Streamlit.

---

## What it does

Upload a brain MRI image and the app tells you whether a tumor is detected — with a confidence percentage and color-coded result.

---

## Project structure

```
├── data/
│   ├── yes/              # MRI images with tumor
│   └── no/               # MRI images without tumor
│
├── data_loader/
│   └── dataset.py        # Dataset class + data augmentation + train/val/test split
│
├── model/
│   └── architecture.py   # ResNet-18 with a custom final layer for binary classification
│
├── training/
│   └── trainer.py        # Training loop (forward pass, backprop, validation)
│
├── evaluation/
│   └── evaluate.py       # Test-set evaluation, confusion matrix, training curves
│
├── outputs/              # Saved model weights + generated plots (git-ignored)
│
├── config.py             # All hyperparameters and paths in one place
├── main.py               # Run this to train the model
└── streamlit_app.py      # Run this to launch the web app
```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python main.py
```

This will:
- Print a dataset summary
- Train for 15 epochs
- Save the model to `outputs/brain_tumor_model.pth`
- Save training curves and a confusion matrix to `outputs/`

### 3. Launch the web app

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL in the terminal — open that in your browser.

> **This link only works on your own computer while the app is running.** It is not a public URL and cannot be opened from GitHub.

> You must train the model first — the app loads `outputs/brain_tumor_model.pth`.

---

## Model

- **Architecture:** ResNet-18 pretrained on ImageNet
- **Approach:** Transfer learning — all layers frozen except the final classification layer
- **Input size:** 224 × 224 RGB
- **Output:** 2 classes — Tumor / No Tumor

## Training settings

All settings are in `config.py`:

| Setting | Value |
|---|---|
| Epochs | 15 |
| Learning rate | 0.001 |
| Batch size | 16 |
| Optimizer | Adam |
| Train / Val / Test split | 70% / 15% / 15% |

## Dataset

253 brain MRI images split into two classes:
- **Yes** (tumor): 155 images
- **No** (no tumor): 98 images
