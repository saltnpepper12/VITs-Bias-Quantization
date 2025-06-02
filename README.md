# 🧑‍🤝‍🧑 FairFace ViT/Swin Bias & Quantization Analysis 🤖

## 📚 Project Description

This project investigates the impact of **INT8 quantization** on model bias and accuracy for Vision Transformer (ViT) and Swin Transformer models trained on the FairFace dataset. The workflow covers model training, ONNX export, post-training quantization, and systematic validation. The main aim is to compare the fairness and performance of baseline (FP32) and INT8-quantized models across demographic groups, using robust, reproducible pipelines on Modal cloud infrastructure.

## 🗂️ Directory Structure & File Overview

```
.
├── Final-output-CSV-Files/              # 📊 Validation results
│   ├── VIT-INT8-Quantized-Validation.csv
│   ├── Swin-INT8-Quantized-Validation.csv
│   ├── Swin-Base-validation.csv
│   └── VIT-BASE-validation.csv
├── model-files/                         # 🧩 Model files
│   ├── base-model-onnx-files/
│   │   ├── vit_fairface_best.onnx
│   │   └── swin_fairface_best.onnx
│   ├── INT8-Model-Quantized-Files/
│   │   ├── swin_fairface_best_int8.onnx
│   │   └── vit_quantized_model.onnx
│   └── base-model-pth-files/
│       ├── swinv2_fairface_best.pth
│       └── vit_fairface_best.pth
├── sample-test-pictures/                # 🖼️ Sample images
│   └── [Sample images for quick testing]
├── calibration_images/                  # 🏷️ Calibration images
│   └── calib_XXX.jpg
├── INT8-Quant-VIT-Swin.ipynb            # 📒 Quantization notebook
├── validation-quantized-models.ipynb    # 📒 Validation notebook
├── vit-fine-tuning.ipynb                # 📒 ViT fine-tuning
├── swin-fine-tuning.ipynb               # 📒 Swin fine-tuning
├── Swin-local-inference.ipynb           # 📒 Swin local inference
├── vit-local-inference.ipynb            # 📒 ViT local inference
├── export_swin_onnx.py                  # 📝 Swin ONNX export script
├── extract_calibration_images.py        # 📝 Calibration image extraction
├── modal_dataset.py                     # 📝 Modal dataset utility
└── __pycache__/
```

## 📁 File & Folder Details

### 🧩 **Model Files**
- `model-files/base-model-pth-files/`: PyTorch checkpoints for the best ViT and Swin models (`.pth`).
- `model-files/base-model-onnx-files/`: Baseline (FP32) ONNX exports of the best models.
- `model-files/INT8-Model-Quantized-Files/`: INT8-quantized ONNX models for ViT and Swin.

### 🏷️ **Calibration Images**
- `calibration_images/`: 100+ images used for static quantization calibration.

### 🖼️ **Sample Test Images**
- `sample-test-pictures/`: Example images for quick inference and demo.

### 📊 **Output CSVs**
- `Final-output-CSV-Files/`: Validation results for all models, including top-1/top-5 predictions and probabilities.

### 📒 **Notebooks**
- `vit-fine-tuning.ipynb`: Fine-tuning ViT on FairFace, saving best `.pth` model.
- `swin-fine-tuning.ipynb`: Fine-tuning Swin on FairFace, saving best `.pth` model.
- `INT8-Quant-VIT-Swin.ipynb`: Quantization workflows for both ViT and Swin (static/dynamic, INT8).
- `validation-quantized-models.ipynb`: Validates quantized ONNX models, generates detailed CSVs.
- `vit-local-inference.ipynb`: Inference and validation for baseline ViT model.
- `Swin-local-inference.ipynb`: Inference and validation for baseline Swin model.

### 📝 **Scripts**
- `export_swin_onnx.py`: Exports Swin PyTorch model to ONNX.
- `extract_calibration_images.py`: Extracts images from the dataset for calibration.
- `modal_dataset.py`: Modal utility for dataset management.

## 🚀 **How to Use**

1. **Train & Export Models:**  
   Use `vit-fine-tuning.ipynb` and `swin-fine-tuning.ipynb` to train and save `.pth` models.  
   Export to ONNX using `export_swin_onnx.py` or similar for ViT.

2. **Quantize Models:**  
   Use `INT8-Quant-VIT-Swin.ipynb` to quantize ONNX models to INT8 using calibration images.

3. **Validate & Analyze:**  
   Use `validation-quantized-models.ipynb` (and local inference notebooks) to run validation, generate CSVs, and compare accuracy and bias.

4. **Results:**  
   All validation results are in `Final-output-CSV-Files/`, with top-1/top-5 predictions and probabilities for each sample.

## 🎯 **Project Aim**

The main goal is to **analyze and compare bias and accuracy between baseline and INT8-quantized ViT/Swin models** on the FairFace dataset, providing insight into the fairness and deployment-readiness of quantized vision transformers.

---

**For more details, see the individual notebooks and scripts.**  
Let us know if you need help running or extending the analysis! 😊
