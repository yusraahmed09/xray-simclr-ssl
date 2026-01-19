# Self-Supervised Learning for Few-Shot Chest X-ray Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This repository implements a **Self-Supervised Learning (SSL)** pipeline designed to tackle the scarcity of labeled data in medical imaging. Using **SimCLR** (Simple Framework for Contrastive Learning), we pre-train a **ResNet-50** backbone on unlabeled Chest X-rays (CheXpert dataset) to learn robust visual representations.

The model is subsequently fine-tuned on a limited subset of labeled data (Few-Shot Learning), demonstrating that SSL can significantly reduce the annotation burden for medical diagnosis tasks.

## 🚀 Key Features
* **Contrastive Pre-training:** Implements SimCLR with a projection head to maximize similarity between augmented views of the same image.
* **Medical-Specific Augmentations:** Customized data augmentation pipeline (random rotation, crop, Gaussian blur) tuned for grayscale medical scans rather than natural images.
* **Few-Shot Adaptation:** Evaluates performance on 1% and 10% labeled data fractions compared to fully supervised baselines.

## 🛠️ Methodology
The pipeline consists of two distinct stages:

### 1. Pre-training (Unsupervised)
We use the **SimCLR** framework to train the encoder without labels.
* **Backbone:** ResNet-50
* **Loss Function:** Normalized Temperature-scaled Cross Entropy (NT-Xent)
* **Batch Size:** 64
* **Optimizer:** Adam/LARS

### 2. Downstream Task (Supervised)
The pre-trained encoder is frozen (or fine-tuned), and a linear classifier is added on top to classify pathologies (e.g., Pneumonia, Cardiomegaly).

## 📊 Experimental Results

We compared the performance of our SimCLR pre-trained model against two baselines: training from scratch (Random Weights) and standard transfer learning (ImageNet weights).

**Table 1: Peak Test Accuracy (%) Comparison**

| Model | Initialization Source | Cardiomegaly | Lung Opacity | Pl. Effusion | ECM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Scratch** | Random Weights | 46.15 | 45.30 | 44.02 | 50.85 |
| **ImageNet** | Natural Images | 57.69 | 52.56 | 49.15 | 61.54 |
| **SimCLR (Ours)**| **Unlabeled CXRs** | **66.24** | **63.68** | **62.39** | **73.93** |

> **Key Finding:** Pre-training on domain-specific medical data (SimCLR) yields a significant performance boost (+8-13% accuracy) compared to using generic natural image weights (ImageNet), validating the effectiveness of self-supervised learning for medical imaging.

> *Note: Our SSL approach achieves comparable performance to supervised baselines using significantly less labeled data.*

