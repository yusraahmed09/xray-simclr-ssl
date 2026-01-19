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
* **Explainability:** Integrated **Grad-CAM** visualizations to interpret model focus regions during classification.

## 🛠️ Methodology
The pipeline consists of two distinct stages:

### 1. Pre-training (Unsupervised)
We use the **SimCLR** framework to train the encoder without labels.
* **Backbone:** ResNet-50
* **Loss Function:** Normalized Temperature-scaled Cross Entropy (NT-Xent)
* **Batch Size:** [Insert your batch size, e.g., 64/128]
* **Optimizer:** Adam/LARS

### 2. Downstream Task (Supervised)
The pre-trained encoder is frozen (or fine-tuned), and a linear classifier is added on top to classify pathologies (e.g., Pneumonia, Cardiomegaly).

## 📊 Results
*Comparison of AUROC scores on the CheXpert validation set:*

| Model | Labeled Data Fraction | AUROC Score |
| :--- | :---: | :---: |
| **ResNet-50 (Supervised)** | 100% | 0.XX |
| **ResNet-50 (Supervised)** | 10% | 0.XX |
| **SimCLR (Ours)** | **1%** | **0.XX** |
| **SimCLR (Ours)** | **10%** | **0.XX** |

> *Note: Our SSL approach achieves comparable performance to supervised baselines using significantly less labeled data.*

## 📸 Visualization
Below is a Grad-CAM visualization showing the model's focus on the lung opacity regions:

![Grad-CAM Example](path/to/your/gradcam_image.png)
*(Place a screenshot here showing the input X-ray vs. the Heatmap)*

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/xray-simclr-ssl.git](https://github.com/yourusername/xray-simclr-ssl.git)
   cd xray-simclr-ssl
