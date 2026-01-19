Self-Supervised Pre-training for Few-Shot X-ray Classification
This project implements a Self-Supervised Learning (SSL) pipeline to address the challenge of label scarcity in medical imaging. By leveraging SimCLR (Simple Framework for Contrastive Learning of Visual Representations), we pre-train a ResNet-50 encoder on unlabeled Chest X-ray data to learn robust visual representations without manual annotation.

Key Features
Contrastive Learning: Utilizes SimCLR to maximize agreement between differently augmented views of the same X-ray image.

Few-Shot Adaptation: Fine-tunes the pre-trained encoder on a limited subset of labeled data (1% - 10%) to achieve high classification performance.

Medical Augmentations: Implements domain-specific data augmentations tailored for grayscale medical scans.

Interpretability: Includes Grad-CAM visualization to highlight the regions of the X-ray driving the model's predictions.

Tech Stack
Frameworks: PyTorch, Torchvision

Model: ResNet-50 (Backbone), SimCLR (Projection Head)

Dataset: CheXpert / ChestX-ray14
