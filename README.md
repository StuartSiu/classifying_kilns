# Classifying Kilns

Course: STAT 440, Fall 2025

Instructor: Lloyd Elliott

Team Members: Stuart Siu, Elysa Lin, Geoffrey Ze-Yu Gao, Varinder Singh, Min Kaung Khant

## Academic Integrity Notice
To comply with Simon Fraser University's academic integrity guidelines, the source code (kilns_1.6.py) and its outputs have been archived in a password-protected RAR file. 

## Project Overview
This project focuses on identifying and classifying brick kilns from satellite imagery in Bangladesh. Traditional brick kilns are a major source of air pollution, which is a leading cause of death in the region. The Bangladeshi government is mandating the replacement of traditional fixed chimney kilns with more modern, less polluting zigzag kilns.  The objective of this classification model is to accurately differentiate between zigzag kilns (Class 1) and fixed chimney kilns (Class 0) using satellite images to assist in monitoring compliance and reducing pollution.  

## Dataset
- Training Set: 1,617 labeled 256 x 256 RGB satellite images.
- Testing Set: 724 unlabeled images used for final evaluation.
- Evaluation Metric: Area Under the Receiver Operating Characteristic Curve (AUC).  

## Methodology & Pipeline
Given that kilns occupy only a small fraction (~5%) of the image and appear at random locations, our pipeline heavily emphasizes automated localization and region-of-interest extraction prior to classification.  
1. Preprocessing & Localization
   - Redness Map: We isolated the brick-red tone of the kilns by calculating heat = r - 0.333*g - 0.333*b, which highlights the kilns while suppressing water and vegetation.
   - Sobel Edge Detection: To eliminate false positives from fields or roads, we generated a structural map using a custom TensorFlow Sobel edge implementation.
   - Fused Heatmap & Cropping: The redness and edge maps were combined using a weighted formula (w_edge = 0.6). A Gaussian-weighted sliding window was then applied to locate the highest-scoring 128 x 128 crop containing the kiln.
<img width="33%" alt="diag_K0919" src="https://github.com/user-attachments/assets/5875b8ab-8870-4754-803c-bd5500da4f32" />
<img width="33%" alt="diag_K0391" src="https://github.com/user-attachments/assets/b5c96a86-3c2d-43c4-b8f8-eb5011ea32c1" />

2. Model Architecture
   - Backbone: EfficientNet pre-trained on ImageNet.
   - Inputs: The network receives three distinct inputs for every image:
     - The localized 128 x 128 RGB crop (processed by EfficientNet).
     - A small auxiliary convolutional branch applied to the 128 x 128 fused heatmap.
     - A 3-dimensional global shape feature vector (aspect ratios and fill fraction).
   - Training Mechanics: The model was trained using binary cross-entropy with label smoothing. We utilized the Adam optimizer and heavily combated overfitting through spatial jitter, contrast changes, and mixup augmentation applied to both images and features.
4. Validation & Ensembling
   - To maximize robustness, we utilized Stratified 10-Fold Cross-Validation. Early stopping was implemented (monitoring validation AUC) to optimize the highly computationally intensive training process. Final predictions were generated using Test-Time Augmentation (TTA), averaging predictions across multiple jittered crops and across all folds.

## Results
- Baseline CNN: 20.5% AUC.
- Single EfficientNet (Heatmap Guided): 48.9% AUC.
- Final Multi-Input Ensemble: 99.93% AUC.

<img width="40%" alt="fold1_auc" src="https://github.com/user-attachments/assets/2bb38684-872b-419d-b985-f69d026f8f90" />
<img width="40%" alt="fold9_auc" src="https://github.com/user-attachments/assets/30d7404a-789f-4bc6-81b4-988e2ac3d24f" />


## Future Improvements
If this project were to be extended for production use, the following techniques could further improve robustness:
- Domain-Specific Pretraining: Utilizing self-supervised learning (e.g., SimCLR) on aerial datasets like GeoNet instead of relying solely on ImageNet weights.
- Multi-Crop Modeling: Applying Multiple-Instance Learning (MIL) to evaluate multiple high-scoring regions rather than just the single highest-scoring crop.
- Architecture Upgrades: Swapping the EfficientNet backbone for ConvNeXt or Vision Transformers (ViT) for stronger global-context representation.
- Learned Heatmap Generation: Training a small U-Net to dynamically generate the defect-likelihood map rather than using fixed RGB/Sobel rules.  

## Dependencies
Python 3.13, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, PIL/Image & Matplotlib
