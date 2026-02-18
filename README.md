# U-Net Segmentation of Electron Microscopy Images

## Objective
To implement a convolutional neural network for membrane segmentation in EM images.

## Dataset
ISBI 2012 EM segmentation dataset.

## Method
- Grayscale normalization
- U-Net architecture
- BCEWithLogitsLoss
- Adam optimizer

## Evaluation
Dice score metric used to quantify overlap.

## Results
Dice Score ≈ 0.78 (after 5 epochs)

## Relevance
Demonstrates computational microscopy pipeline applicable to cryo-electron tomography datasets.
