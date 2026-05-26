# Sparse Point-Supervised Remote Sensing Segmentation

Dense pixel-level annotation for semantic segmentation is expensive and time-consuming, especially in remote sensing imagery where satellite images are large and highly detailed.

This project explores sparse point-supervised semantic segmentation, where only a small subset of pixels are labelled while the remaining pixels are ignored during training.

A lightweight DeepLabV3 segmentation pipeline was implemented using:
- sparse point label simulation
- Partial Cross Entropy Loss
- transfer learning
- remote sensing satellite imagery

The model is trained to predict full segmentation maps while computing loss only on labelled sparse pixel locations using `ignore_index=-1`.

The project also includes experimental analysis on how sparse point density affects optimization behaviour and segmentation learning stability.

## Project Components

- `notebooks/` → full experimentation and training workflow
- `reports/` → detailed technical report and analysis
- `outputs/` → prediction visualizations and experiment outputs

## Core Concepts Explored

- Sparse supervision
- Semantic segmentation
- Partial Cross Entropy Loss
- Remote sensing imagery
- Transfer learning with DeepLabV3
- Lightweight CPU-based experimentation
