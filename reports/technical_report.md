# Sparse Point-Supervised Remote Sensing Segmentation using Partial Cross Entropy Loss

## 1. Introduction

Semantic segmentation is a computer vision task where every pixel in an image is assigned a semantic class label. In remote sensing applications, semantic segmentation is widely used for:

- road extraction
- land cover analysis
- urban planning
- environmental monitoring
- agricultural analysis

Traditional semantic segmentation models require dense pixel-level annotations for training. However, generating full segmentation masks for remote sensing imagery is expensive and time-consuming because satellite images are typically very large and highly detailed.

This project explores sparse point-supervised semantic segmentation, where only a small subset of pixels are labelled while the remaining pixels remain unknown.

The objective of this assessment is to:
- simulate sparse point annotations
- train a segmentation model using Partial Cross Entropy Loss
- evaluate how sparse supervision affects learning behaviour

The implementation focuses on creating a lightweight and reproducible training pipeline suitable for CPU-only experimentation.

---

# 2. Methodology

## 2.1 Problem Formulation

In a fully supervised segmentation setup:

- every pixel contributes to training loss
- dense annotations are required

In sparse point-supervised learning:

- only selected labelled pixels contribute to optimization
- unlabeled pixels are ignored during training

The segmentation model still predicts a full segmentation map, but the loss function only computes gradients on sparse labelled locations.

This reduces annotation requirements while still allowing the model to learn spatial segmentation patterns.

---

# 3. Dataset

## 3.1 Selected Dataset

The DeepGlobe Land Cover Classification Dataset was used for experimentation.

Dataset characteristics:
- high-resolution satellite imagery
- segmentation masks for land regions
- commonly used in remote sensing segmentation research

The dataset contains:
- satellite RGB images
- corresponding segmentation masks

For lightweight experimentation:
- images were resized to 128 × 128
- binary segmentation masks were generated

---

# 4. Sparse Point Supervision

## 4.1 Sparse Label Simulation

The original segmentation masks were converted into sparse point annotations.

Process:
1. convert segmentation masks into binary labels
2. randomly sample a small percentage of pixel locations
3. preserve labels only at selected locations
4. assign ignore label `-1` to all remaining pixels

Example:
- labelled pixels → valid training supervision
- unlabeled pixels → ignored during optimization

This simulates realistic sparse annotation settings where only limited human annotations are available.

---

## 4.2 Partial Cross Entropy Loss

Partial Cross Entropy Loss was implemented using:

```python
nn.CrossEntropyLoss(ignore_index=-1)
```

The `ignore_index=-1` mechanism ensures:
- sparse labelled pixels contribute to loss
- unlabeled pixels are excluded from optimization

This allows the model to learn using only sparse supervision.

---

# 5. Model Architecture

## 5.1 DeepLabV3

A DeepLabV3 semantic segmentation network from Torchvision was used.

Reasons for selection:
- reliable segmentation baseline
- lightweight implementation
- easy transfer learning support
- suitable for remote sensing segmentation

The final classifier layer was modified for binary segmentation:

- class 0 → background
- class 1 → foreground / road

---

## 5.2 Transfer Learning

Instead of training from scratch, pretrained DeepLabV3 weights were used as initialization.

Advantages:
- faster convergence
- improved feature extraction
- more stable optimization
- reduced training time on CPU

This setup is especially useful under limited computational resources.

---

# 6. Experimental Setup

## 6.1 Training Configuration

| Component | Value |
|---|---|
| Model | DeepLabV3 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 4 |
| Image Resolution | 128 × 128 |
| Device | CPU |
| Loss Function | Partial Cross Entropy |
| Ignore Label | -1 |

---

## 6.2 Sparse Supervision Experiments

Two sparse supervision settings were explored.

| Experiment | Point Ratio | Epochs |
|---|---|---|
| Experiment 1 | 0.05 | 3 |
| Experiment 2 | 0.15 | 5 |

---

# 7. Experimental Purpose and Hypothesis

## 7.1 Purpose

The purpose of the experiments was to analyze how sparse point density affects segmentation learning behaviour.

---

## 7.2 Hypothesis

Increasing the number of labelled sparse points should:
- provide stronger supervision
- improve segmentation learning

However:
- excessive sparse density may also introduce instability
- lightweight CPU-only training may require careful balancing

---

# 8. Experimental Results

## 8.1 Experiment 1 — Sparse Ratio 0.05

Results:
- stable optimization behaviour
- gradual reduction in training loss
- improved validation performance

Observations:
- sparse supervision successfully guided learning
- model generalized reasonably well under limited supervision

Final losses:
- Train Loss ≈ 0.44
- Validation Loss ≈ 0.37

---

## 8.2 Experiment 2 — Sparse Ratio 0.15

Results:
- stronger initial supervision
- faster early optimization
- unstable validation behaviour during later epochs

Observations:
- higher sparse density improved learning signal strength
- validation instability appeared after extended training

Final losses:
- Train Loss ≈ 0.40
- Validation Loss ≈ 0.91

---

# 9. Analysis and Discussion

The experiments demonstrate that sparse point supervision can effectively train semantic segmentation models even when only a small subset of labelled pixels is available.

Key findings:
- sparse labels still provide meaningful supervision
- Partial Cross Entropy Loss correctly ignores unlabeled regions
- sparse point density significantly affects optimization behaviour

Lower sparse ratios:
- weaker supervision
- more stable validation behaviour

Higher sparse ratios:
- stronger supervision
- faster learning
- potential instability under lightweight training conditions

The experiments also show that:
- pretrained segmentation backbones help stabilize sparse learning
- lightweight CPU-only training can still produce meaningful results
- practical engineering pipelines are important for reproducibility

---

# 10. Limitations

This implementation was intentionally lightweight due to:
- CPU-only training environment
- limited assessment timeline
- rapid experimentation constraints

Limitations include:
- short training duration
- limited hyperparameter tuning
- no advanced augmentation
- lightweight image resolution
- no mixed precision or GPU acceleration

Visual segmentation predictions remain limited due to these constraints.

---

# 11. Future Improvements

Several improvements could further enhance performance:

- longer training schedules
- stronger data augmentation
- Dice Loss or Focal Loss combinations
- larger segmentation backbones
- GPU acceleration
- mixed precision training
- learning rate scheduling
- better sparse point sampling strategies
- post-processing refinement

Future experiments could also evaluate:
- different sparse ratios
- alternative segmentation architectures
- semi-supervised approaches
- consistency regularization methods

---

# 12. Conclusion

This project successfully implemented sparse point-supervised semantic segmentation using Partial Cross Entropy Loss on remote sensing imagery.

The implementation demonstrated:
- sparse label simulation
- correct Partial Cross Entropy training
- end-to-end segmentation workflow
- lightweight experimentation pipeline

The experiments confirmed that segmentation models can still learn meaningful spatial representations from sparse point annotations.

Overall, the project validates the feasibility of sparse supervision for remote sensing semantic segmentation under practical engineering constraints.