# Unknown-Aware Multi-Label OOD Detection 

Official implementation of the paper: **"Unknown-Aware Multi-Label Learning for Enhanced Out-of-Distribution Detection"** (to appear).

---

## Overview
This repository addresses the critical challenge of **multi-label Out-of-Distribution (OOD) detection**, where traditional methods like JointEnergy suffer from an **imbalanced decision boundary** that misclassifies minority-class ID samples as OOD. To overcome this, we propose a novel framework leveraging **auxiliary Outlier Exposure (OE)** to reshape the energy-based uncertainty landscape. Our key contributions include:
- 🚀 **Energy Score Optimization**: Separately optimizes energy scores for tail ID samples and unknown OOD samples to expand their distribution gap.
- 🔍 **OE Dataset Selection**: A simple yet effective strategy to identify informative OE datasets for improved generalization.
- 🏆 **State-of-the-Art Performance**: Validated on multiple multi-label datasets with diverse OOD benchmarks.

![Framework](overview.png) <!-- Add a diagram if available -->

---

## Key Features
- **Joint Energy Optimization**: Mitigates decision boundary ambiguity for minority classes by explicitly modeling energy gaps.
- **Uncertainty Calibration**: Balances ID classification confidence and OOD detection robustness through energy score disentanglement.
- **OE-Aware Training**: Enhances model discrimination ability without negative transfer from imbalanced ID learning.

---

## Results
Our method achieves consistent improvements over existing multi-label OOD detection baselines. For detailed metrics and comparisons, refer to the [paper](https://arxiv.org/abs/2412.07499).

---

## Installation
```bash
git clone https://github.com/yourusername/multi-label-ood.git
cd multi-label-ood
pip install -r requirements.txt
