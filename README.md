# EDGE: Unknown-aware Multi-label Learning by Energy Distribution Gap Expansion
Official PyTorch implementation for "EDGE: Unknown-aware Multi-label Learning by Energy Distribution Gap Expansion"

This repository contains the implementation of an **unknown-aware multi-label learning framework** for multi-label Out-Of-Distribution (OOD) detection. The proposed method addresses the imbalance problem in OOD detection, particularly when the model lacks sufficient discrimination ability. By leveraging auxiliary outlier exposure (OE) and reshaping the uncertainty energy space layout, our framework improves the detection of OOD samples while preserving the discrimination of tail In-Distribution (ID) samples.

---

## Overview

Multi-label OOD detection aims to distinguish OOD samples from multi-label In-Distribution (ID) samples. Unlike single-label OOD detection, it is crucial to model the joint information among classes. However, existing methods like **JointEnergy** can produce an imbalance problem, where samples related to minority classes are often misclassified as OOD due to ambiguous energy decision boundaries. Additionally, traditional imbalanced multi-label learning methods are not suitable for OOD detection scenarios and may even introduce negative transfer effects.

To address these challenges, we propose:
1. An **unknown-aware multi-label learning framework** that separately optimizes energy scores for tail ID samples and unknown samples, expanding the energy distribution gap between them.
2. A simple yet effective measure to select more informative OE datasets for better OOD detection performance.

---

## Key Features

- **Reshaping the Energy Space Layout**: Our framework optimizes the energy scores for tail ID samples and OOD samples, ensuring that tail ID samples have significantly larger energy scores than OOD samples.
- **Auxiliary Outlier Exposure (OE)**: We leverage OE to improve the model's ability to discriminate between ID and OOD samples.
- **Informative OE Dataset Selection**: A novel measure is designed to select more informative OE datasets, enhancing the effectiveness of the framework.

---

## Installation

To set up the environment and install dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Yuchen-Sunflower/EDGE.git
   cd EDGE
2. Install the required dependencies:
   '''bash
   pip install -r requirements.txt
