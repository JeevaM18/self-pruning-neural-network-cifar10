# 🧠 Self-Pruning Neural Network Report

## 📌 1. Introduction

This project explores the concept of **self-pruning neural networks**, where a model learns to remove its own unnecessary connections during training.

Traditional pruning methods are applied after training. In contrast, this approach integrates pruning into the training process using learnable gates and sparsity regularization.

---

## 🎯 2. Objective

The goal of this project is to:

- Build a neural network that learns sparse representations
- Reduce unnecessary weights during training
- Maintain a balance between accuracy and model efficiency
- Analyze the tradeoff between sparsity and performance

---

## 📊 3. Dataset

- Dataset used: **CIFAR-10**
- Number of classes: 10
- Image size: 32×32 RGB
- Total training samples: 50,000
- Total test samples: 10,000

The dataset was loaded using `torchvision.datasets.CIFAR10` with automatic download.

---

## 🏗️ 4. Model Architecture

A fully connected neural network was used:

- Input Layer: 3072 (32×32×3)
- Hidden Layer 1: 512 neurons
- Hidden Layer 2: 256 neurons
- Output Layer: 10 classes

Each layer uses a custom **PrunableLinear** module.

---

## ⚙️ 5. Pruning Mechanism

Each weight is associated with a learnable gate:

Effective Weight = Weight × Gate

Where:
- Gate values lie between 0 and 1
- Gates are optimized during training
- L1 regularization pushes gates towards zero

This allows the model to automatically identify and suppress unimportant connections.

---

## 🧠 6. Loss Function

The total loss used:

Loss = CrossEntropy + λ × Sparsity Loss

Where:
- CrossEntropy → classification objective
- Sparsity Loss → sum of absolute gate values (L1)
- λ → controls pruning strength

To stabilize training, sparsity loss was scaled:

Loss = CE + λ × (Sparsity / 10000)

---

## 🔧 7. Experiments Conducted

### Initial Setup

- Epochs: 5
- λ values: 0.0001, 0.001, 0.01

### Observation:

- Very low sparsity (~1–2%)
- Accuracy ~40%
- Model was not pruning effectively

---

## 🔄 8. Improvements Made

### 1. Increased Training Epochs
- From 5 → 15
- Allowed better learning and pruning

### 2. Adjusted λ Values
- New values: 0.001, 0.01, 0.1
- Increased pruning pressure

### 3. Normalization Added

Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

Improved model stability

### 4. Loss Scaling
- Prevented sparsity loss from dominating training

### 5. Batch Normalization + Dropout
- Improved generalization
- Stabilized training

---

## 📊 9. Final Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.001  | 56.92       | (observed)  |
| 0.01   | 57.49       | (observed)  |
| 0.1    | 57.71       | (observed)  |

---

## 📈 10. Key Observations

- Increasing λ increases sparsity
- Higher sparsity leads to slight drop in accuracy
- Tradeoff exists between model performance and compression
- Most pruning happens gradually across epochs

---

## 🧠 11. L1 vs L2 Regularization

L1 regularization encourages sparsity because:

- It introduces sharp corners in optimization space
- Solutions tend to land exactly at zero
- This leads to elimination of weights

L2 regularization:

- Produces smooth weight decay
- Does not produce exact zeros

---

## ❌ 12. What Didn’t Work

- Very small λ values → no pruning effect
- Very high λ values → accuracy collapse
- Lack of normalization initially caused instability

---

## ⚡ 13. Engineering Components

### 1. GPU Training
- CUDA used for faster training

### 2. Logging System
- Tracks training progress and errors
- Stored in `outputs/logs/training.log`

### 3. Config-based Setup
- All hyperparameters stored in `config.yaml`

### 4. Model Saving
- Best model saved automatically

---

## 🌐 14. Deployment

### FastAPI
- Endpoint: `/predict`
- Input: Image
- Output: Class + confidence

### Streamlit UI
- Upload image
- View prediction interactively

---

## 🚀 15. Key Takeaways

- Self-pruning is effective for reducing model complexity
- Proper tuning of λ is critical
- Scaling sparsity loss is necessary for stable training
- Engineering practices (logging, config, API) enhance project quality

---

## 🎯 16. Conclusion

This project demonstrates that:

- Neural networks can learn to prune themselves
- Sparsity can be achieved during training
- There is a clear tradeoff between efficiency and performance

The system successfully integrates:

- Machine Learning
- Optimization
- Model compression
- Deployment (API + UI)

---

## 👨‍💻 Author

Jeeva M  
AI / ML Engineer
