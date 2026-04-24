# 🧠 Self-Pruning Neural Network

## CIFAR-10 Image Classification with Dynamic Pruning

<p align="center">
  <img src="assets/A1.png" width="900"/>
</p>

---

# 🚀 Overview

This project implements a self-pruning neural network that dynamically learns to remove unnecessary weights during training using L1-based sparsity regularization.

Unlike traditional pruning (post-training), this approach allows the model to:

- Learn which connections are important
- Suppress irrelevant weights during training
- Achieve a balance between accuracy and sparsity

---

# 🎯 Problem Statement

Deep neural networks often:

- Contain redundant parameters  
- Consume high memory  
- Have unnecessary computational cost  

Goal:

Build a model that:
- Maintains performance  
- Automatically reduces its own complexity  

---

# 🧠 Core Idea

Effective Weight = Weight × Gate

- Gate ∈ [0,1]
- L1 regularization pushes gates → 0

This results in automatic pruning.

---

# 📊 Dataset

- Dataset: CIFAR-10
- Classes: 10
- Images: 32×32 RGB

Loaded using torchvision.

---
# 🏗️ Project Structure

```bash
self-pruning-neural-network-cifar10/
│
├── models/                  # Model architecture
│   ├── prunable_linear.py
│   └── network.py
│
├── training/                # Training pipeline
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
│
├── api/                     # FastAPI backend
│   └── app.py
│
├── ui/                      # Streamlit frontend
│   └── app.py
│
├── utils/                   # Utilities
│   ├── logger.py
│   └── config_loader.py
│
├── config/                  # Configurations
│   └── config.yaml
│
├── data/                    # Dataset (auto-downloaded)
│   └── CIFAR-10 files
│
├── outputs/                 # Models + logs
│   ├── logs/
│   │   └── training.log
│   └── model_lambda_*.pth
│
├── assets/                  # Screenshots (README)
│   └── A1.png
│
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```
## ▶️ How to Run

### 1️⃣ Train the model
```bash
python main.py
```

### 2️⃣ Run API (backend)
```bash
uvicorn api.app:app --reload
```

### 3️⃣ Run UI (frontend)
```bash
streamlit run ui/app.py
```
# 🏗️ Network Architecture

```
Input Image (32×32×3)
        │
        ▼  flatten
  [3072-dim vector]
        │
        ▼
┌───────────────────────────────┐
│ PrunableLinear(3072 → 512)    │  ← 1,572,864 gates
│ BatchNorm1d + ReLU            │
│ Dropout(0.3)                  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ PrunableLinear(512 → 256)     │  ← 131,072 gates
│ BatchNorm1d + ReLU            │
│ Dropout(0.3)                  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ PrunableLinear(256 → 10)      │  ← 2,560 gates
└───────────────────────────────┘
        │
        ▼
  Class Logits (10)
```

### 🔢 Total Learnable Gates

```
Layer 1: 3072 × 512 = 1,572,864
Layer 2: 512 × 256 =   131,072
Layer 3: 256 × 10  =     2,560
--------------------------------
Total Gates ≈ 1,706,496
```

---

### 🧠 Key Idea

Each layer uses:

```
Effective Weight = Weight × Gate
```

- Gates are **learnable parameters**
- L1 regularization pushes gates → 0
- This enables **automatic pruning during training**

---

### ⚙️ Components Used

- PrunableLinear (custom layer)
- Batch Normalization (stability)
- ReLU activation (non-linearity)
- Dropout (regularization)

---

# 📊 Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.001 | ~56% | Low |
| 0.01  | ~57% | Medium |
| 0.1   | ~57% | High |

---
# 🧠 Why L1 Regularization Works

L1 regularization encourages sparsity because:

- It creates sharp corners in optimization space
- Solutions tend to land exactly at zero
- This leads to weight elimination

L2 regularization spreads weights instead of eliminating them.
---
# ❌ What Didn’t Work

- Very high λ (0.5) caused accuracy collapse
- Model over-pruned and lost important features
- Shows importance of balancing sparsity vs performance
---
# 🧠 Key Insights

- Increasing λ increases sparsity but reduces accuracy
- Most pruning happens gradually during training
- High λ leads to over-pruning

---

# 🚀 Technologies Used

- PyTorch
- FastAPI
- Streamlit
- NumPy
- Matplotlib

---

# 👨‍💻 Author

Jeeva M
