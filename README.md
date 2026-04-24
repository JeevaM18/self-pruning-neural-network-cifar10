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

# 📊 Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.001 | ~56% | Low |
| 0.01  | ~57% | Medium |
| 0.1   | ~57% | High |

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
