# Advanced Autoencoder for Anomaly Detection in Banking Transactions

## 📌 Project Overview
This project implements an **advanced autoencoder** model for **fraud/anomaly detection in banking transactions**.  
The dataset used is [`Pix Banking Transaction`](https://www.kaggle.com/datasets/juniorbueno/pix-banking-transaction) from Kaggle.  

The autoencoder is trained only on **normal transactions** to learn their patterns. During inference, transactions with **high reconstruction error** are flagged as anomalies (potential fraud).

---

## ⚙️ Features
- **GPU Optimization**
  - Mixed precision training (`float16`)
  - Memory growth enabled for efficient GPU utilization
- **Advanced Preprocessing**
  - Log transform and standard scaling for numerical features  
  - Label encoding & one-hot encoding for categorical features
- **Autoencoder Architecture**
  - Deep encoder-decoder with **skip connections**
  - Batch normalization & dropout for regularization
  - Custom loss: **MSE + L2 regularization**
- **Training Enhancements**
  - Early stopping, learning rate scheduling, TensorBoard logging
  - Model checkpointing (`best_autoencoder.h5`)
- **Evaluation**
  - ROC-AUC, precision, recall, F1 score
  - Threshold optimization (90–99.5th percentile)
  - Latent space visualization (PCA)
  - Confusion matrix & error analysis
- **Saved Artifacts**
  - Final autoencoder & encoder models (`.h5`)
  - Preprocessing scalers (`.pkl`)
  - Best threshold for anomaly detection

---

## 📊 Dataset
- **Source:** Kaggle `Pix Banking Transaction`  
- **File used:** `comprovantes_pix_10000_anomalias.csv`  
- **Target column:** `Anomalia` (0 = normal, 1 = anomaly)  
- **Class distribution:**  
  - Normal: ~99%  
  - Anomalous: ~1%  

---

## 🚀 Results
- **AUC-ROC Score:** `0.791`
- **Accuracy:** `0.99`
