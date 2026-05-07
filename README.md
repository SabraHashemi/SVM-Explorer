---
title: ML Explorer
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🤖 ML Explorer

**An all-in-one interactive playground for Machine Learning algorithms**

[![Python](https://img.shields.io/badge/Python-3.9%2B-4fc3f7?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00e5ff?style=for-the-badge)](LICENSE)

[🚀 Live Demo on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/ml-explorer) &nbsp;|&nbsp;
[📦 GitHub Repo](https://github.com/YOUR_USERNAME/ml-explorer)

</div>

---

## 🧭 What's Inside

| Algorithm | Type | Key Features |
|-----------|------|-------------|
| ⚡ **SVM** | Classification | Kernel trick, margin visualization, support vectors |
| 📈 **Regression** | Regression | Linear, Polynomial, Ridge, Lasso, residual plots |
| 🌳 **Decision Tree / RF** | Classification | Depth control, feature importance, ensemble power |
| 🔵 **K-Means** | Clustering (Unsupervised) | Elbow curve, centroid animation, cluster explorer |
| 👥 **KNN** | Classification | K sweep, distance metrics, boundary visualization |
| 🧠 **Neural Network (MLP)** | Classification | Custom architecture, loss curve, network diagram |

---

## ✨ Features

- **📂 Upload your own CSV** — pick feature columns and label column interactively
- **📦 Built-in datasets** — Moons, Circles, Blobs, Iris, and more
- **🎛️ Live hyperparameter tuning** — every parameter updates the plot in real time
- **📊 Full evaluation** — confusion matrix, classification report, R² / RMSE for regression
- **🎨 Navy dark theme** — clean, professional UI throughout

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/ml-explorer.git
cd ml-explorer
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` 🎉

---

## ☁️ Deploy to Hugging Face Spaces

**One-click via GitHub import:**

1. [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Streamlit**
2. Click **Import from GitHub** → paste your repo URL
3. **Create Space** ✅

---

## 📂 CSV Upload Format

Any CSV with numeric columns works. Example:

```
sepal_length, sepal_width, species
5.1, 3.5, 0
4.9, 3.0, 1
...
```

- Pick **Feature 1**, **Feature 2**, and **Label** column in the UI
- Label must be binary or multi-class integers
- For regression: pick one feature (X) and one target (Y)

---

## 📁 Project Structure

```
ml-explorer/
├── app.py            # Single-file Streamlit app (~600 lines)
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## 📦 Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
seaborn>=0.12.0
```

---

<div align="center">

Made with ❤️ using [Streamlit](https://streamlit.io) & [scikit-learn](https://scikit-learn.org)

</div>
