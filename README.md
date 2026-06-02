<div align="center">

# 💻 Remote Work Productivity: ML Prediction Pipeline

*An End-to-End Machine Learning Project to Predict, Analyze, and Optimize Employee Productivity and Work Quality in Remote Work Environments.*

[![Status](https://img.shields.io/badge/Status-🚀_Active-brightgreen?style=for-the-badge)](#)
[![Progress](https://img.shields.io/badge/Progress-90%25_Modeling_Completed-blue?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📖 Table of Contents
1. [🎯 Project Overview](#-project-overview)
2. [📊 Dataset Information](#-dataset-information)
3. [🧠 Methodology & Pipeline](#-methodology--pipeline)
4. [🏆 Modeling Results](#-modeling-results)
   - [Productivity Score Models](#productivity-score-models)
   - [Quality Score Models](#quality-score-models)
5. [📁 Project Structure](#-project-structure)
6. [🚀 Setup & Installation](#-setup--installation)
7. [🛣️ Roadmap & Future Steps](#-roadmap--future-steps)

---

## 🎯 Project Overview

This repository hosts a robust, end-to-end **Machine Learning Pipeline** designed to predict and analyze employee performance metrics in remote and hybrid working setups. Leveraging survey data from 1,500 employees, the system uncovers key drivers of performance—such as WFH frequency, home office setup, job satisfaction, stress levels, and team collaboration.

The project solves two major regression problems:
1. **Predicting Productivity Score**: Evaluating output volume and overall efficiency.
2. **Predicting Quality Score**: Evaluating the accuracy, reliability, and value of the work delivered.

---

## 📊 Dataset Information

The dataset comprises survey results capturing the demographical, environmental, professional, and well-being metrics of workers.

| **Metric** | **Value** |
| :--- | :--- |
| 🗃️ **Raw Rows** | 1,500 (`data/raw/remote_work_productivity.csv`) |
| 🧹 **Processed Rows** | 1,500 (Split into two target-optimized datasets to avoid target leakage) |
| 📍 **Features** | 30 raw features $\rightarrow$ 25 engineered/cleaned features |
| 🎯 **Target Variables** | `Productivity_Score` & `Quality_Score` (0-100 scales) |

### 🔑 Core Features Breakdown
* **Demographics:** `Age`, `Gender`, `Education_Level`, `Marital_Status`, `Has_Children`.
* **Work Environment:** `WFH_Days_Per_Week`, `Location_Type`, `Home_Office_Quality`, `Internet_Speed_Category`.
* **Professional Details:** `Department`, `Job_Level`, `Company_Size`, `Industry`, `Years_Experience`, `Work_Hours_Per_Week`.
* **Well-being & Collaboration:** `Meetings_Per_Week`, `Commute_Time_Minutes`, `Job_Satisfaction`, `Stress_Level`, `Work_Life_Balance`, `Team_Collaboration_Frequency`.

---

## 🧠 Methodology & Pipeline

The ML pipeline is structured into clean, modular phases documented in our Jupyter notebooks:

### 1️⃣ Exploratory Data Analysis & Cleaning (`notebooks/data_info.ipynb`)
- **Statistical Profiling:** Analyzed feature distributions and identified extreme multicollinearity among performance metrics (`Productivity_Score`, `Quality_Score`, `Innovation_Score`, etc.).
- **Data Pruning:** Dropped irrelevant metadata (`Employee_ID`, `Survey_Date`) and consolidated high-cardinality classes.
- **Leakage Prevention:** Created completely isolated datasets (`df_productivity_score.csv` and `df_quality_score.csv`) so models wouldn't use one target to predict the other.

### 2️⃣ Pipeline Design & Feature Engineering
- **Encoding & Scaling:** Utilized `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` for categoricals and `StandardScaler()` for numericals.
- **Stratification:** Split the data maintaining gender ratios (`stratify=X['Gender']`).
- **Iterative Experimentation:** We utilized a primary pipeline (`notebooks/`) and a retry pipeline (`Quality-retry-ML/`) to iteratively drop noisy engineered features (like `Smart_Work_Index`) and perform deep hyperparameter tuning.

### 3️⃣ Multi-Model Benchmarking
For both targets, we benchmarked **14 regression algorithms**, ranging from Linear/Ridge/Lasso regressions to powerful tree ensembles (Random Forest, XGBoost, CatBoost, LightGBM, HistGradientBoosting).

---

## 🏆 Modeling Results

### Productivity Score Models
*Evaluated in `notebooks/Productivitiy_Score_train.ipynb`*

| **Rank** | **Model** | **MAE** | **RMSE** | **$R^2$ Score** | **Training Time (s)** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting Regressor** | **3.643** | **4.780** | **0.893** | 0.270 |
| 🥈 | Random Forest Regressor | 3.808 | 4.900 | 0.887 | 0.856 |
| 🥉 | CatBoost Regressor | 3.737 | 4.903 | 0.887 | 1.644 |

> **Conclusion:** Tree-based ensemble methods captured the non-linear dynamics of productivity exceptionally well. The **Gradient Boosting Regressor** was selected as the final production model for Productivity and serialized to `models/Productivity/`.

### Quality Score Models
*Evaluated in `Quality-retry-ML/notebooks/traininng.ipynb`*

Initially, Random Forest achieved an $R^2$ of 0.751. However, by optimizing our feature set (dropping highly-correlated index features that added noise) and running a deep hyperparameter tuning phase, we achieved significant improvements.

| **Rank** | **Model** | **MAE** | **RMSE** | **$R^2$ Score** | **Training Time (s)** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Lasso Regression (alpha=0.1)** | **-** | **-** | **0.824** | 0.001 |
| 🥈 | Random Forest Regressor | 4.542 | 5.639 | 0.817 | 0.979 |
| 🥉 | Ridge Regression | 4.462 | 5.679 | 0.814 | 0.002 |

> **Conclusion:** The refined pipeline revealed that a regularized linear model, **Lasso Regression (alpha=0.1)**, outperformed complex ensembles for Quality prediction. It was selected as the final production model and serialized to `models/Quality/`.

---

## 📁 Project Structure

```text
Remote-Work-Productivity-ML/
├── 📄 .gitignore                         # Git ignore configurations
├── 📄 README.md                          # Main project documentation
├── 📊 data/
│   ├── raw/
│   │   └── remote_work_productivity.csv  # Raw dataset
│   └── processed/
│       ├── df_productivity_score.csv     # Cleaned dataset for Productivity
│       └── df_quality_score.csv          # Cleaned dataset for Quality
├── 📂 models/                            # Serialized Production Models
│   ├── 📂 Productivity/
│   │   ├── 📄 GradientBoostingRegressor.joblib
│   │   └── 📄 preprocessor.joblib
│   └── 📂 Quality/
│       ├── 📄 Lasso_model.joblib
│       └── 📄 preprocessor.joblib
├── 📓 notebooks/
│   ├── data_info.ipynb                   # Initial EDA and Cleaning
│   ├── Productivitiy_Score_train.ipynb   # 14-Model Benchmark for Productivity
│   └── Quality_Score_train.ipynb         # Initial Benchmark for Quality
└── 📂 Quality-retry-ML/                   # Advanced Tuning & Retry Experiments
    ├── 📊 data/                          
    └── 📓 notebooks/
        ├── preprocessing.ipynb           # Feature optimization
        └── traininng.ipynb               # Hyperparameter tuning for Lasso
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Remote-Work-Productivity-ML.git
cd Remote-Work-Productivity-ML
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn jupyter joblib
```

### 4. Explore the Notebooks
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory to review the training pipelines.

---

## 🛣️ Roadmap & Future Steps

- [x] **Initial Exploration (EDA):** Understand distributions and target correlations.
- [x] **Data Processing & Cleaning:** Handle missing values, encode categoricals, remove target leakage.
- [x] **Stratified Benchmarking:** Build and benchmark 14 regression pipelines.
- [x] 🤖 **Productivity Score Training:** Achieved an excellent $R^2 \approx 0.893$ using Gradient Boosting.
- [x] 🔄 **Quality Score Optimization:** Refined features and tuned Lasso Regression (alpha=0.1) to achieve $R^2 \approx 0.824$.
- [x] 📦 **Model Serialization:** Saved production models as `.joblib` files.
- [ ] 🌐 **FastAPI Inference Service:** Develop a scalable API endpoint to serve predictions via REST.
- [ ] 📊 **Streamlit Dashboard:** Build an interactive frontend to visualize employee productivity profiles.

---

<div align="center">
  <i>Built with ❤️ utilizing</i><br><br>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/XGBoost-1572B6?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
</div>