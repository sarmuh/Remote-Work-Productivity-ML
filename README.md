<div align="center">

# 💻 Remote Work Productivity: Employee Performance ML Pipeline

*An End-to-End Machine Learning Project to Predict, Analyze, and Optimize Employee Productivity and Work Quality in Remote Work Environments.*

[![Status](https://img.shields.io/badge/Status-🚀_Active-brightgreen?style=for-the-badge)](https://github.com/)
[![Progress](https://img.shields.io/badge/Progress-80%25_Modeling_In_Progress-blue?style=for-the-badge)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📖 Table of Contents
1. [🎯 Project Overview](#-project-overview)
2. [📊 Dataset Information](#-dataset-information)
3. [🧠 Methodology & Pipeline](#-methodology--pipeline)
   - [Phase 1: Exploratory Data Analysis (EDA)](#1️⃣-exploratory-data-analysis-eda)
   - [Phase 2: Data Cleaning & Feature Engineering](#2️⃣-data-cleaning--feature-engineering)
   - [Phase 3: Pipeline Design & Model Training](#3️⃣-pipeline-design--model-training)
4. [🏆 Modeling Results (Quality Score)](#-modeling-results-quality-score)
5. [📁 Project Structure](#-project-structure)
6. [🚀 Setup & Installation](#-setup--installation)
7. [🛣️ Roadmap & Future Steps](#-roadmap--future-steps)

---

## 🎯 Project Overview

This repository hosts a robust, end-to-end **Machine Learning Pipeline** designed to predict and analyze employee productivity and work quality in remote working setups. Leveraging survey data from 1,500 employees, the system uncovers key drivers of performance—such as work-from-home (WFH) frequency, home office setup, job satisfaction, stress levels, and team collaboration—and evaluates a wide array of state-of-the-art regression models to predict performance metrics.

Specifically, the pipeline aims to solve two major regression problems:
1. **Predicting Productivity Score** (reflecting output volume and efficiency).
2. **Predicting Quality Score** (reflecting accuracy, reliability, and value of output).

---

## 📊 Dataset Information

The data contains survey results capturing demographical, environmental, professional, and well-being metrics of remote and hybrid workers.

| **Metric** | **Value** |
| :--- | :--- |
| 🗃️ **Raw Rows** | 1,500 |
| 🧹 **Processed Rows** | 1,500 (Split into two target-optimized datasets) |
| 📍 **Features** | 30 raw features $\rightarrow$ 25 engineered/cleaned features |
| 🎯 **Target Variables** | `Productivity_Score` & `Quality_Score` (0-100 scales) |

### 🔑 Core Feature Breakdown
* **Demographics:** `Age`, `Gender`, `Education_Level`, `Marital_Status`, `Has_Children`.
* **Work Environment:** `WFH_Days_Per_Week`, `Location_Type`, `Home_Office_Quality`, `Internet_Speed_Category`.
* **Professional Details:** `Department`, `Job_Level`, `Company_Size`, `Industry`, `Years_Experience`, `Work_Hours_Per_Week`.
* **Well-being & Collaboration:** `Meetings_Per_Week`, `Commute_Time_Minutes`, `Job_Satisfaction`, `Stress_Level`, `Work_Life_Balance`, `Team_Collaboration_Frequency`, `Manager_Support_Level`.
* **Performance Metrics:** `Productivity_Score`, `Quality_Score`, `Innovation_Score`, `Efficiency_Rating`, `Task_Completion_Rate`.

---

## 🧠 Methodology & Pipeline

The project development is structured into clean, modular phases within the Jupyter notebooks:

### 1️⃣ Exploratory Data Analysis (EDA)
> *Found in `notebooks/data_info.ipynb`*
* **Statistical Profiling:** Generated descriptive statistics (`mean`, `std`, `min`, `max`) to check for anomalies or missing data.
* **Feature Distribution:** Explored how performance scores vary across age, gender, education, and WFH frequency.
* **Multicollinearity Checks:** Identified extremely high correlations ($>0.78$) among target/productivity features (`Productivity_Score`, `Quality_Score`, `Task_Completion_Rate`, `Innovation_Score`, and `Efficiency_Rating`), confirming that overall productivity and output quality are highly aligned.

### 2️⃣ Data Cleaning & Feature Engineering
> *Found in `notebooks/data_info.ipynb`*
* **High-Cardinality/Metadata Filtering:** Dropped irrelevant identifiers (`Employee_ID`, `Survey_Date`, `Response_Quality`) to prevent models from learning trivial noise.
* **Gender Normalization:** Consolidated non-binary categories into `Other` to simplify representation during one-hot encoding.
* **High-Value Feature Creation:** 
  * Engineered a custom **`Smart_Work_Index`** defined as:
    $$\text{Smart\_Work\_Index} = \text{Task\_Completion\_Rate} \times \text{Innovation\_Score}$$
    This acts as a synthetic metric reflecting high-quality, innovative output.
* **Target Leakage Mitigation:** Split the processed data into two separate, leakage-free sub-datasets:
  * 📂 **`df_productivity_score.csv`**: Drops `Quality_Score` and `Task_Completion_Rate` to isolate volume metrics.
  * 📂 **`df_quality_score.csv`**: Drops `Productivity_Score` and `Task_Completion_Rate` to isolate quality metrics.

### 3️⃣ Pipeline Design & Model Training
> *Found in `notebooks/Quality_Score_train.ipynb`*
* **ColumnTransformer Integration:** Categorical variables are preprocessed using `OneHotEncoder(handle_unknown='ignore')`, and numerical features are standardized using `StandardScaler()`.
* **Stratified Splitting:** Applied `train_test_split` with `stratify=X['Gender']` to maintain gender ratios across train and test sets.
* **Multi-Model Benchmark:** Evaluated **14 different regression algorithms** simultaneously, including linear formulations, tree ensembles, and advanced gradient boosters.

---

## 🏆 Modeling Results (Quality Score)

Below are the performance metrics of the 14 models benchmarked on the test set for predicting **`Quality_Score`**, sorted by **$R^2$ Score**:

| **Rank** | **Model** | **MAE** | **RMSE** | **$R^2$ Score** | **Training Time (sec)** |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest Regressor** | **5.177** | **6.578** | **0.751** | 0.955 |
| 🥈 | **Gradient Boosting Regressor** | **5.275** | **6.620** | **0.748** | 0.281 |
| 🥉 | **CatBoost Regressor** | **5.193** | **6.637** | **0.746** | 1.473 |
| 4 | LightGBM Regressor | 5.335 | 6.799 | 0.734 | 0.110 |
| 5 | Ridge Regression | 5.472 | 6.954 | 0.722 | 0.004 |
| 6 | Hist Gradient Boosting Regressor | 5.435 | 6.953 | 0.722 | 0.256 |
| 7 | Linear Regression | 5.474 | 6.966 | 0.721 | 0.015 |
| 8 | Lasso Regression | 5.608 | 6.958 | 0.721 | 0.001 |
| 9 | MLP Regressor | 5.454 | 6.993 | 0.718 | 1.556 |
| 10 | XGBoost Regressor | 5.731 | 7.207 | 0.701 | 0.086 |
| 11 | ElasticNet Regression | 6.046 | 7.437 | 0.682 | 0.003 |
| 12 | Support Vector Regressor | 6.079 | 7.529 | 0.674 | 0.111 |
| 13 | K-Nearest Neighbors Regressor | 6.175 | 7.857 | 0.645 | 0.012 |
| 14 | Decision Tree Regressor | 7.869 | 9.892 | 0.437 | 0.026 |

> [!NOTE]
> Tree ensemble models (Random Forest, Gradient Boosting, CatBoost, and LightGBM) perform exceptionally well, achieving $R^2$ scores above **0.73-0.75**. This indicates that tree-based models successfully capture non-linear relationships between environment variables (like home office quality, internet speed) and output quality.

---

## 📁 Project Structure

```text
Remote-Work-Productivity-ML/
├── 📄 .gitignore                         # Comprehensive git ignore configuration (Added)
├── 📄 README.md                          # Detailed documentation guide (Updated)
├── 📊 data/
│   ├── raw/
│   │   └── remote_work_productivity.csv  # Raw survey dataset (1,500 rows, 30 columns)
│   └── processed/
│       ├── df_productivity_score.csv     # Preprocessed dataset for Productivity prediction
│       └── df_quality_score.csv          # Preprocessed dataset for Quality prediction
└── 📓 notebooks/
    ├── data_info.ipynb                   # EDA, Data Cleaning & Feature Engineering Notebook
    └── Quality_Score_train.ipynb         # Model Training, Pipelines & Evaluation (14 Models)
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Remote-Work-Productivity-ML.git
cd Remote-Work-Productivity-ML
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install required libraries
```bash
pip install -r requirements.txt
```
*(If a `requirements.txt` is not provided, manual packages to install include: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`, `jupyter`)*

### 4. Run the notebooks
Start Jupyter notebook server:
```bash
jupyter notebook
```
Navigate to `notebooks/` and open either `data_info.ipynb` or `Quality_Score_train.ipynb`.

---

## 🛣️ Roadmap & Future Steps

- [x] **Initial Exploration (EDA):** Check distributions, correlation patterns.
- [x] **Data Processing:** Clean categorical fields, drop noise features.
- [x] **Feature Engineering:** Build custom indices (`Smart_Work_Index`).
- [x] **Leakage Separation:** Split target-specific sub-datasets.
- [x] **Stratified Benchmarking:** Implemented a standardized pipeline across 14 regression models for `Quality_Score`.
- [ ] 🤖 **Productivity Score Training:** Extend the model pipeline to benchmark `Productivity_Score`.
- [ ] 🔄 **Hyperparameter Tuning:** Fine-tune top-performing tree-based models (Random Forest, CatBoost) to squeeze out more performance.
- [ ] 📦 **Model Serialization:** Save the best performing models as `.joblib` files.
- [ ] 🌐 **FastAPI Inference Service:** Develop a lightweight API endpoint to serve model predictions in real-time.

---

<div align="center">
  <i>Built with ❤️ utilizing</i><br><br>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/XGBoost-1572B6?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
</div>