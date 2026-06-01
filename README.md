<div align="center">

# 💻 Remote Work Productivity: Employee Performance ML Pipeline

*An End-to-End Machine Learning Project to Predict and Analyze Employee Productivity and Work Quality in Remote Work Environments.*

[![Status](https://img.shields.io/badge/Status-🚀_Active-brightgreen?style=for-the-badge)](https://github.com/)
[![Progress](https://img.shields.io/badge/Progress-40%25_EDA_&_Preprocessing_Done-blue?style=for-the-badge)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📖 Table of Contents
1. [🎯 Project Overview](#-project-overview)
2. [📊 Dataset Information](#-dataset-information)
3. [🧠 Methodology & Pipeline](#-methodology--pipeline)
   - [Exploratory Data Analysis (EDA)](#1️⃣-exploratory-data-analysis-eda)
   - [Feature Engineering & Cleaning](#2️⃣-data-cleaning--feature-engineering)
   - [Next Steps: Modeling & Evaluation](#3️⃣-next-steps-modeling--evaluation)
4. [📁 Project Structure](#-project-structure)
5. [🚀 Future Roadmap](#-future-roadmap)

---

## 🎯 Project Overview

This project aims to build a robust **Machine Learning Pipeline** to predict, analyze, and optimize employee productivity and work quality in remote work environments. By analyzing survey data from 1,500 employees, the system uncovers key drivers of performance—such as work-from-home frequency, home office setup, job satisfaction, and work-life balance—and prepares distinct high-fidelity datasets optimized for predicting **Productivity Score** and **Quality Score**.

---

## 📊 Dataset Information

The data contains survey results capturing the demographics, professional backgrounds, remote work setups, and performance metrics of employees.

| **Metric** | **Value** |
| :--- | :--- |
| 🗃️ **Raw Rows** | 1,500 |
| 🧹 **Processed Rows** | 1,500 (Split into two target-optimized datasets) |
| 📍 **Features** | 30 initial, 27 after feature engineering and cleaning |
| 🎯 **Target Variables** | `Productivity_Score` & `Quality_Score` (0-100 scales) |

### 🔑 Core Features
*   **Demographics:** `Age`, `Gender` (Male, Female, Other), `Education_Level`, `Marital_Status`, `Has_Children`.
*   **Work Environment:** `WFH_Days_Per_Week`, `Location_Type` (Urban, Rural, Suburban), `Home_Office_Quality`, `Internet_Speed_Category`.
*   **Professional Details:** `Department`, `Job_Level`, `Company_Size`, `Industry`, `Years_Experience`, `Work_Hours_Per_Week`.
*   **Well-being & Collaboration:** `Meetings_Per_Week`, `Commute_Time_Minutes`, `Job_Satisfaction`, `Stress_Level`, `Work_Life_Balance`, `Team_Collaboration_Frequency`, `Manager_Support_Level`.
*   **Performance Metrics:** `Productivity_Score`, `Quality_Score`, `Innovation_Score`, `Efficiency_Rating`, `Task_Completion_Rate`.

---

## 🧠 Methodology & Pipeline

The project development is structured into organized phases within the Jupyter notebooks:

### 1️⃣ Exploratory Data Analysis (EDA)
> *Found in `notebooks/data_info.ipynb`*
*   **Statistical Analysis:** Generated summary statistics (`mean`, `std`, `min`, `max`) to understand employee profiles.
*   **Gender Normalization:** Categorized `Non-binary` entries to `Other` to streamline classification and encoding.
*   **Correlation Mapping:** Checked linear correlations between all numerical variables. Productivity features (`Productivity_Score`, `Quality_Score`, `Task_Completion_Rate`, `Innovation_Score`, and `Efficiency_Rating`) show extremely high mutual correlation ($>0.78$), suggesting high alignment between quality and throughput.
*   **Visualizations:** Plotted and exported multiple visualizations, including bar plots comparing `Productivity_Score` across `Education_Level` and `Gender`.

### 2️⃣ Data Cleaning & Feature Engineering
> *Found in `notebooks/data_info.ipynb`*
*   **High-Cardinality Removal:** Dropped irrelevant/metadata columns `Employee_ID`, `Survey_Date`, and `Response_Quality` to avoid model overfitting.
*   **Custom Feature Engineering:** 
    *   🧠 `Smart_Work_Index`: Engineered as $\text{Task\_Completion\_Rate} \times \text{Innovation\_Score}$ to represent high-value output.
*   **Target Leakage Mitigation:** Formulated two specialized sub-datasets to prevent target leakage and collinearity during future regression modeling:
    *   📂 **`df_productivity_score.csv`**: Dropped `Quality_Score` and `Task_Completion_Rate` to ensure clean modeling of overall productivity.
    *   📂 **`df_quality_score.csv`**: Dropped `Productivity_Score` and `Task_Completion_Rate` to isolate factors impacting high-quality output.

### 3️⃣ Next Steps: Modeling & Evaluation
> *Planned for future scripts*
*   **Pipeline Transformations:** Apply `OneHotEncoder` on categorical columns and `StandardScaler` on numerical columns.
*   **Model Training:** Train multiple baseline algorithms (e.g., Linear Regression, Ridge/Lasso, Decision Trees, and Random Forest Regressor) to predict both scores.
*   **Validation:** Use K-Fold Cross Validation evaluating **RMSE** and **R²** metrics.

---

## 📁 Project Structure

```text
Remote-Work-Productivity-ML/
├── 📄 README.md                          # You are here!
├── 📊 data/
│   ├── raw/
│   │   └── remote_work_productivity.csv  # Raw survey dataset (1,500 rows, 30 columns)
│   └── processed/
│       ├── df_productivity_score.csv     # Preprocessed dataset for Productivity prediction
│       └── df_quality_score.csv          # Preprocessed dataset for Quality prediction
└── 📓 notebooks/
    └── data_info.ipynb                   # EDA, Data Cleaning & Feature Engineering Notebook
```

---

## 🚀 Future Roadmap

- [x] Initial Data Exploration (EDA) and analysis of correlations.
- [x] Categorical data cleaning (normalize Gender).
- [x] Metadata column dropping.
- [x] Feature Engineering (`Smart_Work_Index`).
- [x] Target-leakage-free processed dataset splits.
- [ ] 🧪 **Train-Test Stratified Split** (Ensuring departmental and gender representation).
- [ ] 🤖 **Machine Learning Pipelines** (Integration of preprocessing and multiple regressors).
- [ ] 🔄 **Hyperparameter Tuning** (GridSearchCV / RandomizedSearchCV for Tree-based models).
- [ ] 🌐 **Inference API** (Expose productivity prediction via a lightweight FastAPI).

---
<div align="center">
  <i>Built with ❤️ utilizing</i><br><br>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
</div>