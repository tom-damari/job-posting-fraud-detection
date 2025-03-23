# Job Posting Fraud Detection – Machine Learning Project

**Course:** Machine Learning  
**Department:** Industrial Engineering and Management, Ben-Gurion University  
**Submission Date:** June 2023  
**Team Members:** Tom Damari, Doron Ortner  
**Instructor:** Dr. Nir Nissim

---

## Project Overview
This project aims to detect fraudulent job postings using machine learning techniques. The dataset includes real and fake job ads, and the goal is to build and evaluate classification models capable of distinguishing between them.
The work was divided into two parts:
- **Part A:** Data understanding, exploratory analysis, and data preparation  
- **Part B:** Model training, evaluation, comparison, and optimization

---

## Dataset Description
The dataset includes metadata and content from job postings such as:
- `title`, `location`, `company_profile`, `description`, `requirements`, `salary_range`
- Categorical fields like `employment_type`, `required_experience`, `industry`, etc.
- Boolean flags: `has_company_logo`, `telecommuting`, `has_questions`
- **Target variable:** `fraudulent` (0 = real, 1 = fake)

---

## Part A – Data Exploration & Preprocessing
- **EDA:** Analyzed feature distributions, relationships, class imbalance, and feature interactions
- **Text fields** were processed using tokenization, cleaning, and length features
- **Encoding** of categorical features (One-Hot, Label Encoding)
- **Missing value handling** and **outlier detection**
- **Feature engineering**: created features like description length, company reputation proxy, etc.
- **Feature selection** and **dimensionality reduction** to improve model performance

---

## Part B – Model Training & Evaluation

### Models Trained
| Model                  | Tuning | Final Validation AUC |
|------------------------|--------|-----------------------|
| Decision Tree          | ✔️     | ...                  |
| SVM (Linear Kernel)    | ✔️     | ...                  |
| Neural Network (MLP)   | ✔️     | ...                  |

### Hyperparameter Tuning
- Grid search and cross-validation were used (K-Fold / Holdout)
- For neural networks: number of layers, hidden units, batch size, dropout, activation
- For trees: depth, min samples per leaf, and splitting criteria

### Evaluation Metric
- **AUC-ROC** was chosen due to class imbalance in the dataset
- Additional metrics: confusion matrix, precision-recall, F1-score

---

## Interpretability & Insights
- Decision tree structure and feature importance visualized
- Neural network learned key non-linear patterns
- Identified critical features such as:
  - Presence of company logo
  - Text-based features from description/requirements
  - Salary range anomalies

---

## Unsupervised Learning (Clustering)
- Applied **K-Medoids** clustering on selected features
- Tested multiple `k` values and evaluated clusters using silhouette scores and interpretability
- Provided insights into subgroup patterns among job postings

---

## Final Prediction Output
- Final model applied to `X_test.csv`
- Predictions submitted in `Gx_ytest.csv` format as required
- Preprocessing pipeline was consistently applied to avoid data leakage

---

## Tools & Libraries
- **Python** (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
- **NLTK / spaCy / scikit-text** for text preprocessing
- **Jupyter Notebook / PyCharm**
- **Excel** (for manual checks)

---

## Project Files

| File | Description |
|------|-------------|
| `part_a_report.pdf` | Pdf report for Part A |
| `part_b_report.pdf` | Pdf report for Part B |
| `main.py / preprocessing.py / models.py` | Python scripts for training and evaluation |
| `X_test.csv` | Final test set |
| `Gx_ytest.csv` | Submitted predictions |
| `eda_plots/` | Visuals from exploratory analysis |
| `results/` | Confusion matrices, ROC curves, etc.

---

## Summary
Through rigorous data preparation, feature engineering, and model evaluation, this project demonstrates how machine learning can assist in identifying fraudulent job advertisements and contribute to safer recruitment platforms.
