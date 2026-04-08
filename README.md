The badges use external URLs from `shields.io` — they're not local image files, so they'll render perfectly on GitHub without any screenshots in your repo. But if you'd prefer to exclude them entirely, here's the full README without any `<img>` tags whatsoever:

````markdown
# Triathlete Injury Prediction Using Machine Learning

## Overview

**Triathlete Injury Prediction Using Machine Learning** is a sports analytics and predictive modeling project that uses daily physiological data, wearable biometrics, and training session records to forecast injury risk in endurance athletes. Built on a large-scale synthetic triathlete dataset spanning the full year of 2024, the system models injury probability using athlete workload patterns, recovery signals, and biometric trends.

The project covers the full data science lifecycle: multi-table data integration, feature engineering from time-series training logs, exploratory data analysis, class imbalance handling, model training, evaluation, and interpretability.

---

## Key Highlights

- Injury prediction across 1,000 synthetic triathletes over 366 days
- Multi-table data integration across athlete profiles, daily biometrics, and activity sessions
- Feature engineering from wearable signals: HRV, heart rate, sleep, and workload metrics
- Handles severe class imbalance inherent to injury prediction tasks
- Comprehensive evaluation: accuracy, precision, recall, F1, ROC-AUC
- Athlete-level and day-level prediction granularity
- End-to-end reproducible pipeline in a single Jupyter Notebook

---

## Dataset

This project uses the **Synthetic Triathlete Dataset for Injury Prediction Research (2024)**, publicly available on Zenodo.

### Dataset Link

> [https://zenodo.org/records/15401061](https://zenodo.org/records/15401061)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Dataset Description

The dataset contains synthetic data for **1,000 triathletes** over the full year **2024 (January 1 – December 31)**. It was generated to reflect realistic physiological and training patterns observed in endurance athletes while maintaining complete privacy through synthetic generation techniques.

| File | Size | Description |
|---|---|---|
| `athletes.csv` | 465.2 kB | Static demographic and profile information per athlete (age, gender, training background) |
| `daily_data.csv` | 71.2 MB | Daily physiological and biometric readings from simulated wearables (heart rate, HRV, sleep) |
| `activity_data.csv` | 115.4 MB | Timestamped individual training sessions (type, intensity, duration) linked to athlete ID |

- **Total daily records:** 366,000
- **Total activity sessions:** 384,153

### Citation

> Rossi, Leonardo. (2025). *Synthetic Triathlete Dataset for Injury Prediction Research (2024)*. Zenodo. https://doi.org/10.5281/zenodo.15401061

---

## System Architecture

### End-to-End Pipeline

```
athletes.csv + daily_data.csv + activity_data.csv
                    ↓
        Multi-Table Merge on Athlete ID
                    ↓
     Exploratory Data Analysis & Distributions
                    ↓
Feature Engineering (rolling workload, HRV trends,
    ACWR, sleep deficit, training load spikes)
                    ↓
        Class Imbalance Handling (SMOTE / class weights)
                    ↓
         Train / Validation / Test Split
                    ↓
     Model Training (multiple classifiers compared)
                    ↓
  Evaluation: Accuracy, F1, Recall, ROC-AUC, Confusion Matrix
                    ↓
       Feature Importance & Interpretability Analysis
```

---

## Machine Learning Methodology

### Data Integration

Three CSV files are merged on `athlete_id` to form a unified daily record per athlete. Activity sessions are aggregated per day (e.g., total training load, session count, average intensity) before joining with the daily biometric table.

### Feature Engineering

Key predictive features derived from raw data include:

- **Acute:Chronic Workload Ratio (ACWR)** — rolling 7-day vs. 28-day training load ratio, a widely used injury risk proxy
- **HRV trend** — rolling mean and standard deviation of heart rate variability
- **Sleep deficit** — deviation from baseline sleep duration
- **Resting heart rate trend** — multi-day rolling average
- **Training monotony** — standard deviation of daily load (low variety = higher risk)
- **Session frequency and type** — swim, bike, run session counts per rolling window
- **Cumulative fatigue indicators** — multi-day lagged load accumulation

### Class Imbalance Handling

Injury events are rare relative to non-injury days, creating significant class imbalance. The project addresses this through:

- **SMOTE** (Synthetic Minority Oversampling Technique) for training set balancing
- **Class-weighted loss** in applicable models
- **Threshold tuning** to optimize recall for injury-positive predictions

### Training Configuration

| Parameter | Value |
|---|---|
| Task | Binary Classification (injury / no injury) |
| Primary Metric | ROC-AUC, Recall |
| Split Strategy | Athlete-wise stratified split |
| Imbalance Handling | SMOTE + class weights |
| Feature Selection | Correlation analysis + importance ranking |

### Models Evaluated

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)
- Support Vector Machine
- Best model selected by ROC-AUC on validation set

---

## Evaluation

The model is evaluated using metrics appropriate for imbalanced binary classification in a health-critical setting.

### Metrics

- **ROC-AUC** — primary ranking metric for injury risk scoring
- **Recall (Sensitivity)** — prioritized to minimize missed injury predictions
- **Precision** — to assess false alarm rate
- **F1-Score** — harmonic mean for overall classification quality
- **Confusion Matrix** — breakdown of true/false positives and negatives
- **Feature Importance** — ranking of most predictive biometric and workload signals

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/triathlete-injury-prediction.git
cd triathlete-injury-prediction
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn
```

### Download the Dataset

Download the three CSV files from Zenodo and place them in the project directory:

> [https://zenodo.org/records/15401061](https://zenodo.org/records/15401061)

```
├── athletes.csv
├── daily_data.csv
└── activity_data.csv
```

---

## Usage

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Ensure `athletes.csv`, `daily_data.csv`, and `activity_data.csv` are in the working directory (or mounted via Google Drive).
3. Run notebook cells in order:
   - Data loading and multi-table merging
   - Exploratory data analysis and visualizations
   - Feature engineering (rolling windows, ACWR, HRV trends)
   - Class imbalance handling
   - Model training and hyperparameter tuning
   - Evaluation and metric reporting
   - Feature importance and interpretability analysis
4. Review evaluation outputs and per-athlete injury risk scores.

---

## Repository Structure

```
├── triathlete_injury_prediction.ipynb     # Main notebook
├── README.md
└── LICENSE
```

> Dataset files (`athletes.csv`, `daily_data.csv`, `activity_data.csv`) are not included in the repository. Download them directly from [Zenodo](https://zenodo.org/records/15401061).

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Google Colab

---

## Engineering Principles

- Athlete-wise data splitting to prevent identity leakage across train/test sets
- Rolling window feature engineering grounded in sports science literature
- ACWR-based workload modeling aligned with established injury risk research
- SMOTE and class-weighted training to handle injury event rarity
- Recall-prioritized threshold selection for safety-critical prediction
- Modular notebook structure for reproducibility and experimentation
- All dataset citation and licensing requirements respected

---

## Potential Extensions

- Deep learning approaches: LSTM or Transformer on daily time series per athlete
- Multi-class prediction: injury type or severity classification
- Personalized risk thresholds calibrated per athlete baseline
- Real-time wearable integration via streaming pipeline
- Flask or FastAPI deployment for coach-facing injury risk dashboard
- SHAP-based per-athlete explainability reports
- Integration with GPS and power meter data for richer load modeling

---

## Dataset License

The dataset is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Please cite the dataset appropriately and review the full license terms on the [Zenodo record](https://zenodo.org/records/15401061) before redistribution or commercial use.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**  
[GitHub](https://github.com/sushantkothari)
````
