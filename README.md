# Triathlete Injury Prediction Using Machine Learning

## Overview

**Triathlete Injury Prediction Using Machine Learning** is a sports analytics and predictive modeling project that uses daily physiological data, wearable biometrics, and training session records to forecast injury risk in endurance athletes. Built on a large-scale synthetic triathlete dataset spanning the full year of 2024, the system models injury probability using athlete workload patterns, recovery signals, and biometric trends.

The project covers the full data science lifecycle: multi-table data integration, feature engineering from time-series training logs, exploratory data analysis, class imbalance handling, model training, evaluation, and interpretability.

---

## Key Highlights

- Injury prediction across 1,000 synthetic triathletes over 366 days
- Multi-table relational merge across athlete profiles, daily biometrics, and activity sessions
- Temporal alignment of activity sessions to daily grain before modeling
- Feature engineering from wearable signals: HRV, heart rate, sleep debt, and workload metrics
- ACWR-based workload modeling grounded in sports science injury risk literature
- SHAP-based interpretability for per-athlete explainability
- Handles severe class imbalance inherent to injury prediction tasks
- Comprehensive evaluation: ROC-AUC, PR-AUC, precision, recall, F1, calibration curve
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

## Machine Learning Methodology

### Data Integration

Three CSV files are merged on `athlete_id` to form a unified daily record per athlete. Activity sessions are aggregated per day — computing total training load, session count, average RPE, and discipline-specific volumes — before joining with the daily biometric table to produce a single flat feature matrix at daily grain.

### Feature Engineering

Key predictive features derived from raw data include:

- **Acute:Chronic Workload Ratio (ACWR)** — rolling 7-day vs. 28-day training load ratio, a widely used injury risk proxy in sports science
- **Training Monotony & Strain Index** — standard deviation and product-based load quality metrics
- **HRV rolling mean, std, and trend slope** — multi-day heart rate variability trajectory
- **Sleep debt accumulation** — cumulative deviation from each athlete's baseline sleep duration
- **Resting heart rate trend** — multi-day rolling average indicating recovery state
- **Discipline-specific load** — separate swim, bike, and run session volume per rolling window
- **RPE-weighted session load** — perceived exertion scaled by duration
- **Lagged injury flag** — temporal context from prior injury events

### Class Imbalance Handling

Injury events are rare relative to non-injury days, creating significant class imbalance. The project addresses this through:

- **SMOTE** applied strictly to the training set only to avoid data leakage
- **Class-weighted loss functions** in applicable models
- **Probability threshold calibration** to optimize recall for injury-positive predictions

### Training Configuration

| Parameter | Value |
|---|---|
| Task | Binary Classification (injury / no injury) |
| Primary Metrics | ROC-AUC, PR-AUC, Recall |
| Split Strategy | Athlete-wise stratified train / validation / test split |
| Imbalance Handling | SMOTE + class weights + threshold calibration |
| Feature Selection | Correlation analysis + SHAP importance ranking |

### Models Evaluated

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine
- Best model selected by ROC-AUC on the validation set

---

## Evaluation

The model is evaluated using metrics appropriate for imbalanced binary classification in a health-critical setting.

### Metrics

- **ROC-AUC** — primary ranking metric for injury risk scoring
- **PR-AUC** — precision-recall area under curve, robust to class imbalance
- **Recall (Sensitivity)** — prioritized to minimize missed injury predictions
- **Precision** — to assess false alarm rate
- **F1-Score** — harmonic mean for overall classification quality
- **Confusion Matrix** — breakdown of true/false positives and negatives
- **Calibration Curve** — reliability of predicted injury probabilities
- **SHAP Values** — feature-level contribution analysis per prediction

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/triathlete-injury-prediction.git
cd triathlete-injury-prediction
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

### Download the Dataset

Download the three CSV files from Zenodo and place them in the project directory:

> [https://zenodo.org/records/15401061](https://zenodo.org/records/15401061)

Expected file layout:

```
├── athletes.csv
├── daily_data.csv
└── activity_data.csv
```

---

## Usage

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Ensure `athletes.csv`, `daily_data.csv`, and `activity_data.csv` are in the working directory or mounted via Google Drive.
3. Run notebook cells in order:
   - Data loading and multi-table relational merge
   - Temporal alignment and activity aggregation to daily grain
   - Exploratory data analysis and visualizations
   - Feature engineering (ACWR, HRV trends, sleep debt, monotony, strain)
   - Class imbalance handling (SMOTE, class weights, threshold tuning)
   - Model training and hyperparameter tuning
   - Evaluation: ROC-AUC, PR-AUC, F1, confusion matrix, calibration curve
   - SHAP-based feature importance and interpretability analysis
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
- SHAP
- Matplotlib
- Seaborn
- Google Colab

---

## Engineering Principles

- Athlete-wise stratified splitting to prevent identity leakage across train/test sets
- Activity aggregation to daily grain before any feature computation or merging
- SMOTE applied only to training set to prevent target leakage
- ACWR-based workload modeling aligned with established sports science literature
- Recall-prioritized threshold calibration for safety-critical injury prediction
- SHAP values for transparent, per-prediction explainability
- Modular notebook structure for reproducibility and experimentation
- All dataset citation and licensing requirements respected

---

## Potential Extensions

- LSTM or Transformer models operating directly on raw daily time series per athlete
- Multi-class prediction for injury type or body region classification
- Personalized risk thresholds calibrated to each athlete's historical baseline
- Real-time wearable integration via a streaming data pipeline
- Flask or FastAPI deployment for a coach-facing injury risk dashboard
- Integration with GPS and power meter data for richer load modeling
- Federated learning to train across athlete cohorts while preserving privacy

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
