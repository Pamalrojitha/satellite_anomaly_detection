# Residual-Based Anomaly Detection of Satellite Orbital Manoeuvres via ARIMA and XGBoost Forecasting

## Overview
This repository contains the complete implementation of a residual-based forecasting framework for detecting satellite orbital manoeuvres from Two-Line Element (TLE)-derived orbital elements. The project applies both **ARIMA** and **XGBoost** models to forecast normal orbital element dynamics and identifies large deviations in residuals as potential manoeuvres.

The codebase is fully modular and structured for reproducibility. All essential input data (orbital element CSVs and manoeuvre files) are included to allow inspection and rerunning of experiments. Generated outputs such as figures and intermediate logs are excluded to keep the repository size manageable.

The study and methodology are described in detail in the accompanying report.

---

## Repository Structure

```
satellite_anomaly_detection/
│
├── utils/                   # Modular utility scripts for preprocessing, forecasting, evaluation
│   ├── eda_tools.py
│   ├── forecasting_arima_tools.py
│   ├── forecasting_xgb_tools.py
│   ├── precision_recall_tools.py
│
├── runners/                 # Runner scripts for executing experiments
│   ├── eda_runner.py
│   ├── arima_runner.py
│   ├── xgboost_runner.py
│
├── satellite_data/          # Compact dataset for reproducibility
│   ├── orbital_elements/    # CSVs of TLE-derived orbital elements
│   ├── manoeuvres/          # Original manoeuvre text files
│
├── arima_logs/               # Example logs (PR summaries, outlier overlaps)
│   ├── pr_summary.csv
│   ├── outlier_overlap_log.csv
│
├── xgb_logs/                 # Example logs (PR summaries, outlier overlaps)
│   ├── pr_summary.csv
│   ├── outlier_overlap_log.csv
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
---

## Key Features

**EDA**
- Outlier detection via Interquartile Range (IQR)
- Feature selection based on stability and anomaly separability

**Forecasting Models**
- ARIMA (raw, smoothed, outlier-cleaned variants)
- XGBoost (univariate & multivariate, raw & cleaned variants)

**Evaluation**
- Residual-based anomaly detection
- Event-level precision–recall analysis with varying temporal tolerances (±3 and ±5 days)

---

## Requirements
See `requirements.txt` for the full list of Python packages.  
Main dependencies:

- Python 3.10+
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- xgboost
- plotly

---

## Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/Pamalrojitha/satellite_anomaly_detection.git
   cd satellite_anomaly_detection

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run EDA**:
Open ```runners/eda_runner.ipynb``` in Jupyter Notebook and execute all cells.

4. **Run ARIMA and XGBoost experiments**:
- ```runners/forecast_runner_arima.ipynb```
- ```runners/XGBoost_Runner.ipynb```

5. **Generate combined PR curve comparisons**:
- ```runners/combined_pr_plots_runner.ipynb```

Generated plots and logs will be saved in structured directories (```images/```, ```arima_logs/```, ```xgb_logs/```).

## Notes
- The repository is self-contained with the necessary orbital element and manoeuvre files for inspection.
- Output images and large generated files are not version-controlled to keep the repository size small. They can be reproduced by running the provided scripts.
- The methodology is documented in the research report, which explains the preprocessing, modelling, and evaluation procedures.

## Citation
If you use this code or methodology in your research, please cite:
Nanayakkara, P. (2025). Residual-Based Anomaly Detection for Satellite Orbital Manoeuvres using ARIMA and XGBoost. University of Adelaide.


