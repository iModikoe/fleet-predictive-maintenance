# Fleet Predictive Maintenance and Remaining Useful Life

An end-to-end prognostics project that estimates the remaining useful life (RUL) of turbofan engines and flags abnormal operating behaviour. The project uses NASA C-MAPSS run-to-failure time-series data and enforces engine-level validation to prevent leakage.

## Business question

Can sensor deterioration be detected early enough to estimate how many operating cycles remain before failure and prioritise maintenance?

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation, FD001 subset:

- 100 training engines run until failure
- 100 test engines with withheld future life
- 21 sensor channels and 3 operating settings
- One operating condition and one fault mode

Source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The raw dataset is not committed. Download it with:

```bash
python -m src.download_data
```

## Methodology

- RUL is calculated independently for every training engine.
- RUL is capped at 125 cycles to reduce the influence of long healthy periods.
- Rolling means, rolling standard deviations and short-term sensor trends are engineered.
- Entire engines—not individual rows—are assigned to training and validation sets.
- Linear Regression, Random Forest and HistGradientBoosting are compared using validation RMSE. HistGradientBoosting is preferred when it is within 1% of the best RMSE because it produces a substantially smaller deployment artifact.
- Final performance is measured on the official NASA test engines.
- Isolation Forest provides a separate anomaly score trained on early healthy cycles.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_data
python -m src.train
pytest
streamlit run app/app.py
```

## Repository structure

```text
fleet-predictive-maintenance/
├── app/app.py
├── data/sample_engine.csv
├── models/
├── reports/figures/
├── src/data_loader.py
├── src/download_data.py
├── src/features.py
├── src/train.py
├── tests/
├── .github/workflows/tests.yml
├── MODEL_CARD.md
├── LICENSE
└── requirements.txt
```

## Dashboard output

- Predicted remaining useful life
- Maintenance priority: Normal, Monitor, Warning or Critical
- Anomaly score
- Sensor trajectory visualisation
- Fleet-level engine ranking when multiple engines are uploaded

## Responsible use

This is a portfolio demonstration built on simulated aircraft-engine data. It is not certified for aviation, vehicle maintenance or safety-critical decisions. Production use requires asset-specific failure definitions, representative sensor data, maintenance costs, uncertainty estimates, monitoring and qualified engineering review.
