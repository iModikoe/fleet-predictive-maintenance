# Model Card

## Purpose

Estimate remaining useful life for simulated turbofan engines and support maintenance prioritisation.

## Validation design

Engines are separated with GroupShuffleSplit. No measurements from a validation engine appear in training. The official NASA test trajectories and RUL labels are reserved for final evaluation.

The lowest validation RMSE wins unless HistGradientBoosting is within 1% of the best score. In that case, the compact HistGradientBoosting model is selected to reduce deployment size with negligible validation-performance loss.

## Limitations

- C-MAPSS is simulated aircraft-engine data, not vehicle-fleet data.
- Point estimates do not express predictive uncertainty.
- Anomaly scores indicate deviation, not a diagnosed fault.
- Maintenance thresholds are illustrative and not cost-optimised.
- Production deployment requires engineering validation and asset-specific data.
