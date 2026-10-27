# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. How to Run

Activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the scripts in this order:

```python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

## 2. Data Findings

- Raw shipment records: 1,224
- Cleaned shipment records: 1,200
- Delayed shipments: 278
- Overall delay rate: 23.17%
- Tier 1 delay rate: 4.0%
- Tier 2 delay rate: 21.5%
- Tier 3 delay rate: 44.0%
- Shipments over 600 km had an observed delay rate of 85.23%.

## 3. Model Results (Regression)

The dataset was divided into 840 training, 180 validation, and 180 test records.

Final model: Linear Regression.

- MAE: 3.5671 hours
- MSE: 20.4625
- RMSE: 4.5236 hours
- R-squared: 0.9493

## 4. Model Results (Classification)

Classification test results at a probability threshold of 0.5:

- Accuracy: 86.67%
- Precision: 68.89%
- Recall: 75.61%
- F1-score: 72.09%
- True negatives: 125
- False positives: 14
- False negatives: 10
- True positives: 31

## 5. Operations Recommendation

- Prioritize longer-distance shipments.
- Monitor Tier 3 hubs because they have the highest observed delay rate.
- Use model predictions to identify potentially delayed shipments.
- Escalate high-risk shipments to hub managers and contact customers when appropriate.

## 6. Production Plan

1. Validate the pipeline on representative real-world shipment data.
2. Integrate predictions with shipment-tracking systems.
3. Record predicted and actual delivery outcomes.
4. Monitor data quality and model performance.
5. Retrain and validate the model periodically.
6. Retain human oversight for operational decisions.

## Limitations

- The project uses generated data, so real-world performance may differ.
- Correlation does not establish causation.
- The model requires validation on real operational data before deployment.
