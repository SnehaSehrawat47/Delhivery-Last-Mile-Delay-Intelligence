# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. How to Run

Activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Generate the data and run the scripts in this order:

```powershell
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

## 2. Data Findings

- Raw shipment records: 1,224.
- Exact duplicate rows removed: 24.
- Cleaned shipment records: 1,200.
- Delayed shipments: 278 (23.17% overall delay rate).
- Missing parcel weights before imputation: 24.
- Scanner-fault parcel-weight sentinel rows identified: 5.
- Missing values in the final modelling features and target: 0.
- Express shipments had the highest delay rate at 48.97%, compared with 20.56% for Standard and 8.57% for Economy.
- Delay rates increased with hub tier: Tier 1 = 4.0%, Tier 2 = 21.5%, and Tier 3 = 44.0%.
- Delay rates increased with distance: 0–200 km = 2.19%; 201–400 km = 13.68%; 401–600 km = 43.64%; over 600 km = 85.23%.
- Delivery promised hours had the strongest observed correlation with delivery hours (0.8372), followed by distance in kilometres (0.4013).
- Declared shipment value was log-transformed to reduce the influence of its right-skewed distribution. Its mean (₹1,533.05) was higher than its median (₹1,121.50), indicating that higher-value shipments pulled the mean upward.

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

### Hub prioritisation

- **Escalate:** 9 hubs, with a mean delay rate of 46.00%, mean delivery time of 64.65 hours, and mean distance of 470.48 km. These hubs should receive priority operational review.
- **Watch:** 8 hubs, with a mean delay rate of 15.50%. Monitor their performance and investigate rising delay rates.
- **Standard:** 7 hubs, with a mean delay rate of 2.57%. Continue routine monitoring.
- Review the four hub tier/segment mismatches rather than relying on hub tier alone: HUB07 (Kolkata, Tier 1) is Watch; HUB13 (Nagpur, Tier 2) and HUB15 (Kochi, Tier 2) are Escalate; HUB22 (Dehradun, Tier 3) is Watch. Actual observed performance should inform operational priorities.

### New-shipment actions

- The pipeline scored 48 new shipments. At the 0.5 classification threshold, 8 were flagged: 6 for hub-manager escalation and 2 for proactive customer calls. The remaining 40 received no action.
- Classification precision was 68.89% and recall was 75.61% on the held-out test set. In practical terms, some flagged parcels will not be delayed, and some delayed parcels will be missed; predictions should support, not replace, operational judgement.
- Prioritise long-distance shipments and review high-risk parcels early so teams have time to intervene.

## 6. Production Plan

1. **Accuracy:** Track regression MAE and RMSE, and classification precision, recall, and F1-score on newly labelled shipments. Investigate if MAE exceeds 5 hours or classification recall falls below 70%.
2. **Reliability:** Keep a saved-model reload check and verify that reloaded predictions match the original predictions. The current reload check has a maximum probability difference of 0.0.
3. **Reproducibility:** Preserve the ordered scripts, requirements file, fixed data splits, and model configuration. Rerun the pipeline from data generation through prediction to reproduce results.
4. **Consistent preprocessing:** Apply the same cleaning, imputation, feature construction, and feature order during training and inference.
5. **Speed:** Measure prediction latency and monitor whether scoring new shipment batches meets operational service-level targets.
6. **Maintainability:** Keep cleaning, modelling, packaging, and prediction steps separated into scripts, with generated results stored in the `outputs/` directory.
7. **Monitorability:** Track weekly delay rate, missing input values, and the proportion of shipments flagged for action. Investigate if the delay rate rises above 30%, required input fields have missing values after preprocessing, or the flagged share changes substantially from its baseline.
8. **Retraining schedule:** Review performance monthly and retrain when performance degrades or shipment patterns change materially. Validate any replacement model before deployment.
9. **Sale-event monitoring:** Compare shipments during the 15–28 September sale-event window, identified by `sale_event_flag`, with comparable sale-event periods where available. Do not compare sale-event performance only with ordinary weeks because shipment volume and delivery conditions may differ.
10. **Data drift:** Monitor changes in distance, parcel weight, declared value, service type, hub tier, and sale-event frequency. Investigate material shifts before trusting predictions for operational decisions.
11. **Human oversight:** Use model outputs to prioritise investigation; hub managers retain responsibility for escalation and customer communication.

## 7. Limitations

- **Synthetic data:** The pipeline uses generated shipment data. Its metrics may not represent real-world performance, and validation on historical operational data is required before deployment.
- **Correlation versus causation:** Relationships between distance, promised delivery time, hub tier, and delays do not establish that any single factor causes delays.
- **Classification errors:** False positives can lead to unnecessary escalations, while false negatives can leave genuinely delayed shipments unidentified. Predictions require human review.
- **Distribution shift:** Sale events, seasonal demand, changes in hub operations, and changes in shipment mix may reduce model performance over time.
- **Model generalisation:** The reported results come from the current train/validation/test split. Performance on new hubs, unseen conditions, or future shipments may differ.
- **Operational integration:** The current project produces CSV outputs and packaged model files; it has not yet been validated as a live production service.