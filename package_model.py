
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


DATA_PATH = "data/modelling_table.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

TARGET = "is_delayed"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read the selected configuration from Part C
    config = pd.read_csv(
        os.path.join(OUTPUT_DIR, "part_c_final_config.csv")
    ).iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    min_samples_leaf = int(config["min_samples_leaf"])

    if str(config["max_depth"]).lower() in ("unlimited", "none", "nan"):
        max_depth = None
    else:
        max_depth = int(config["max_depth"])

    threshold = float(config["threshold"])

    # Rebuild the exact stratified split
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    # Build and fit the final model on training rows only
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Save the reusable model and supporting files
    joblib.dump(
        pipeline,
        os.path.join(MODEL_DIR, "delay_model.pkl"),
    )

    pd.DataFrame({"threshold": [threshold]}).to_csv(
        os.path.join(MODEL_DIR, "delay_threshold.csv"),
        index=False,
    )

    pd.DataFrame({"feature": FEATURES}).to_csv(
        os.path.join(MODEL_DIR, "feature_order.csv"),
        index=False,
    )

    # The three imputation medians from shipment preparation
    medians = {
        "distance_km": df["distance_km"].median(),
        "parcel_weight_kg": df["parcel_weight_kg"].median(),
        "declared_value_inr": df["declared_value_inr"].median(),
    }

    pd.DataFrame([
        {"column": column, "median": value}
        for column, value in medians.items()
    ]).to_csv(
        os.path.join(MODEL_DIR, "impute_values.csv"),
        index=False,
        float_format="%.4f",
    )

    # Verify the saved model by reloading it
    loaded_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "delay_model.pkl")
    )

    original_probabilities = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probabilities = loaded_pipeline.predict_proba(X_test)[:, 1]

    max_difference = abs(
        original_probabilities - reloaded_probabilities
    ).max()

    pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {
            "metric": "reload_max_abs_probability_diff",
            "value": max_difference,
        },
        {"metric": "threshold", "value": threshold},
    ]).to_csv(
        os.path.join(OUTPUT_DIR, "part_d_reload_check.csv"),
        index=False,
        float_format="%.12f",
    )

    print("Model packaging completed.")
    print("Training rows:", len(X_train))
    print("Test rows scored for reload check:", len(X_test))
    print("Threshold:", threshold)
    print("Maximum probability difference:", max_difference)
    print("Saved model and supporting files in models/.") 


if __name__ == "__main__":
    main()
