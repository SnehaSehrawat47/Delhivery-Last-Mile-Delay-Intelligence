
import os
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features


DATA_PATH = "data/new_shipments.csv"
HUBS_PATH = "data/hubs.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the new parcels, hub information, and packaged model
    new_shipments = pd.read_csv(DATA_PATH)
    hubs = pd.read_csv(HUBS_PATH)

    model = joblib.load(
        os.path.join(MODEL_DIR, "delay_model.pkl")
    )

    threshold = float(
        pd.read_csv(
            os.path.join(MODEL_DIR, "delay_threshold.csv")
        )["threshold"].iloc[0]
    )

    feature_order = pd.read_csv(
        os.path.join(MODEL_DIR, "feature_order.csv")
    )["feature"].tolist()

    impute_df = pd.read_csv(
        os.path.join(MODEL_DIR, "impute_values.csv")
    )
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    segments = pd.read_csv(
        os.path.join(OUTPUT_DIR, "hub_segments.csv")
    )[["hub_id", "segment"]]

    # Use the shared preprocessing functions from shipment_prep.py
    cleaned, _ = clean_shipments(new_shipments, medians)
    featured = build_features(cleaned, hubs)

    # Ensure features are in the exact training order
    X_new = featured[feature_order]

    probabilities = model.predict_proba(X_new)[:, 1]
    flagged = (probabilities >= threshold).astype(int)

    actions = pd.DataFrame({
        "shipment_id": featured["shipment_id"],
        "hub_id": featured["hub_id"],
        "delay_probability": probabilities.round(4),
        "flagged": flagged,
    })

    actions = actions.merge(
        segments,
        on="hub_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Assign an operational action to each parcel
    actions["action"] = "No action"

    actions.loc[
        (actions["flagged"] == 1)
        & (actions["segment"] == "Escalate"),
        "action",
    ] = "Escalate to hub manager"

    actions.loc[
        (actions["flagged"] == 1)
        & (actions["segment"] != "Escalate"),
        "action",
    ] = "Proactive customer call"

    # Required output column order
    actions = actions[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action",
        ]
    ]

    output_path = os.path.join(
        OUTPUT_DIR, "new_shipment_actions.csv"
    )
    actions.to_csv(output_path, index=False, float_format="%.4f")

    print("Prediction completed.")
    print("New parcels scored:", len(actions))
    print("Flagged parcels:", int(actions["flagged"].sum()))
    print(
        "Hub-manager escalations:",
        int((actions["action"] == "Escalate to hub manager").sum()),
    )
    print("\nAction counts:")
    print(actions["action"].value_counts().to_string())
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
