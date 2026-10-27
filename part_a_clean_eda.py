
import os
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


def main():
    os.makedirs("outputs", exist_ok=True)

    # Load the original synthetic data
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Capture checks before cleaning
    raw_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())

    missing_distance = int(raw["distance_km"].isna().sum())
    missing_weight = int(
        (raw["parcel_weight_kg"].isna()
         | (raw["parcel_weight_kg"] == -1)).sum()
    )
    missing_value = int(raw["declared_value"].isna().sum())
    missing_payment = int(raw["payment_mode"].isna().sum())
    service_spellings_before = int(raw["service_type"].nunique())

    clean, medians = clean_shipments(raw)

    service_spellings_after = int(clean["service_type"].nunique())
    sentinel_weight_rows = int(
        (clean["parcel_weight_kg"] == -1).sum()
    )

    # Build features and target
    features = build_features(clean, hubs)
    features["is_delayed"] = (
        features["delivery_hours"] > features["promised_hours"]
    ).astype(int)

    features.to_csv(
        "data/modelling_table.csv",
        index=False,
        float_format="%.4f",
    )

    # Delay by service type
    service = (
        features.groupby("service_type")
        .agg(
            shipments=("is_delayed", "size"),
            delayed_count=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )
    service["group_type"] = "service_type"
    service["group"] = service["service_type"]
    service = service[
        ["group_type", "group", "shipments", "delayed_count", "delay_rate"]
    ]

    # Delay by hub tier
    tier = (
        features.groupby("hub_tier")
        .agg(
            shipments=("is_delayed", "size"),
            delayed_count=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )
    tier["group_type"] = "hub_tier"
    tier["group"] = tier["hub_tier"].astype(str)
    tier = tier[
        ["group_type", "group", "shipments", "delayed_count", "delay_rate"]
    ]

    # Delay by distance band
    bins = [0, 200, 400, 600, float("inf")]
    labels = ["0-200", "201-400", "401-600", "601+"]

    features["distance_band"] = pd.cut(
        features["distance_km"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distance = (
        features.groupby("distance_band", observed=False)
        .agg(
            shipments=("is_delayed", "size"),
            delayed_count=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )
    distance["group_type"] = "distance_band"
    distance["group"] = distance["distance_band"].astype(str)
    distance = distance[
        ["group_type", "group", "shipments", "delayed_count", "delay_rate"]
    ]

    delay_groups = pd.concat(
        [service, tier, distance], ignore_index=True
    )
    delay_groups.to_csv(
        "outputs/part_a_delay_by_group.csv",
        index=False,
        float_format="%.4f",
    )

    # Correlations required by the brief
    correlation_columns = [
        "delivery_hours",
        "promised_hours",
        "distance_km",
        "is_delayed",
    ]
    correlations = features[correlation_columns].corr()
    correlations.to_csv(
        "outputs/part_a_correlations.csv",
        float_format="%.4f",
    )

    # IQR outlier check for delivery hours
    q1 = features["delivery_hours"].quantile(0.25)
    q3 = features["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(
        (
            (features["delivery_hours"] < lower_bound)
            | (features["delivery_hours"] > upper_bound)
        ).sum()
    )

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", raw_rows - len(clean)),
        ("rows_after_dedup", len(clean)),
        ("missing_distance_km_before", missing_distance),
        ("missing_parcel_weight_before", missing_weight),
        ("missing_declared_value_before", missing_value),
        ("missing_payment_mode_before", missing_payment),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("declared_value_mean_inr", clean["declared_value_inr"].mean()),
        ("declared_value_median_inr", clean["declared_value_inr"].median()),
        ("log_declared_value_mean", features["log_declared_value"].mean()),
        ("log_declared_value_median", features["log_declared_value"].median()),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(features)),
        ("nulls_in_features_and_target",
         int(features.isna().sum().sum())),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("delayed_count", int(features["is_delayed"].sum())),
        ("delay_rate", features["is_delayed"].mean()),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower_bound),
        ("hours_upper_bound", upper_bound),
        ("hours_outlier_count", outlier_count),
        ("corr_delivery_promised",
         correlations.loc["delivery_hours", "promised_hours"]),
        ("corr_delivery_distance",
         correlations.loc["delivery_hours", "distance_km"]),
        ("corr_delayed_distance",
         correlations.loc["is_delayed", "distance_km"]),
    ]

    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
        "outputs/part_a_checks.csv",
        index=False,
        float_format="%.4f",
    )

    print("Part A completed.")
    print("Original rows:", raw_rows)
    print("Cleaned rows:", len(clean))
    print("Modelling rows:", len(features))
    print("Historical imputation medians:", medians)
    print("\nDelay rate by service:")
    print(service.to_string(index=False))
    print("\nDelay rate by hub tier:")
    print(tier.to_string(index=False))
    print("\nDelay rate by distance band:")
    print(distance.to_string(index=False))
    print("\nCreated Part A checks, group results, and correlations.")


if __name__ == "__main__":
    main()
