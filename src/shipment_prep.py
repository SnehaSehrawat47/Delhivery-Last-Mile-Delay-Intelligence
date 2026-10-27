
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

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

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean shipment data and return the cleaned data plus imputation medians."""
    df = raw.copy()

    # 1. Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # 2. Standardize service type.
    df["service_type"] = (
        df["service_type"].astype("string").str.strip().str.title()
    )

    # 3. Fill missing payment modes.
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. Convert declared values to numeric INR.
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    df = df.drop(columns=["declared_value"])

    # 5. Convert dispatch time to datetime.
    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"], errors="coerce"
    )

    # 6. Treat implausible parcel weights as missing.
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Compute or reuse historical medians.
    columns_to_fill = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            col: float(df[col].median())
            for col in columns_to_fill
        }
    else:
        medians = {col: float(medians[col]) for col in columns_to_fill}

    for col in columns_to_fill:
        df[col] = df[col].astype(float).fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Add hub information and model features without using delivery time."""
    df = clean.copy()

    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    df = df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (
        df["payment_mode"].astype("string").str.strip().str.upper() == "COD"
    ).astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log1p(df["declared_value_inr"])

    return df
