
import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/modelling_table.csv"
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # 1. Create one profile row for each hub
    profile = (
        df.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    profile.to_csv(
        os.path.join(OUTPUT_DIR, "hub_profile.csv"),
        index=False,
        float_format="%.4f",
    )

    # 2. Scale the four hub-profile columns
    cluster_features = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(profile[cluster_features])

    # 3. Compare cluster separation for K = 2, 3, 4, 5
    separation_rows = []

    for k in [2, 3, 4, 5]:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(scaled)

        sizes = sorted(
            np.bincount(labels).tolist(), reverse=True
        )

        same_distances = []
        different_distances = []

        for i in range(len(profile)):
            for j in range(i + 1, len(profile)):
                distance = np.linalg.norm(scaled[i] - scaled[j])

                if labels[i] == labels[j]:
                    same_distances.append(distance)
                else:
                    different_distances.append(distance)

        within_mean = np.mean(same_distances)
        across_mean = np.mean(different_distances)

        separation_rows.append({
            "K": k,
            "across_over_within": across_mean / within_mean,
            "cluster_sizes": "-".join(map(str, sizes)),
        })

    pd.DataFrame(separation_rows).to_csv(
        os.path.join(OUTPUT_DIR, "part_d_separation.csv"),
        index=False,
        float_format="%.4f",
    )

    # 4. Fit the required three-cluster solution
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )
    profile["cluster"] = kmeans.fit_predict(scaled)

    # Give names based on the cluster's mean hub delay rate
    cluster_order = (
        profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    segment_names = {
        cluster_order[0]: "Standard",
        cluster_order[1]: "Watch",
        cluster_order[2]: "Escalate",
    }

    profile["segment"] = profile["cluster"].map(segment_names)

    # 5. Create hub_segments.csv
    hub_info = (
        df[["hub_id", "hub_city", "hub_tier"]]
        .drop_duplicates("hub_id")
    )

    segments = profile[["hub_id", "segment"]].merge(
        hub_info, on="hub_id", how="left", validate="one_to_one"
    )

    segments = segments.sort_values("hub_id").reset_index(drop=True)

    segments.to_csv(
        os.path.join(OUTPUT_DIR, "hub_segments.csv"),
        index=False,
    )

    # 6. Summarize each segment using hub-level means
    segment_profile = (
        profile.groupby("segment", as_index=False)
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
    )

    segment_order = ["Standard", "Watch", "Escalate"]
    segment_profile["segment"] = pd.Categorical(
        segment_profile["segment"],
        categories=segment_order,
        ordered=True,
    )
    segment_profile = (
        segment_profile.sort_values("segment")
        .reset_index(drop=True)
    )
    segment_profile["segment"] = segment_profile["segment"].astype(str)

    segment_profile.to_csv(
        os.path.join(OUTPUT_DIR, "segment_profile.csv"),
        index=False,
        float_format="%.4f",
    )

    # 7. Find hubs whose tier disagrees with the segment
    expected_segment = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    segments["expected_segment"] = segments["hub_tier"].map(
        expected_segment
    )

    mismatches = segments[
        segments["segment"] != segments["expected_segment"]
    ][["hub_id", "segment", "hub_city", "hub_tier"]]

    mismatches.to_csv(
        os.path.join(OUTPUT_DIR, "tier_vs_segment_mismatch.csv"),
        index=False,
    )

    print("Part D segmentation completed.")
    print("\nHub profile:", len(profile), "hubs")
    print("\nSeparation results:")
    print(pd.DataFrame(separation_rows).to_string(index=False))
    print("\nSegment profile:")
    print(segment_profile.to_string(index=False))
    print("\nTier/segment mismatches:")
    print(mismatches.to_string(index=False))
    print("\nPart D CSV files saved in outputs/.")


if __name__ == "__main__":
    main()
