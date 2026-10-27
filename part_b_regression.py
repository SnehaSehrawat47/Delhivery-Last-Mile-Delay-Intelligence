
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

def get_metrics(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": round(mean_absolute_error(y_true, predictions), 4),
        "MSE": round(mse, 4),
        "RMSE": round(np.sqrt(mse), 4),
        "R2": round(r2_score(y_true, predictions), 4),
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Split: 70% train, 15% validation, 15% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 2. Fit the scaler on training rows only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    checks = [
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {
            "metric": "train_scaled_max_abs_mean",
            "value": round(float(np.abs(X_train_scaled.mean(axis=0)).max()), 4),
        },
        {
            "metric": "test_scaled_max_abs_mean",
            "value": round(float(np.abs(X_test_scaled.mean(axis=0)).max()), 4),
        },
    ]

    pd.DataFrame(checks).to_csv(
        "outputs/part_b_checks.csv", index=False
    )

    # 3. Mean baseline.
    baseline_predictions = np.full(len(y_val), y_train.mean())
    metric_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            **get_metrics(y_val, baseline_predictions),
        }
    ]

    # 4. Find the best decision-tree depth using validation RMSE.
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    tree_models = {}

    for depth in depths:
        tree = DecisionTreeRegressor(
            max_depth=depth, random_state=42
        )
        tree.fit(X_train, y_train)

        train_predictions = tree.predict(X_train)
        val_predictions = tree.predict(X_val)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_predictions)
        )
        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_predictions)
        )

        depth_rows.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": round(float(train_rmse), 4),
            "val_rmse": round(float(val_rmse), 4),
        })
        tree_models[depth] = tree

    pd.DataFrame(depth_rows).to_csv(
        "outputs/part_b_tree_depth.csv", index=False
    )

    best_depth = min(
        depths,
        key=lambda d: (
            next(r["val_rmse"] for r in depth_rows
                 if r["max_depth"] == (
                     "unlimited" if d is None else d
                 )),
            float("inf") if d is None else d,
        ),
    )

    # 5. Four candidate models.
    candidates = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "decision_tree": tree_models[best_depth],
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        metric_rows.append({
            "model": name,
            "split": "val",
            **get_metrics(y_val, predictions),
        })

    # 6. Gradient-boosting learning-rate experiment.
    learning_rate_rows = []

    for lr in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        learning_rate_rows.append({
            "learning_rate": lr,
            "val_rmse": round(
                float(np.sqrt(mean_squared_error(y_val, predictions))), 4
            ),
            "val_r2": round(float(r2_score(y_val, predictions)), 4),
        })

    pd.DataFrame(learning_rate_rows).to_csv(
        "outputs/part_b_gb_learning_rate.csv", index=False
    )

    # 7. Cross-validate the two best models by validation RMSE.
    candidate_rows = [
        row for row in metric_rows if row["model"] in candidates
    ]
    ranked = sorted(candidate_rows, key=lambda row: row["RMSE"])
    top_two = [row["model"] for row in ranked[:2]]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for name in top_two:
        scores = cross_val_score(
            candidates[name],
            X_train_val,
            y_train_val,
            cv=cv,
        )
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })

    pd.DataFrame(cv_rows).to_csv(
        "outputs/part_b_cv.csv", index=False
    )

    # 8. Evaluate the validation winner on the test set once.
    winner_name = ranked[0]["model"]
    winner = candidates[winner_name]
    test_predictions = winner.predict(X_test)

    metric_rows.append({
        "model": winner_name,
        "split": "test",
        **get_metrics(y_test, test_predictions),
    })

    pd.DataFrame(
        metric_rows,
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
    ).round(4).to_csv(
        "outputs/part_b_metrics.csv", index=False
    )

    print("Part B completed successfully.")
    print("Split sizes:", len(X_train), len(X_val), len(X_test))
    print("Best tree depth:", best_depth)
    print("Validation ranking:")
    print(pd.DataFrame(ranked)[["model", "RMSE", "R2"]].to_string(index=False))
    print("Top two for cross-validation:", top_two)
    print("Final test model:", winner_name)
    print("Test metrics:", get_metrics(y_test, test_predictions))
    print("Created four CSV files in outputs/.")


if __name__ == "__main__":
    main()
