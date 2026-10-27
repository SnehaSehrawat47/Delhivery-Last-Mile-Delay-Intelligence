
import os
import ast
import json
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/modelling_table.csv"
OUTPUT_DIR = "outputs"
TARGET = "is_delayed"

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


def get_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def save_csv(data, filename):
    pd.DataFrame(data).to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False,
        float_format="%.4f",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Required stratified 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=42, stratify=y_temp
    )

    print("Split sizes:", len(X_train), len(X_val), len(X_test))
    print(
        "Delayed counts:",
        int(y_train.sum()), int(y_val.sum()), int(y_test.sum())
    )

    # 1. Baseline, logistic regression, and best decision tree
    model_rows = []

    baseline_predictions = [0] * len(y_val)
    model_rows.append({
        "model": "blind_all_on_time",
        **get_metrics(y_val, baseline_predictions),
    })

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    logistic.fit(X_train, y_train)
    model_rows.append({
        "model": "logistic_regression",
        **get_metrics(y_val, logistic.predict(X_val)),
    })

    tree_rows = []
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, random_state=42
        )
        tree.fit(X_train, y_train)
        metrics = get_metrics(y_val, tree.predict(X_val))
        tree_rows.append((depth, metrics["f1"], tree, metrics))

    # Highest F1, with smaller depth winning ties
    tree_rows.sort(key=lambda row: (-row[1], row[0]))
    best_depth, _, best_tree, best_tree_metrics = tree_rows[0]

    model_rows.append({
        "model": f"decision_tree_depth_{best_depth}",
        **best_tree_metrics,
    })

    default_forest = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    default_forest.fit(X_train, y_train)
    model_rows.append({
        "model": "random_forest_default",
        **get_metrics(y_val, default_forest.predict(X_val)),
    })

    save_csv(model_rows, "part_c_models.csv")

    # 2. Feature scores and forest importances using training rows only
    score_selector = SelectKBest(score_func=f_classif, k="all")
    score_selector.fit(X_train, y_train)

    importance_forest = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    importance_forest.fit(X_train, y_train)

    scores = score_selector.scores_
    ranks = pd.Series(scores).rank(
        ascending=False, method="min"
    ).astype(int)

    feature_rows = []
    for i, feature in enumerate(FEATURES):
        feature_rows.append({
            "feature": feature,
            "f_score": scores[i],
            "f_rank": ranks.iloc[i],
            "rf_importance": importance_forest.feature_importances_[i],
            "is_noise": feature in NOISE_COLUMNS,
        })

    save_csv(feature_rows, "part_c_feature_scores.csv")

    # 3. Compare feature-selection values of k
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipeline = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(
                n_estimators=100, random_state=42
            )),
        ])
        pipeline.fit(X_train, y_train)
        val_f1 = f1_score(
            y_val, pipeline.predict(X_val), zero_division=0
        )
        selectk_rows.append({"k": k, "val_f1": val_f1})

    save_csv(selectk_rows, "part_c_selectk.csv")

    # 4. Tune forests on the seven selected features
    selector = SelectKBest(f_classif, k=7)
    X_train_7 = selector.fit_transform(X_train, y_train)
    X_val_7 = selector.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
    )
    grid.fit(X_train_7, y_train)

    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    randomized.fit(X_train_7, y_train)

    grid_val_f1 = f1_score(
        y_val, grid.predict(X_val_7), zero_division=0
    )
    random_val_f1 = f1_score(
        y_val, randomized.predict(X_val_7), zero_division=0
    )

    default_7 = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )
    default_7.fit(X_train_7, y_train)
    default_val_f1 = f1_score(
        y_val, default_7.predict(X_val_7), zero_division=0
    )

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": json.dumps(grid.best_params_, sort_keys=True),
            "best_cv_accuracy": grid.best_score_,
            "val_f1": grid_val_f1,
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": json.dumps(
                randomized.best_params_, sort_keys=True
            ),
            "best_cv_accuracy": randomized.best_score_,
            "val_f1": random_val_f1,
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": json.dumps({
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_leaf": 1,
            }, sort_keys=True),
            "best_cv_accuracy": "",
            "val_f1": default_val_f1,
        },
    ]

    save_csv(tuning_rows, "part_c_tuning.csv")

    # 5. Select configuration using validation F1 and specified tie priority
    priority = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2,
    }
    winner = sorted(
        tuning_rows,
        key=lambda row: (-row["val_f1"], priority[row["search"]]),
    )[0]

    if winner["search"] == "grid_search":
        chosen_params = grid.best_params_
    elif winner["search"] == "randomized_search":
        chosen_params = randomized.best_params_
    else:
        chosen_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        }

    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(
            random_state=42, **chosen_params
        )),
    ])
    final_model.fit(X_train, y_train)

    # 6. Select threshold from validation recall; do not use test rows
    val_probabilities = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    threshold_rows = []
    for threshold in thresholds:
        predictions = (val_probabilities >= threshold).astype(int)
        threshold_rows.append({
            "threshold": threshold,
            **get_metrics(y_val, predictions),
        })

    save_csv(threshold_rows, "part_c_thresholds.csv")

    eligible = [
        row for row in threshold_rows if row["recall"] >= 0.80
    ]
    chosen_threshold = (
        max(eligible, key=lambda row: row["threshold"])["threshold"]
        if eligible else 0.1
    )

    # 7. Evaluate test set exactly once at the selected threshold
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= chosen_threshold).astype(int)
    test_metrics = get_metrics(y_test, test_predictions)

    save_csv([{
        "threshold": chosen_threshold,
        **test_metrics,
    }], "part_c_final_test.csv")

    save_csv([{
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": (
            chosen_params["max_depth"]
            if chosen_params["max_depth"] is not None
            else "unlimited"
        ),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": winner["search"],
    }], "part_c_final_config.csv")

    print("\nPart C completed.")
    print("Best decision-tree depth:", best_depth)
    print("Winning configuration:", winner["search"])
    print("Chosen alert threshold:", chosen_threshold)
    print("Test metrics:", test_metrics)
    print("Created Part C output CSV files in outputs/.")


if __name__ == "__main__":
    main()
