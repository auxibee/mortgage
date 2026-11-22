"""Train and evaluate a mortgage maximum-loan regression pipeline.

Dataset expected: new_mortage_data_set(1).csv
Target: Max Loan Amount (USD)

The default feature policy excludes Gender, Age, Married and Education.
Job and Area are retained as candidate predictors but should receive proxy-
fairness review before any real lending use.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TARGET = "Max Loan Amount (USD)"
NUMERIC_FEATURES = [
    "Employment Years",
    "Annual Income (USD)",
    "Interest Rate",
    "Down Payment (USD)",
    "Credit Score",
    "Existing Monthly Debt (USD)",
    "Loans Repaid",
]
CATEGORICAL_FEATURES = ["Job", "Area"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_COLUMNS = ["Gender", "Age", "Married", "Education"]


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def make_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=5, max_features=1.0,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=60, learning_rate=0.05, max_depth=3,
            loss="huber", random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=5, max_features=1.0,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def regression_metrics(y_true: pd.Series | np.ndarray,
                       y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_and_validate_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Convert numeric columns safely; invalid strings become NaN and are imputed.
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    dropped_missing_target = before - len(df)
    duplicate_count = int(df.duplicated().sum())
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Basic domain checks: reject impossible values from model training.
    valid = (
        (X["Employment Years"].isna() | (X["Employment Years"] >= 0)) &
        (X["Annual Income (USD)"].isna() | (X["Annual Income (USD)"] > 0)) &
        (X["Down Payment (USD)"].isna() | (X["Down Payment (USD)"] >= 0)) &
        (X["Existing Monthly Debt (USD)"].isna() | (X["Existing Monthly Debt (USD)"] >= 0)) &
        (X["Loans Repaid"].isna() | (X["Loans Repaid"] >= 0)) &
        (X["Credit Score"].isna() | X["Credit Score"].between(300, 850)) &
        (y >= 0)
    )
    invalid_rows = int((~valid).sum())
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    audit = {
        "source_file": csv_path.name,
        "rows_read": int(before),
        "rows_used": int(len(X)),
        "columns": df.columns.tolist(),
        "missing_values_by_column": df[required].isna().sum().astype(int).to_dict(),
        "duplicate_rows_detected": duplicate_count,
        "rows_dropped_missing_target": int(dropped_missing_target),
        "rows_excluded_by_basic_domain_checks": invalid_rows,
        "features_used": FEATURES,
        "features_excluded_by_default": EXCLUDED_COLUMNS,
        "target": TARGET,
    }
    return X, y, audit


def make_explainer(model: Any, background: np.ndarray):
    if isinstance(model, LinearRegression):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def create_shap_outputs(pipeline: Pipeline, X_test: pd.DataFrame,
                         output_dir: Path, max_explain: int = 1000,
                         reason_sample: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Use a bounded sample to keep explanation time and memory manageable.
    explain_n = min(max_explain, len(X_test))
    X_explain = X_test.sample(n=explain_n, random_state=RANDOM_STATE)
    transformed = np.asarray(preprocessor.transform(X_explain), dtype=float)
    background_n = min(200, len(transformed))
    background = transformed[:background_n]
    explainer = make_explainer(estimator, background)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs,
    }).sort_values("Mean_Absolute_SHAP", ascending=False)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    # Summary plot: useful for reviewing direction and spread of contributions.
    plt.figure()
    shap.summary_plot(shap_values, transformed, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    # Persist a small transformed background for the later API explanation helper.
    joblib.dump(background, output_dir / "shap_background.joblib")

    # Human-readable reason codes for a sample of held-out test applications.
    n_reasons = min(reason_sample, len(X_explain))
    base_values = getattr(explainer, "expected_value", 0.0)
    if isinstance(base_values, np.ndarray):
        base_values = float(np.ravel(base_values)[0])
    elif isinstance(base_values, (list, tuple)):
        base_values = float(base_values[0])
    else:
        base_values = float(base_values)

    reason_rows: list[dict[str, Any]] = []
    X_reason = X_explain.iloc[:n_reasons]
    pred_reason = pipeline.predict(X_reason)
    for row_pos, (idx, row) in enumerate(X_reason.iterrows()):
        contributions = shap_values[row_pos]
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]
        for rank, feature_idx in enumerate(top_idx, start=1):
            contribution = float(contributions[feature_idx])
            direction = "increased" if contribution >= 0 else "decreased"
            reason_rows.append({
                "Application_Row_Index": int(idx),
                "Predicted_Max_Loan_USD": float(pred_reason[row_pos]),
                "Baseline_Prediction_USD": base_values,
                "Reason_Rank": rank,
                "Reason_Code": f"RC{rank:02d}",
                "Feature": feature_names[feature_idx],
                "Applicant_Feature_Value": str(row.get(feature_names[feature_idx], "")),
                "SHAP_Contribution_USD": contribution,
                "Direction_vs_Baseline": direction,
                "Explanation": (
                    f"{feature_names[feature_idx]} {direction} the model estimate "
                    f"relative to the model baseline by approximately "
                    f"${abs(contribution):,.2f}."
                ),
            })
    reasons = pd.DataFrame(reason_rows)
    reasons.to_csv(output_dir / "sample_shap_reason_codes.csv", index=False)
    return importance, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mortgage maximum-loan regression models.")
    parser.add_argument("--data", default="new_mortage_data_set(1).csv",
                        help="Path to the mortgage CSV dataset")
    parser.add_argument("--output", default="artifacts",
                        help="Directory where model and reports will be saved")
    parser.add_argument("--explain-sample", type=int, default=100,
                        help="Maximum held-out rows used for SHAP summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csv_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, audit = load_and_validate_data(csv_path)
    logging.info("Loaded %s usable rows from %s", len(X), csv_path.name)

    # 70% train, 15% validation, 15% final untouched test set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    validation_rows = []
    fitted_candidates: dict[str, Pipeline] = {}
    for name, estimator in make_models().items():
        logging.info("Training %s", name)
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_valid)
        row = {"Model": name, **regression_metrics(y_valid, pred)}
        validation_rows.append(row)
        fitted_candidates[name] = pipeline
        logging.info("%s validation RMSE: $%s", name, f"{row['RMSE']:,.2f}")

    validation_results = pd.DataFrame(validation_rows).sort_values("RMSE")
    validation_results.to_csv(output_dir / "validation_model_comparison.csv", index=False)
    best_name = str(validation_results.iloc[0]["Model"])
    logging.info("Selected %s using validation RMSE", best_name)

    # Refit selected pipeline on train + validation; test set remains untouched.
    X_trainval = pd.concat([X_train, X_valid], axis=0)
    y_trainval = pd.concat([y_train, y_valid], axis=0)
    final_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", make_models()[best_name]),
    ])
    final_pipeline.fit(X_trainval, y_trainval)
    test_predictions = final_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)

    test_predictions_df = pd.DataFrame({
        "Actual_Max_Loan_USD": y_test,
        "Predicted_Max_Loan_USD": test_predictions,
        "Absolute_Error_USD": np.abs(y_test.to_numpy() - test_predictions),
    }, index=y_test.index).sort_index()
    test_predictions_df.to_csv(output_dir / "test_set_predictions.csv", index_label="Source_Row_Index")

    # Save model and metadata in an API-friendly form.
    model_path = output_dir / "mortgage_model_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)
    metadata = {
        "selected_model": best_name,
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": EXCLUDED_COLUMNS,
        "currency": "USD (as labelled in source dataset)",
        "random_state": RANDOM_STATE,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "selection_metric": "Validation RMSE (lower is better)",
        "final_test_metrics": test_metrics,
        "model_file": model_path.name,
        "shap_version": shap.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "warning": (
            "Prototype only. The target is historical Max Loan Amount, not an observed "
            "approval/default outcome or a verified affordability limit. Validate policy, "
            "fairness, data provenance and out-of-time performance before real lending use."
        ),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    # Persist exact input column contract for API validation.
    joblib.dump({"features": FEATURES, "numeric_features": NUMERIC_FEATURES,
                 "categorical_features": CATEGORICAL_FEATURES},
                output_dir / "input_schema.joblib")

    importance, reasons = create_shap_outputs(
        final_pipeline, X_test, output_dir,
        max_explain=args.explain_sample,
        reason_sample=100,
    )

    summary = {
        "selected_model": best_name,
        "validation_comparison": validation_results.to_dict(orient="records"),
        "final_test_metrics": test_metrics,
        "test_rows": int(len(X_test)),
        "top_shap_features": importance.head(10).to_dict(orient="records"),
        "reason_code_rows": int(len(reasons)),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Final test metrics: MAE=$%s | RMSE=$%s | R2=%.4f",
                 f"{test_metrics['MAE']:,.2f}", f"{test_metrics['RMSE']:,.2f}", test_metrics["R2"])
    logging.info("Saved pipeline and reports to %s", output_dir)


if __name__ == "__main__":
    main()
