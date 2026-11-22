"""Reusable inference helpers for a future FastAPI scoring endpoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression


class MortgageScorer:
    def __init__(self, artifact_dir: str | Path = "artifacts"):
        self.artifact_dir = Path(artifact_dir)
        self.pipeline = joblib.load(self.artifact_dir / "mortgage_model_pipeline.joblib")
        self.metadata = json.loads((self.artifact_dir / "model_metadata.json").read_text())
        self.background = joblib.load(self.artifact_dir / "shap_background.joblib")
        self.features = self.metadata["features"]
        self.feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        estimator = self.pipeline.named_steps["model"]
        if isinstance(estimator, LinearRegression):
            self.explainer = shap.LinearExplainer(estimator, self.background)
        else:
            self.explainer = shap.TreeExplainer(estimator)

    def validate(self, applicant: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.features:
            if field not in applicant or applicant[field] is None:
                errors.append(f"Missing required field: {field}")
        if errors:
            return errors
        numeric_rules = {
            "Employment Years": lambda v: v >= 0,
            "Annual Income (USD)": lambda v: v > 0,
            "Down Payment (USD)": lambda v: v >= 0,
            "Credit Score": lambda v: 300 <= v <= 850,
            "Existing Monthly Debt (USD)": lambda v: v >= 0,
            "Loans Repaid": lambda v: v >= 0,
            "Interest Rate": lambda v: v > 0,
        }
        for field, rule in numeric_rules.items():
            try:
                if not rule(float(applicant[field])):
                    errors.append(f"Invalid value for {field}: {applicant[field]}")
            except (TypeError, ValueError):
                errors.append(f"Non-numeric value for {field}: {applicant[field]}")
        return errors

    def score(self, applicant: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
        errors = self.validate(applicant)
        if errors:
            return {
                "status": "MANUAL_UNDERWRITING",
                "validation_status": "FAILED",
                "errors": errors,
                "predicted_max_loan_usd": None,
                "reason_codes": [],
                "model_version": self.metadata.get("selected_model", "unknown"),
            }

        row = pd.DataFrame([{field: applicant[field] for field in self.features}])
        prediction = float(self.pipeline.predict(row)[0])
        transformed = np.asarray(
            self.pipeline.named_steps["preprocessor"].transform(row), dtype=float
        )
        values = self.explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[0]
        contributions = np.asarray(values).reshape(-1)
        if contributions.ndim > 1:
            contributions = contributions[0]

        base = getattr(self.explainer, "expected_value", 0.0)
        base_value = float(np.ravel(base)[0])
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_n]
        reason_codes = []
        for rank, idx in enumerate(top_indices, start=1):
            amount = float(contributions[idx])
            direction = "increased" if amount >= 0 else "decreased"
            reason_codes.append({
                "code": f"RC{rank:02d}",
                "feature": self.feature_names[idx],
                "contribution_usd": round(amount, 2),
                "direction_vs_baseline": direction,
                "explanation": (
                    f"{self.feature_names[idx]} {direction} the model estimate "
                    f"relative to baseline by approximately ${abs(amount):,.2f}."
                ),
            })

        return {
            "status": "REVIEW_REQUIRED",
            "validation_status": "PASSED",
            "predicted_max_loan_usd": round(prediction, 2),
            "baseline_prediction_usd": round(base_value, 2),
            "reason_codes": reason_codes,
            "model_version": self.metadata.get("selected_model", "unknown"),
            "decision_note": "Prediction is decision support only; it is not an approval or verified safe lending limit.",
        }


if __name__ == "__main__":
    # Example local smoke test. Replace values with an actual application payload.
    scorer = MortgageScorer("artifacts")
    example = {
        "Employment Years": 15,
        "Annual Income (USD)": 120000,
        "Interest Rate": 0.05,
        "Down Payment (USD)": 60000,
        "Credit Score": 750,
        "Existing Monthly Debt (USD)": 1200,
        "Loans Repaid": 4,
        "Job": "Software Engineer",
        "Area": "Urban",
    }
    print(json.dumps(scorer.score(example), indent=2))
