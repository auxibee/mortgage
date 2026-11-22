"""PostgreSQL persistence layer for mortgage scoring and underwriting.

Configure POSTGRES_DSN in the environment or .env file, for example:
postgresql://mortgage_app:change_me@localhost:5432/MortgageAnalytics
Requires psycopg (v3).
"""
from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


def get_connection_string() -> str:
    value = os.getenv("POSTGRES_DSN", "").strip()
    if not value:
        raise RuntimeError("POSTGRES_DSN is not configured. Set it in your .env file.")
    return value


class MortgageRepository:
    """Store application, scoring, SHAP and underwriting events in PostgreSQL."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or get_connection_string()

    def connect(self):
        return psycopg.connect(
            self.connection_string,
            connect_timeout=8,
            row_factory=dict_row,
        )

    def health_check(self) -> bool:
        with self.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return True

    def save_scoring_event(
        self,
        application_id: str,
        applicant: dict[str, Any],
        result: dict[str, Any],
    ) -> int:
        """Persist one scoring attempt atomically, including failed validations."""
        errors = result.get("errors", [])
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO mortgage_applications
                    (application_id, employment_years, annual_income_usd, interest_rate,
                     down_payment_usd, credit_score, existing_monthly_debt_usd,
                     loans_repaid, job, area, applicant_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (application_id) DO NOTHING""",
                    (
                        application_id,
                        applicant.get("Employment Years"),
                        applicant.get("Annual Income (USD)"),
                        applicant.get("Interest Rate"),
                        applicant.get("Down Payment (USD)"),
                        applicant.get("Credit Score"),
                        applicant.get("Existing Monthly Debt (USD)"),
                        applicant.get("Loans Repaid"),
                        applicant.get("Job"),
                        applicant.get("Area"),
                        json.dumps(applicant, default=str),
                    ),
                )

                row = cur.execute(
                    """INSERT INTO mortgage_predictions
                    (application_id, predicted_max_loan_usd, baseline_prediction_usd,
                     validation_status, scoring_status, validation_errors_json, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING prediction_id""",
                    (
                        application_id,
                        result.get("predicted_max_loan_usd"),
                        result.get("baseline_prediction_usd"),
                        result.get("validation_status", "UNKNOWN"),
                        result.get("status", "UNKNOWN"),
                        json.dumps(errors) if errors else None,
                        result.get("model_version"),
                    ),
                ).fetchone()
                prediction_id = int(row["prediction_id"])

                reasons = result.get("reason_codes", [])
                if reasons:
                    cur.executemany(
                        """INSERT INTO prediction_reasons
                        (prediction_id, reason_code, feature_name, contribution_usd,
                         direction_vs_baseline, explanation)
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                        [
                            (
                                prediction_id,
                                reason.get("code", ""),
                                reason.get("feature", ""),
                                reason.get("contribution_usd", 0),
                                reason.get("direction_vs_baseline", "unknown"),
                                reason.get("explanation", ""),
                            )
                            for reason in reasons
                        ],
                    )
        # psycopg connection context commits on success and rolls back on exception.
        return prediction_id



    def save_underwriting_decision(
        self,
        application_id: str,
        prediction_id: int,
        decision: str,
        predicted_amount: float | None,
        approved_amount: float | None,
        override_reason: str | None,
        underwriter_id: str,
    ) -> int:
        allowed = {"ACCEPTED", "OVERRIDDEN", "DECLINED", "FURTHER_REVIEW"}
        decision = decision.upper()
        if decision not in allowed:
            raise ValueError(f"Unsupported underwriting decision: {decision}")
        if not underwriter_id or not underwriter_id.strip():
            raise ValueError("Underwriter ID is required.")
        if decision in {"ACCEPTED", "OVERRIDDEN"} and approved_amount is None:
            raise ValueError("Approved amount is required for accepted or overridden decisions.")
        if decision == "OVERRIDDEN" and not (override_reason or "").strip():
            raise ValueError("An override reason is required when overriding the model recommendation.")

        with self.connect() as conn:
            row = conn.execute(
                """INSERT INTO underwriting_decisions
                (application_id, prediction_id, decision, predicted_amount_usd,
                 approved_amount_usd, override_reason, underwriter_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING decision_id""",
                (
                    application_id, prediction_id, decision, predicted_amount,
                    approved_amount, override_reason, underwriter_id.strip(),
                ),
            ).fetchone()
            return int(row["decision_id"])

    def recent_operations(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT application_id AS "ApplicationID",
                          prediction_id AS "PredictionID",
                          predicted_max_loan_usd AS "PredictedMaxLoanUSD",
                          validation_status AS "ValidationStatus",
                          scoring_status AS "ScoringStatus",
                          model_version AS "ModelVersion",
                          prediction_date AS "PredictionDate",
                          decision AS "Decision",
                          approved_amount_usd AS "ApprovedAmountUSD",
                          underwriter_id AS "UnderwriterID",
                          decision_date AS "DecisionDate"
                   FROM vw_mortgage_operations
                   ORDER BY prediction_date DESC
                   LIMIT %s""",
                (limit,),
            ).fetchall()
        return rows
