"""Streamlit front end for the mortgage maximum-loan prototype.

Run from this directory:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mortgage_inference import MortgageScorer
from sql_repository import MortgageRepository

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Mortgage Underwriting Decision Support",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading trained mortgage model and SHAP explainer...")
def load_scorer() -> MortgageScorer:
    return MortgageScorer(ARTIFACT_DIR)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


metadata = load_metadata()

st.title("🏠 Mortgage Underwriting Decision Support")
st.caption("Prototype application · Maximum-loan prediction with SHAP explanations · Human review required")

with st.sidebar:
    st.header("System information")
    st.write(f"**Selected model:** {metadata.get('selected_model', 'Not available')}")
    st.write(f"**Target:** {metadata.get('target', 'Max Loan Amount (USD)')}")
    st.write(f"**Model version:** {metadata.get('selected_model', '1.0')}")
    st.divider()
    st.subheader("PostgreSQL connection")
    if os.getenv("POSTGRES_DSN"):
        try:
            MortgageRepository("postgresql://postgres:7aKBRpabgqkZSYZC@db.ypyqiqkcrjwtmmkzcfxc.supabase.co:5432/postgres").health_check()
            st.success("PostgreSQL connected")
        except Exception as exc:
            st.error(f"PostgreSQL connection failed: {exc}")
    else:
        st.warning("PostgreSQL is not configured. Scoring works locally, but records will not be persisted.")
    st.divider()
    st.subheader("Important notice")
    st.warning(
        "This prototype estimates the dataset's historical maximum-loan target. "
        "It does not approve or decline mortgages, establish affordability, or "
        "calculate a legally safe lending limit. A qualified underwriter must "
        "review all information and applicable lending policies."
    )

try:
    scorer = load_scorer()
except Exception as exc:  # Show a useful setup message instead of a blank app.
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.info(
        "Confirm that artifacts/mortgage_model_pipeline.joblib, "
        "artifacts/model_metadata.json and artifacts/shap_background.joblib "
        "are present, then install the requirements in this project."
    )
    st.stop()

# Tabs keep the applicant workflow separate from model diagnostics.
tab_application, tab_model, tab_database = st.tabs(["New application / score", "Model information", "Database activity"])

with tab_application:
    st.subheader("Applicant information")
    st.write("Enter the fields used by the trained model. All monetary amounts are in USD, matching the source dataset.")

    with st.form("mortgage_application_form", clear_on_submit=False):
        st.markdown("#### Financial and credit information")
        col1, col2, col3 = st.columns(3)
        with col1:
            annual_income = st.number_input(
                "Annual income (USD)", min_value=0.0, value=120000.0,
                step=5000.0, help="Must be greater than zero."
            )
            credit_score = st.number_input(
                "Credit score", min_value=0, max_value=1000, value=750,
                step=1, help="Prototype validation accepts scores from 300 to 850."
            )
            down_payment = st.number_input(
                "Down payment (USD)", min_value=0.0, value=60000.0, step=2500.0
            )
        with col2:
            monthly_debt = st.number_input(
                "Existing monthly debt (USD)", min_value=0.0, value=1200.0, step=100.0
            )
            employment_years = st.number_input(
                "Employment years", min_value=0.0, value=15.0, step=1.0
            )
            loans_repaid = st.number_input(
                "Loans repaid", min_value=0, value=4, step=1
            )
        with col3:
            interest_rate_pct = st.number_input(
                "Interest rate (%)", min_value=0.0, value=5.0, step=0.25,
                help="Enter 5 for 5%. The model receives 0.05."
            )
            job_options = [
                "Accountant", "Banker", "Business Owner", "Doctor", "Engineer",
                "Freelancer", "Lawyer", "Nurse", "Retired", "Sales Manager",
                "Software Engineer", "Teacher",
            ]
            job = st.selectbox("Occupation / job", job_options)
            area = st.selectbox("Area", ["Urban", "Suburban", "Rural"])

        st.divider()
        st.caption("The form uses the nine predictors in the saved model. Demographic fields excluded during training are not requested.")
        submitted = st.form_submit_button("Validate and score application", type="primary", use_container_width=True)

    if submitted:
        applicant = {
            "Employment Years": float(employment_years),
            "Annual Income (USD)": float(annual_income),
            "Interest Rate": float(interest_rate_pct) / 100.0,
            "Down Payment (USD)": float(down_payment),
            "Credit Score": int(credit_score),
            "Existing Monthly Debt (USD)": float(monthly_debt),
            "Loans Repaid": int(loans_repaid),
            "Job": job,
            "Area": area,
        }

        application_id = f"MTG-{uuid.uuid4().hex[:12].upper()}"
        with st.spinner("Validating application and generating model explanation..."):
            result = scorer.score(applicant)

        st.session_state["last_applicant"] = applicant
        st.session_state["last_result"] = result
        st.session_state["last_application_id"] = application_id
        st.session_state["last_prediction_id"] = None
        st.session_state["last_db_message"] = None

        # Persist every scoring attempt, including failed validation, when SQL is configured.
        if os.getenv("POSTGRES_DSN"):
            try:
                repository = MortgageRepository()
                prediction_id = repository.save_scoring_event(application_id, applicant, result)
                st.session_state["last_prediction_id"] = prediction_id
                st.session_state["last_db_message"] = f"Saved to PostgreSQL (Prediction ID: {prediction_id})."
            except Exception as exc:
                st.session_state["last_db_message"] = f"PostgreSQL save failed: {exc}"
        else:
            st.session_state["last_db_message"] = "Not saved: configure POSTGRES_DSN to enable persistence."

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Scoring result")
        st.caption(f"Application ID: {st.session_state.get('last_application_id', '—')}")
        db_message = st.session_state.get("last_db_message")
        if db_message:
            if db_message.startswith("Saved"):
                st.success(db_message)
            else:
                st.warning(db_message)
        if result["validation_status"] == "FAILED":
            st.error("Validation failed — send this record to manual underwriting. No prediction was generated.")
            for error in result.get("errors", []):
                st.write(f"- {error}")
            prediction_id = st.session_state.get("last_prediction_id")
            if prediction_id is not None:
                st.markdown("#### Manual underwriting disposition")
                with st.form("invalid_application_disposition_form"):
                    manual_decision = st.selectbox(
                        "Manual review outcome",
                        ["FURTHER_REVIEW", "DECLINED"],
                        format_func=lambda x: {
                            "FURTHER_REVIEW": "Keep in manual underwriting",
                            "DECLINED": "Decline after manual review",
                        }[x],
                    )
                    manual_underwriter = st.text_input("Underwriter ID")
                    manual_note = st.text_area("Manual review notes")
                    manual_submitted = st.form_submit_button(
                        "Save manual underwriting outcome", type="primary", use_container_width=True
                    )
                if manual_submitted:
                    try:
                        decision_id = MortgageRepository().save_underwriting_decision(
                            application_id=st.session_state["last_application_id"],
                            prediction_id=int(prediction_id),
                            decision=manual_decision,
                            predicted_amount=None,
                            approved_amount=None,
                            override_reason=manual_note or None,
                            underwriter_id=manual_underwriter,
                        )
                        st.success(f"Manual underwriting outcome saved. Decision ID: {decision_id}")
                    except Exception as exc:
                        st.error(f"Could not save manual underwriting outcome: {exc}")
        else:
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Predicted maximum loan", money(result.get("predicted_max_loan_usd")))
            metric2.metric("Model baseline", money(result.get("baseline_prediction_usd")))
            metric3.metric("Validation", result.get("validation_status", "—"))
            st.info("Status: REVIEW REQUIRED — this is a model estimate, not an approval or safe lending limit.")

            st.markdown("#### SHAP reason codes")
            reasons = result.get("reason_codes", [])
            if reasons:
                reason_df = pd.DataFrame(reasons)
                reason_df["contribution_usd"] = reason_df["contribution_usd"].map(money)
                reason_df = reason_df.rename(columns={
                    "code": "Reason code",
                    "feature": "Model feature",
                    "contribution_usd": "Contribution (USD)",
                    "direction_vs_baseline": "Direction",
                    "explanation": "Explanation",
                })
                st.dataframe(reason_df, use_container_width=True, hide_index=True)
                st.caption("Positive and negative SHAP contributions describe movement relative to the model baseline. They are not causal explanations or policy decisions.")
            else:
                st.write("No reason codes were returned.")

            st.markdown("#### Applicant input summary")
            summary = pd.DataFrame(
                [{"Field": key, "Value": value} for key, value in st.session_state["last_applicant"].items()]
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            export_payload = {
                "application": st.session_state["last_applicant"],
                "scoring_result": result,
            }
            st.download_button(
                "Download scoring result (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name="mortgage_scoring_result.json",
                mime="application/json",
            )

            st.markdown("#### Underwriter decision")
            prediction_id = st.session_state.get("last_prediction_id")
            if prediction_id is None:
                st.info("Save this scoring event to PostgreSQL before recording an underwriting decision.")
            else:
                with st.form("underwriting_decision_form"):
                    decision = st.selectbox(
                        "Decision",
                        ["ACCEPTED", "OVERRIDDEN", "DECLINED", "FURTHER_REVIEW"],
                        format_func=lambda x: {
                            "ACCEPTED": "Accept model recommendation",
                            "OVERRIDDEN": "Override recommendation",
                            "DECLINED": "Decline application",
                            "FURTHER_REVIEW": "Request further review",
                        }[x],
                    )
                    predicted_amount = float(result.get("predicted_max_loan_usd") or 0)
                    approved_amount = st.number_input(
                        "Approved amount (USD)", min_value=0.0,
                        value=predicted_amount, step=5000.0,
                        disabled=decision in {"DECLINED", "FURTHER_REVIEW"},
                    )
                    underwriter_id = st.text_input("Underwriter ID")
                    override_reason = st.text_area(
                        "Override reason (required when overriding)",
                        disabled=decision != "OVERRIDDEN",
                    )
                    decision_submitted = st.form_submit_button(
                        "Save underwriting decision", type="primary", use_container_width=True
                    )
                if decision_submitted:
                    try:
                        repository = MortgageRepository()
                        decision_id = repository.save_underwriting_decision(
                            application_id=st.session_state["last_application_id"],
                            prediction_id=int(prediction_id),
                            decision=decision,
                            predicted_amount=predicted_amount,
                            approved_amount=(None if decision in {"DECLINED", "FURTHER_REVIEW"} else float(approved_amount)),
                            override_reason=override_reason or None,
                            underwriter_id=underwriter_id,
                        )
                        st.success(f"Underwriting decision saved. Decision ID: {decision_id}")
                    except Exception as exc:
                        st.error(f"Could not save underwriting decision: {exc}")

with tab_database:
    st.subheader("PostgreSQL application and underwriting records")
    if not os.getenv("POSTGRES_DSN"):
        st.info("Configure POSTGRES_DSN in your .env file to view saved database activity.")
    else:
        try:
            repository = MortgageRepository()
            st.success("Connected to PostgreSQL")
            operations = repository.recent_operations(limit=250)
            if operations:
                operations_df = pd.DataFrame(operations)
                st.dataframe(operations_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download recent operations (CSV)",
                    data=operations_df.to_csv(index=False),
                    file_name="mortgage_operations.csv",
                    mime="text/csv",
                )
            else:
                st.info("No scoring events have been saved yet.")
        except Exception as exc:
            st.error(f"Could not load PostgreSQL records: {exc}")

with tab_model:
    st.subheader("Model performance and configuration")
    metrics = metadata.get("final_test_metrics", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", money(metrics.get("MAE")))
    m2.metric("Test RMSE", money(metrics.get("RMSE")))
    m3.metric("Test R²", f"{metrics['R2']:.4f}" if metrics.get("R2") is not None else "—")

    st.markdown("#### Model inputs")
    feature_names = metadata.get("features", [])
    st.write(", ".join(feature_names) if feature_names else "Feature metadata unavailable.")

    st.markdown("#### Training / evaluation setup")
    st.write(f"**Model selection:** {metadata.get('selection_metric', 'Not recorded')}")
    split = metadata.get("split", {})
    if split:
        st.write(f"**Data split:** {split.get('train')} training / {split.get('validation')} validation / {split.get('test')} test")
    st.write(f"**Currency label:** {metadata.get('currency', 'Not specified')}")

    comparison_path = ARTIFACT_DIR / "validation_model_comparison.csv"
    if comparison_path.exists():
        st.markdown("#### Validation comparison of five models")
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    shap_plot = ARTIFACT_DIR / "shap_summary.png"
    if shap_plot.exists():
        st.markdown("#### Global SHAP summary")
        st.image(str(shap_plot), use_container_width=True)

    st.markdown("#### Model governance warning")
    st.warning(metadata.get("warning", "Prototype only. Validate before any real lending use."))

st.divider()
st.caption("Mortgage analytics prototype · Do not use for automated lending decisions. Store and transmit applicant data only in an appropriately secured environment.")
