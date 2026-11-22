-- Mortgage underwriting prototype schema for PostgreSQL.
-- Safe to rerun: CREATE TABLE/INDEX/VIEW IF NOT EXISTS preserves existing data.

CREATE TABLE IF NOT EXISTS mortgage_applications (
    application_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    employment_years NUMERIC(10,2),
    annual_income_usd NUMERIC(18,2),
    interest_rate NUMERIC(9,6),
    down_payment_usd NUMERIC(18,2),
    credit_score INTEGER,
    existing_monthly_debt_usd NUMERIC(18,2),
    loans_repaid INTEGER,
    job VARCHAR(150),
    area VARCHAR(80),
    applicant_json JSONB,
    CONSTRAINT ck_mortgage_applications_income CHECK (annual_income_usd IS NULL OR annual_income_usd >= 0),
    CONSTRAINT ck_mortgage_applications_credit CHECK (credit_score IS NULL OR credit_score BETWEEN 0 AND 1000)
);

CREATE TABLE IF NOT EXISTS mortgage_predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    application_id VARCHAR(50) NOT NULL REFERENCES mortgage_applications(application_id),
    predicted_max_loan_usd NUMERIC(18,2),
    baseline_prediction_usd NUMERIC(18,2),
    validation_status VARCHAR(30) NOT NULL,
    scoring_status VARCHAR(40) NOT NULL,
    validation_errors_json JSONB,
    model_version VARCHAR(100),
    prediction_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_mortgage_predictions_application_id
    ON mortgage_predictions(application_id);
CREATE INDEX IF NOT EXISTS ix_mortgage_predictions_prediction_date
    ON mortgage_predictions(prediction_date DESC);

CREATE TABLE IF NOT EXISTS prediction_reasons (
    reason_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES mortgage_predictions(prediction_id) ON DELETE CASCADE,
    reason_code VARCHAR(20) NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    contribution_usd NUMERIC(18,2) NOT NULL DEFAULT 0,
    direction_vs_baseline VARCHAR(30) NOT NULL,
    explanation VARCHAR(1000) NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_prediction_reasons_prediction_id
    ON prediction_reasons(prediction_id);

CREATE TABLE IF NOT EXISTS underwriting_decisions (
    decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    application_id VARCHAR(50) NOT NULL REFERENCES mortgage_applications(application_id),
    prediction_id BIGINT NOT NULL REFERENCES mortgage_predictions(prediction_id),
    decision VARCHAR(40) NOT NULL CHECK (decision IN ('ACCEPTED', 'OVERRIDDEN', 'DECLINED', 'FURTHER_REVIEW')),
    predicted_amount_usd NUMERIC(18,2),
    approved_amount_usd NUMERIC(18,2),
    override_reason VARCHAR(2000),
    underwriter_id VARCHAR(150) NOT NULL,
    decision_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_override_reason_required CHECK (decision <> 'OVERRIDDEN' OR NULLIF(BTRIM(override_reason), '') IS NOT NULL),
    CONSTRAINT ck_approved_amount_required CHECK (decision NOT IN ('ACCEPTED', 'OVERRIDDEN') OR approved_amount_usd IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ix_underwriting_decisions_application_id
    ON underwriting_decisions(application_id);
CREATE INDEX IF NOT EXISTS ix_underwriting_decisions_prediction_id
    ON underwriting_decisions(prediction_id);

CREATE OR REPLACE VIEW vw_mortgage_operations AS
SELECT
    a.application_id,
    a.created_at AS application_date,
    a.annual_income_usd,
    a.credit_score,
    a.down_payment_usd,
    a.existing_monthly_debt_usd,
    p.prediction_id,
    p.predicted_max_loan_usd,
    p.baseline_prediction_usd,
    p.validation_status,
    p.scoring_status,
    p.model_version,
    p.prediction_date,
    d.decision,
    d.approved_amount_usd,
    d.override_reason,
    d.underwriter_id,
    d.decision_date
FROM mortgage_applications a
JOIN mortgage_predictions p ON p.application_id = a.application_id
LEFT JOIN LATERAL (
    SELECT ud.decision, ud.approved_amount_usd, ud.override_reason,
           ud.underwriter_id, ud.decision_date
    FROM underwriting_decisions ud
    WHERE ud.prediction_id = p.prediction_id
    ORDER BY ud.decision_date DESC, ud.decision_id DESC
    LIMIT 1
) d ON TRUE;
