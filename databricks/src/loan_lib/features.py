"""Features for the loan default model: the no-age variant from the Azure notebook.

Date of birth never reaches the model, so it can't use age (Equal Credit Opportunity Act). The
registered model's input schema is INPUTS, which leaves it out entirely.
"""

TARGET = "target"
INPUTS = [
    "number_dependants", "credit_utilization", "debt_to_income_ratio", "monthly_income",
    "number_open_credit_lines", "number_open_loans", "number_90_days_past_due", "number_charged_off",
    "score1", "score2",
]


def engineer_features(df):
    """Same steps as the Azure notebook's no-age variant."""
    df = df[INPUTS].copy()
    df["number_dependants"] = df["number_dependants"].replace(-1, 0)  # -1 is an entry error; read it as none
    df["total_open_accounts"] = df["number_open_credit_lines"] + df["number_open_loans"]
    return df
