
import numpy as np
import pandas as pd

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
GRADE_MAP = {g: i + 1 for i, g in enumerate("ABCDEFG")}


def _encode_sub_grade(sub_grade: pd.Series) -> pd.Series:
    letters = "ABCDEFG"
    order = {f"{L}{n}": (letters.index(L) * 5) + n for L in letters for n in range(1, 6)}
    return sub_grade.map(order)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Ordinal encodings ---
    df["grade_ordinal"] = df["grade"].map(GRADE_MAP)
    df["sub_grade_ordinal"] = _encode_sub_grade(df["sub_grade"])
    df["emp_length_years"] = df["emp_length"].map(EMP_LENGTH_MAP)

    # --- Derived ratios ---
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["installment_to_income"] = (df["installment"] * 12) / df["annual_inc"].replace(0, np.nan)

    df["earliest_cr_line_dt"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["issue_d_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["credit_history_months"] = (df["issue_d_dt"] - df["earliest_cr_line_dt"]).dt.days / 30.44

    # --- Log-transform (Annual_inc is Log-Normal, not Normal) ---
    df["log_annual_inc"] = np.log(df["annual_inc"].clip(lower=1))

    # --- Binning: credit utilization into interpretable buckets ---
    df["revol_util_bucket"] = pd.cut(
        df["revol_util"], bins=[-0.01, 30, 70, 100, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    # --- IQR-based outlier flag (Five-number summary applied) ---
    q1, q3 = df["loan_amnt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["loan_amnt_is_outlier"] = ((df["loan_amnt"] < lower) | (df["loan_amnt"] > upper)).astype(int)

    # --- Missing value imputation ---
    df["loan_to_income"] = df["loan_to_income"].fillna(df["loan_to_income"].median())
    df["installment_to_income"] = df["installment_to_income"].fillna(df["installment_to_income"].median())
    df["mort_acc"] = df["mort_acc"].fillna(df["mort_acc"].median())
    df["emp_length_years"] = df["emp_length_years"].fillna(df["emp_length_years"].median())
    df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())
    df["revol_util_bucket"] = df["revol_util_bucket"].astype(object).fillna("unknown")
    df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)
    df["credit_history_months"] = df["credit_history_months"].fillna(df["credit_history_months"].median())

    # --- Nominal encodings ---
    # Low-cardinality nominal columns -> one-hot, but KEEP the original text
    # columns too. A model wants the one-hot columns; a dashboard wants a
    # plain category to group and slice by.
    dummies = pd.get_dummies(
        df[["home_ownership", "verification_status", "purpose", "application_type"]],
        prefix=["home", "verif", "purpose", "apptype"], dtype=int,
    )
    df = pd.concat([df, dummies], axis=1)

    # High-cardinality nominal column (50+ states) -> frequency encoding
    state_freq = df["addr_state"].value_counts(normalize=True)
    df["addr_state_freq"] = df["addr_state"].map(state_freq)

    return df
