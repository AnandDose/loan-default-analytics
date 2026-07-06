import os

import pandas as pd

from data_utils import load_clean_loans
from feature_engineering import engineer_features

pd.set_option("display.width", 160)
OUTPUT_PATH = "data/processed/loans_engineered.csv"


def main():
    print("Loading cleaned data...")
    df = load_clean_loans()
    print(f"Before feature engineering: {df.shape[0]:,} rows, {df.shape[1]} columns")

    df_fe = engineer_features(df)
    print(f"After feature engineering:  {df_fe.shape[0]:,} rows, {df_fe.shape[1]} columns")

    new_cols = [c for c in df_fe.columns if c not in df.columns]
    print(f"\nNew/derived columns ({len(new_cols)}):")
    for c in sorted(new_cols):
        print(f"  {c}")

    print("\nMissing values in the NEW/derived columns (should be none):")
    missing_new = df_fe[new_cols].isna().sum()
    print(missing_new[missing_new > 0].to_string() if missing_new.sum() else "  none")

    print("\nMissing values remaining in original raw columns (expected -- these")
    print("are either superseded by a cleaned/derived column, or were never")
    print("targeted for cleaning):")
    original_cols = [c for c in df.columns if c in df_fe.columns]
    missing_orig = df_fe[original_cols].isna().sum()
    print(missing_orig[missing_orig > 0].to_string() if missing_orig.sum() else "  none")

    print("\nSample of key engineered features:")
    preview_cols = [
        "grade_ordinal", "sub_grade_ordinal", "emp_length_years", "term_months",
        "loan_to_income", "installment_to_income", "credit_history_months",
        "log_annual_inc", "revol_util_bucket", "loan_amnt_is_outlier", "addr_state_freq",
    ]
    print(df_fe[preview_cols].head().to_string())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_fe.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved engineered dataset to {OUTPUT_PATH} ({df_fe.shape[0]:,} rows, {df_fe.shape[1]} columns)")


if __name__ == "__main__":
    main()
