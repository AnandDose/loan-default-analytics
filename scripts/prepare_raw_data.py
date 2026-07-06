
import pandas as pd

RAW_PATH = "data/raw/loans_original.csv"
OUTPUT_PATH = "data/raw/loans_clean.csv"
SAMPLE_SIZE = 200_000       # cap the row count for a fast, manageable project
RANDOM_STATE = 42           # fixed seed = reproducible sample

KEEP_COLUMNS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "application_type",
    "mort_acc", "pub_rec_bankruptcies", "addr_state",
]

RESOLVED_STATUSES = ["Fully Paid", "Charged Off", "Default"]


def main():
    print(f"Reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, usecols=lambda c: c in KEEP_COLUMNS, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    before = len(df)
    df = df[df["loan_status"].isin(RESOLVED_STATUSES)].copy()
    print(f"Kept {len(df):,} / {before:,} rows with a resolved loan_status")

    df["default_flag"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)
    df = df[KEEP_COLUMNS + ["default_flag"]]


    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"Randomly sampled down to {SAMPLE_SIZE:,} rows (random_state={RANDOM_STATE})")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned file to {OUTPUT_PATH}")
    print(f"Default rate in this sample: {df['default_flag'].mean():.2%}")


if __name__ == "__main__":
    main()
