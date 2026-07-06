
import os

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 160)


def load_data() -> pd.DataFrame:
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )
    df = pd.read_sql("SELECT * FROM RAW.RAW_LOANS", conn)
    conn.close()

    df.columns = [c.lower() for c in df.columns]
    return df


def build_classification_table() -> pd.DataFrame:
    """
    Classify every column by data type and scale of measurement.
      - Qualitative (categorical): nominal = no order, ordinal = order exists
      - Quantitative (numerical): discrete = countable, continuous = measurable
      - Scale of measurement: nominal / ordinal / interval / ratio
    """
    return pd.DataFrame([
        ("id",                   "Qualitative", "Identifier",              "Nominal",  "Unique key, not used in analysis"),
        ("loan_amnt",            "Quantitative", "Continuous",             "Ratio",    "Dollar amount, true zero"),
        ("funded_amnt",          "Quantitative", "Continuous",             "Ratio",    "Dollar amount"),
        ("term",                 "Quantitative", "Discrete",               "Ratio",    "Text ('36 months')"),
        ("int_rate",             "Quantitative", "Continuous",             "Ratio",    "Text with '%'"),
        ("installment",          "Quantitative", "Continuous",             "Ratio",    "Dollar amount"),
        ("grade",                "Qualitative", "Ordinal",                 "Ordinal",  "A (best) through G (worst) -- order matters"),
        ("sub_grade",            "Qualitative", "Ordinal",                 "Ordinal",  "A1..G5, finer-grained grade"),
        ("emp_length",           "Qualitative", "Ordinal",                 "Ordinal",  "'<1 year'..'10+ years', top-coded at 10+"),
        ("home_ownership",       "Qualitative", "Nominal",                 "Nominal",  "RENT / OWN / MORTGAGE / OTHER, no order"),
        ("annual_inc",           "Quantitative", "Continuous",             "Ratio",    "Dollar amount, right-skewed"),
        ("verification_status",  "Qualitative", "Nominal (arguably ordinal)", "Nominal", "Could be read as verification 'strength' -- treated as nominal here"),
        ("issue_d",              "Quantitative", "Date",                   "Interval", "No true zero on a calendar, but equal month spacing"),
        ("loan_status",          "Qualitative", "Nominal",                 "Nominal",  "Source of default_flag; excluded from modeling to avoid leakage"),
        ("purpose",              "Qualitative", "Nominal",                 "Nominal",  "debt_consolidation, credit_card, etc."),
        ("dti",                  "Quantitative", "Continuous",             "Ratio",    "Debt-to-income ratio"),
        ("delinq_2yrs",          "Quantitative", "Discrete",               "Ratio",    "Count, mostly 0 -- Poisson candidate"),
        ("earliest_cr_line",     "Quantitative", "Date",                   "Interval", "Used to derive credit history length"),
        ("inq_last_6mths",       "Quantitative", "Discrete",               "Ratio",    "Count of recent credit inquiries"),
        ("open_acc",             "Quantitative", "Discrete",               "Ratio",    "Count of open credit lines"),
        ("pub_rec",              "Quantitative", "Discrete",               "Ratio",    "Count of derogatory public records -- rare event"),
        ("revol_bal",            "Quantitative", "Continuous",             "Ratio",    "Dollar amount"),
        ("revol_util",           "Quantitative", "Continuous",             "Ratio",    "Percentage, text with '%'"),
        ("total_acc",            "Quantitative", "Discrete",               "Ratio",    "Count of total credit lines ever opened"),
        ("application_type",     "Qualitative", "Nominal",                 "Nominal",  "Individual vs. Joint App"),
        ("mort_acc",             "Quantitative", "Discrete",               "Ratio",    "Count of mortgage accounts"),
        ("pub_rec_bankruptcies", "Quantitative", "Discrete",               "Ratio",    "Count, mostly 0 -- rare event"),
        ("addr_state",           "Qualitative", "Nominal",                 "Nominal",  "US state code, 50+ categories"),
        ("default_flag",         "Quantitative", "Discrete (binary)",      "Ratio",    "0/1 -- a Bernoulli random variable"),
    ], columns=["column", "data_type", "sub_type", "scale", "notes"])


def build_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness + cardinality per column"""
    return pd.DataFrame({
        "pandas_dtype": df.dtypes.astype(str),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    }).sort_values("pct_missing", ascending=False)


def main():
    print("Loading RAW_LOANS from Snowflake...")
    df = load_data()
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns\n")
    print(df.head())

    print("\n" + "=" * 70)
    print("POPULATION VS. SAMPLE")
    print("=" * 70)
    print(
        "Lending Club's full issued-loan history is the population. The Kaggle\n"
        "export is already a sample of it (loans issued 2007-2018).Sampled\n"
        "again: every resolved loan had roughly a 15% chance of being kept\n"
        "(Bernoulli / simple random sampling). Every number computed from here on\n"
        "is a sample statistic -- an estimate of a population parameter,\n"
        "not the parameter itself."
    )

    print("\n" + "=" * 70)
    print("COLUMN CLASSIFICATION")
    print("=" * 70)
    classification = build_classification_table()
    print(classification.to_string(index=False))

    print(
        "\nWorth calling out:\n"
        "- grade/sub_grade/emp_length are ordinal: rankable, but the 'distance'\n"
        "  between categories isn't numerically meaningful -- mean of grade is\n"
        "  invalid, median/mode are fine.\n"
        "- default_flag is a Bernoulli random variable -- not building\n"
        "  up to it later, it already is one.\n"
        "- delinq_2yrs, pub_rec, pub_rec_bankruptcies, inq_last_6mths are mostly\n"
        "  zero with a long right tail -- Poisson candidates."
    )

    print("\n" + "=" * 70)
    print("MISSINGNESS + CARDINALITY AUDIT")
    print("=" * 70)
    audit = build_audit_table(df)
    print(audit.to_string())


if __name__ == "__main__":
    main()