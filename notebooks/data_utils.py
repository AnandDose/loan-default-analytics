
import os

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

DTI_MAX_VALID = 100
INCOME_WINSORIZE_PCT = 0.995


def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )


def load_raw_loans() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM RAW.RAW_LOANS", conn)
    conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def clean_loans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["id"]).copy()

    df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)

    before = len(df)
    df = df[df["dti"].isna() | (df["dti"] < DTI_MAX_VALID)]
    dropped = before - len(df)
    if dropped:
        print(f"[clean_loans] Dropped {dropped} rows with dti >= {DTI_MAX_VALID} (invalid/sentinel values)")

    cap = df["annual_inc"].quantile(INCOME_WINSORIZE_PCT)
    n_capped = int((df["annual_inc"] > cap).sum())
    df["annual_inc"] = df["annual_inc"].clip(upper=cap)
    if n_capped:
        print(f"[clean_loans] Capped {n_capped} annual_inc values above the {INCOME_WINSORIZE_PCT:.1%} percentile (${cap:,.0f})")

    return df


def load_clean_loans() -> pd.DataFrame:
    """One call: pull from Snowflake and apply all cleaning decisions."""
    return clean_loans(load_raw_loans())
