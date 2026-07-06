import os
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path("data/processed/loans_engineered.csv").resolve()
TABLE_NAME = "ANALYTICS.LOANS_ENGINEERED"

DTYPE_MAP = {
    "int64": "NUMBER",
    "float64": "FLOAT",
    "object": "STRING",
    "bool": "BOOLEAN",
}


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


def build_create_table_sql(df: pd.DataFrame) -> str:
    columns_sql = [f'    "{col}" {DTYPE_MAP.get(str(dtype), "STRING")}' for col, dtype in df.dtypes.items()]
    columns_block = ",\n".join(columns_sql)
    return f"CREATE OR REPLACE TABLE {TABLE_NAME} (\n{columns_block}\n);"


def main():
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Sanitize column names for use as Snowflake identifiers -- one-hot
    # columns like 'verif_Not Verified' contain spaces, which aren't safe
    # to leave unescaped. This only relabels the in-memory copy used to
    # build the CREATE TABLE statement; COPY INTO matches by column
    # POSITION, not name, so the original CSV on disk doesn't need to change.
    df.columns = [c.upper().replace(" ", "_").replace("-", "_") for c in df.columns]

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("CREATE SCHEMA IF NOT EXISTS ANALYTICS")

        create_sql = build_create_table_sql(df)
        print("\nCreating table ANALYTICS.LOANS_ENGINEERED ...")
        cur.execute(create_sql)

        cur.execute("CREATE STAGE IF NOT EXISTS ANALYTICS.ENGINEERED_STAGE")

        file_uri = DATA_PATH.as_posix()
        print(f"Uploading {file_uri} ...")
        cur.execute(
            f"PUT file://{file_uri} @ANALYTICS.ENGINEERED_STAGE "
            "OVERWRITE = TRUE AUTO_COMPRESS = TRUE"
        )

        print("Copying into table...")
        cur.execute(
            f"""
            COPY INTO {TABLE_NAME}
            FROM @ANALYTICS.ENGINEERED_STAGE/loans_engineered.csv
            FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1
                           FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                           NULL_IF = ('', 'NA', 'nan'))
            ON_ERROR = 'CONTINUE'
            """
        )
        for row in cur.fetchall():
            print(row)

        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cur.fetchone()[0]
        print(f"\nDone. {TABLE_NAME} now has {count:,} rows.")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
