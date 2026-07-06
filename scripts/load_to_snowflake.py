
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

LOCAL_FILE = Path("data/raw/loans_clean.csv").resolve()

CREATE_TABLE_SQL = """
CREATE OR REPLACE TABLE RAW.RAW_LOANS (
    id                      STRING,
    loan_amnt               FLOAT,
    funded_amnt             FLOAT,
    term                    STRING,
    int_rate                STRING,
    installment             FLOAT,
    grade                   STRING,
    sub_grade               STRING,
    emp_length              STRING,
    home_ownership          STRING,
    annual_inc              FLOAT,
    verification_status     STRING,
    issue_d                 STRING,
    loan_status             STRING,
    purpose                 STRING,
    dti                     FLOAT,
    delinq_2yrs             FLOAT,
    earliest_cr_line        STRING,
    inq_last_6mths          FLOAT,
    open_acc                FLOAT,
    pub_rec                 FLOAT,
    revol_bal               FLOAT,
    revol_util              STRING,
    total_acc               FLOAT,
    application_type        STRING,
    mort_acc                FLOAT,
    pub_rec_bankruptcies    FLOAT,
    addr_state              STRING,
    default_flag            INTEGER
);
"""

COPY_INTO_SQL = """
COPY INTO RAW.RAW_LOANS
FROM @RAW.LOAN_STAGE/loans_clean.csv
FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"'
               NULL_IF = ('', 'NA', 'n/a'))
ON_ERROR = 'CONTINUE';
"""


def main():
    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            f"{LOCAL_FILE} not found. Run scripts/prepare_raw_data.py first."
        )

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )
    cur = conn.cursor()

    try:
        print("Creating RAW.RAW_LOANS table...")
        cur.execute(CREATE_TABLE_SQL)

        file_uri = LOCAL_FILE.as_posix()
        print(f"Uploading {file_uri} to stage @RAW.LOAN_STAGE ...")
        cur.execute(
            f"PUT file://{file_uri} @RAW.LOAN_STAGE OVERWRITE = TRUE AUTO_COMPRESS = TRUE"
        )

        print("Copying staged file into RAW_LOANS ...")
        cur.execute(COPY_INTO_SQL)
        for row in cur.fetchall():
            print(row)

        cur.execute("SELECT COUNT(*) FROM RAW.RAW_LOANS")
        count = cur.fetchone()[0]
        print(f"\nLoad complete. RAW.RAW_LOANS now has {count:,} rows.")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
