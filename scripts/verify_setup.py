import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_aws() -> bool:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    bucket = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not bucket:
        print("[FAIL] AWS: S3_BUCKET_NAME is not set in .env")
        return False

    try:
        s3 = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket)
        print(f"[OK]   AWS: connected, bucket '{bucket}' is reachable.")
        return True
    except NoCredentialsError:
        print("[FAIL] AWS: no credentials found. Check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")
        return False
    except ClientError as e:
        print(f"[FAIL] AWS: could not reach bucket '{bucket}'. {e}")
        return False
    except Exception as e:
        print(f"[FAIL] AWS: unexpected error. {e}")
        return False


def check_snowflake() -> bool:
    import snowflake.connector

    required = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"[FAIL] Snowflake: missing .env values: {', '.join(missing)}")
        return False

    try:
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
        cur.execute("SELECT CURRENT_VERSION(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
        version, warehouse, database, schema = cur.fetchone()
        print(
            f"[OK]   Snowflake: connected. version={version}, "
            f"warehouse={warehouse}, database={database}, schema={schema}"
        )
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Snowflake: could not connect. {e}")
        return False


if __name__ == "__main__":
    print("Checking AWS S3 connection...")
    aws_ok = check_aws()

    print("\nChecking Snowflake connection...")
    sf_ok = check_snowflake()

    print("\n--- Summary ---")
    if aws_ok and sf_ok:
        print("All systems go. You're ready for SQL foundation")
        sys.exit(0)
    else:
        print("Fix the failed connection(s) above before moving further.")
        sys.exit(1)
