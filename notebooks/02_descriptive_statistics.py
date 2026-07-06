
import os

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import snowflake.connector
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
pd.set_option("display.width", 160)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


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

    df = df.drop(columns=["id"])


    df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)

    return df


def central_tendency_and_dispersion(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CENTRAL TENDENCY & DISPERSION (topics 46-47)")
    print("=" * 70)

    for col in ["loan_amnt", "annual_inc", "int_rate", "dti"]:
        s = df[col].dropna()
        mean, median, mode = s.mean(), s.median(), s.mode().iloc[0]
        value_range = s.max() - s.min()
        var, std = s.var(ddof=1), s.std(ddof=1)
        print(
            f"\n{col}:\n"
            f"  mean={mean:,.2f}  median={median:,.2f}  mode={mode:,.2f}\n"
            f"  range={value_range:,.2f}  variance={var:,.2f}  std_dev={std:,.2f}"
        )


def n_minus_1_demo(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("WHY SAMPLE VARIANCE DIVIDES BY N-1 (topic 48)")
    print("=" * 70)

    s = df["int_rate"].dropna()
    n = len(s)
    mean = s.mean()
    sq_devs = (s - mean) ** 2

    var_n = sq_devs.sum() / n                 
    var_n_minus_1 = sq_devs.sum() / (n - 1)  

    print(
        f"int_rate, n={n:,}\n"
        f"  dividing by n     -> variance = {var_n:.6f}\n"
        f"  dividing by n-1   -> variance = {var_n_minus_1:.6f}\n"
        f"  pandas .var(ddof=1) confirms  = {s.var(ddof=1):.6f}\n\n"
        "  Dividing by n underestimates the true population variance, because\n"
        "  the sample mean is itself computed from this same data and sits\n"
        "  slightly closer to every point than the true population mean would.\n"
        "  Dividing by n-1 corrects that bias. The gap shrinks as n grows --\n"
        "  with 200,000 rows it's tiny here, but with a small sample it matters."
    )


def percentiles_and_five_number_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PERCENTILES, QUARTILES & 5-NUMBER SUMMARY")
    print("=" * 70)

    for col in ["loan_amnt", "annual_inc"]:
        s = df[col].dropna()
        five_num = {
            "min": s.min(),
            "Q1 (25th pct)": s.quantile(0.25),
            "median (50th pct)": s.quantile(0.50),
            "Q3 (75th pct)": s.quantile(0.75),
            "max": s.max(),
        }
        print(f"\n{col}:")
        for label, val in five_num.items():
            print(f"  {label:<20} {val:,.2f}")
        print(f"  {'90th pct':<20} {s.quantile(0.90):,.2f}")
        print(f"  {'99th pct':<20} {s.quantile(0.99):,.2f}")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="grade", y="loan_amnt", order=sorted(df["grade"].unique()))
    plt.title("Loan Amount Distribution by Grade (5-Number Summary)")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/boxplot_loan_amnt_by_grade.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nSaved: {path}")


def histograms_and_skewness(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("HISTOGRAMS & SKEWNESS")
    print("=" * 70)

    cols = ["annual_inc", "loan_amnt", "int_rate"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        skewness = stats.skew(s)
        sns.histplot(s, bins=50, ax=ax, kde=True)
        ax.set_title(f"{col}\nskew={skewness:.2f}")
        label = "right-skewed" if skewness > 0.5 else "left-skewed" if skewness < -0.5 else "roughly symmetric"
        print(f"{col}: skewness = {skewness:.2f} ({label})")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/histograms_skewness.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nSaved: {path}")
    print(
        "\nannual_inc's positive skew means a long right tail of high earners --"
        " this is exactly why we apply a log-transform to it before"
        " feeding it into anything sensitive to scale."
    )


def covariance_and_correlation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("COVARIANCE & CORRELATION")
    print("=" * 70)

    numeric_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
        "revol_bal", "revol_util", "open_acc", "total_acc", "default_flag",
    ]
    corr = df[numeric_cols].corr()
    print(corr.round(2).to_string())

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Numeric Loan Features")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nSaved: {path}")

    top_corr = corr["default_flag"].drop("default_flag").abs().sort_values(ascending=False)
    print(f"\nStrongest linear relationships with default_flag:\n{top_corr.round(3).to_string()}")


def main():
    print("Loading and cleaning data from Snowflake...")
    df = load_data()
    print(f"Ready: {len(df):,} rows")

    central_tendency_and_dispersion(df)
    n_minus_1_demo(df)
    percentiles_and_five_number_summary(df)
    histograms_and_skewness(df)
    covariance_and_correlation(df)

    print("\nDone. Charts saved under reports/figures/.")


if __name__ == "__main__":
    main()
