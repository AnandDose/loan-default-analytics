import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data_utils import load_clean_loans

pd.set_option("display.width", 160)
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
RNG = np.random.default_rng(42)


def pdf_pmf_cdf(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PDF, PMF, CDF")
    print("=" * 70)
    print(
        "int_rate is continuous -> described by a PDF (probability DENSITY;\n"
        "individual points have zero probability, only ranges do).\n"
        "default_flag is discrete -> described by a PMF (probability MASS;\n"
        "each of its two outcomes has a direct probability). The CDF works\n"
        "for both: P(X <= x)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["int_rate"], bins=60, stat="density", ax=axes[0])
    axes[0].set_title("int_rate -- PDF (density)")

    sorted_rate = np.sort(df["int_rate"])
    cdf_vals = np.arange(1, len(sorted_rate) + 1) / len(sorted_rate)
    axes[1].plot(sorted_rate, cdf_vals)
    axes[1].set_title("int_rate -- empirical CDF")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/pdf_cdf_int_rate.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def bernoulli_and_binomial(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BERNOULLI & BINOMIAL DISTRIBUTIONS")
    print("=" * 70)

    p_hat = df["default_flag"].mean()
    print(
        f"Every single loan's outcome is a Bernoulli(p={p_hat:.4f}) trial:\n"
        f"  P(default) = {p_hat:.4f}, P(no default) = {1 - p_hat:.4f}\n"
        f"This IS the distribution default_flag follows -- not an analogy."
    )

    N = 50
    n_batches = 2000
    batch_defaults = [
        df["default_flag"].sample(N, replace=True, random_state=i).sum()
        for i in range(n_batches)
    ]
    theoretical_mean = N * p_hat
    print(
        f"\nGrouping loans into {n_batches:,} random batches of {N}:\n"
        f"  observed mean defaults/batch = {np.mean(batch_defaults):.2f}\n"
        f"  Binomial(n={N}, p={p_hat:.4f}) theoretical mean = {theoretical_mean:.2f}"
    )

    plt.figure(figsize=(7, 4))
    sns.histplot(batch_defaults, bins=range(0, N), discrete=True)
    plt.title(f"Defaults per batch of {N} loans -- Binomial(n={N}, p={p_hat:.3f})")
    plt.xlabel("defaults in batch")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/binomial_defaults_per_batch.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def poisson_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("POISSON DISTRIBUTION")
    print("=" * 70)

    for col in ["pub_rec_bankruptcies", "delinq_2yrs"]:
        s = df[col].dropna()
        mean, var = s.mean(), s.var(ddof=1)
        print(
            f"\n{col}: mean={mean:.4f}, variance={var:.4f}"
            f" (Poisson property: mean ≈ variance -- {'holds reasonably' if abs(mean - var) / mean < 0.5 else 'does NOT hold well'})"
        )

    plt.figure(figsize=(7, 4))
    sns.histplot(df["pub_rec_bankruptcies"], bins=range(0, 6), discrete=True)
    plt.title("pub_rec_bankruptcies -- rare-event count, Poisson-shaped")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/poisson_pub_rec_bankruptcies.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def normal_and_standard_normal(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NORMAL & STANDARD NORMAL DISTRIBUTIONS")
    print("=" * 70)

    s = df["int_rate"].dropna()
    z = (s - s.mean()) / s.std(ddof=1)
    outliers = (z.abs() > 3).sum()
    print(
        f"int_rate standardized to z-scores (Standard Normal, mean=0, std=1).\n"
        f"Rows with |z| > 3 (a common outlier rule under normality): {outliers} "
        f"({outliers / len(s):.2%})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(s, kde=True, stat="density", ax=axes[0])
    axes[0].set_title("int_rate distribution")
    stats.probplot(s, dist="norm", plot=axes[1])
    axes[1].set_title("int_rate -- Q-Q plot vs. Normal")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/normal_qq_int_rate.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")
    print(
        "If the Q-Q plot points hug the diagonal line, int_rate is close to\n"
        "Normal; systematic curvature (common in the tails here, since we\n"
        "already know int_rate is right-skewed) means it's Normal-ish in the\n"
        "middle but not a clean fit overall."
    )


def uniform_illustration() -> None:
    print("\n" + "=" * 70)
    print("UNIFORM DISTRIBUTION -- illustrative, not from loan data")
    print("=" * 70)
    print(
        "None of our columns are naturally Uniform, so this is a synthetic\n"
        "example for contrast only: every outcome in a Uniform distribution\n"
        "is equally likely, unlike everything else in this dataset."
    )

    synthetic_uniform = RNG.uniform(0, 1, size=10_000)
    plt.figure(figsize=(7, 4))
    sns.histplot(synthetic_uniform, bins=30)
    plt.title("Synthetic Uniform(0, 1) sample -- for contrast, not real loan data")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/uniform_synthetic_example.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def log_normal_income(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("LOG-NORMAL DISTRIBUTION")
    print("=" * 70)

    income = df["annual_inc"].dropna()
    income = income[income > 0]
    log_income = np.log(income)

    print(
        f"annual_inc raw skewness: {stats.skew(income):.2f} (heavily right-skewed)\n"
        f"log(annual_inc) skewness: {stats.skew(log_income):.2f} (much closer to symmetric)\n"
        "This is the definition of Log-Normal: the RAW variable is skewed,\n"
        "but its LOG is approximately Normal."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(income, bins=60, ax=axes[0])
    axes[0].set_title("annual_inc (raw) -- right-skewed")
    sns.histplot(log_income, bins=60, kde=True, ax=axes[1])
    axes[1].set_title("log(annual_inc) -- approximately Normal")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/lognormal_income.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def power_law_and_pareto(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("POWER LAW & PARETO DISTRIBUTIONS")
    print("=" * 70)

    sorted_loans = df["loan_amnt"].sort_values(ascending=False)
    total = sorted_loans.sum()
    top_20pct_n = int(len(sorted_loans) * 0.20)
    top_20pct_value = sorted_loans.iloc[:top_20pct_n].sum()
    share = top_20pct_value / total

    print(
        f"Top 20% of loans by amount account for {share:.1%} of total loan_amnt "
        f"funded.\n"
        f"(A classic 80/20 Pareto pattern would show ~80% here -- loan_amnt is "
        f"capped at $40,000 by Lending Club's own policy, which limits how "
        f"extreme this concentration can get compared to something truly "
        f"unbounded like wealth or city population.)"
    )

    plt.figure(figsize=(7, 4))
    cumulative_share = sorted_loans.cumsum() / total
    plt.plot(np.arange(1, len(sorted_loans) + 1) / len(sorted_loans), cumulative_share)
    plt.xlabel("Fraction of loans (largest first)")
    plt.ylabel("Cumulative fraction of total loan value")
    plt.title("Loan Value Concentration (Pareto-style curve)")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/pareto_loan_concentration.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def central_limit_theorem(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CENTRAL LIMIT THEOREM")
    print("=" * 70)

    population = df["loan_amnt"].dropna().values
    print(
        f"Population (loan_amnt) skewness: {stats.skew(population):.2f} -- "
        f"clearly not Normal.\n"
        "Now repeatedly draw samples and look at the distribution of their MEANS:"
    )

    sample_size = 100
    n_samples = 3000
    sample_means = [
        RNG.choice(population, size=sample_size, replace=True).mean()
        for _ in range(n_samples)
    ]
    print(
        f"  {n_samples:,} samples of size {sample_size}\n"
        f"  skewness of the sampling distribution of the mean: {stats.skew(sample_means):.2f} "
        f"(near 0 -- much more symmetric than the population)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(population, bins=60, ax=axes[0])
    axes[0].set_title("Population: loan_amnt (skewed)")
    sns.histplot(sample_means, bins=60, kde=True, ax=axes[1])
    axes[1].set_title(f"Sampling distribution of the mean (n={sample_size}) -- Normal-shaped")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/clt_sampling_distribution.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")
    print(
        "This is the CLT in action: regardless of the population's shape, the\n"
        "distribution of sample means approaches Normal as sample size grows.\n"
        "It's the reason t-tests and z-tests are allowed to assume\n"
        "normality of the mean even when the underlying data isn't Normal."
    )


def estimates(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ESTIMATES")
    print("=" * 70)

    p_hat = df["default_flag"].mean()
    n = len(df)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    print(
        f"p_hat = {p_hat:.4f} is a POINT ESTIMATE of the true population default\n"
        f"rate -- our best single guess, not a certainty. Its standard error is\n"
        f"{se:.5f}. We will turn this into a proper confidence interval instead\n"
        f"of reporting p_hat as if it were exact."
    )


def main():
    print("Loading cleaned data...")
    df = load_clean_loans()
    print(f"Ready: {len(df):,} rows")

    pdf_pmf_cdf(df)
    bernoulli_and_binomial(df)
    poisson_check(df)
    normal_and_standard_normal(df)
    uniform_illustration()
    log_normal_income(df)
    power_law_and_pareto(df)
    central_limit_theorem(df)
    estimates(df)

    print("\nDone. Charts saved under reports/figures/.")


if __name__ == "__main__":
    main()
