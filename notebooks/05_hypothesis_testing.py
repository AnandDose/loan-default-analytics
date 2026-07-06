import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.width", 160)
ALPHA = 0.05
DATA_PATH = "data/processed/loans_engineered.csv"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def hypothesis_testing_mechanism() -> None:
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING MECHANISM")
    print("=" * 70)
    print(
        "Every test below follows the same mechanism:\n"
        "  1. State H0 (no effect/no difference) and H1 (an effect exists)\n"
        f"  2. Choose a significance level, alpha = {ALPHA}\n"
        "  3. Compute a test statistic from the sample\n"
        "  4. Convert it to a p-value: P(seeing a result this extreme\n"
        "     IF H0 were true)\n"
        "  5. If p-value < alpha, reject H0 -- the result is 'statistically\n"
        "     significant'. Otherwise, fail to reject H0.\n"
        "A p-value is NOT the probability H0 is true -- it's the probability of\n"
        "the data, assuming H0 is true. That distinction trips people up constantly."
    )


def two_proportion_z_test(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Z-TEST FOR TWO PROPORTIONS")
    print("=" * 70)


    is_verified = df["verif_Not Verified"] == 0
    verified = df.loc[is_verified, "default_flag"]
    not_verified = df.loc[~is_verified, "default_flag"]

    print(
        "H0: default rate is the same for verified and unverified applicants\n"
        "H1: default rates differ\n\n"
        f"Verified     (n={len(verified):,}): default rate = {verified.mean():.4f}\n"
        f"Not verified (n={len(not_verified):,}): default rate = {not_verified.mean():.4f}"
    )

    counts = np.array([verified.sum(), not_verified.sum()])
    nobs = np.array([len(verified), len(not_verified)])
    z_stat, p_value = proportions_ztest(counts, nobs)

    print(
        f"\nz-statistic = {z_stat:.3f}, p-value = {p_value:.6f}\n"
        f"{'Reject H0' if p_value < ALPHA else 'Fail to reject H0'}: the difference "
        f"{'is' if p_value < ALPHA else 'is not'} statistically significant at alpha={ALPHA}.\n"
        "(If verified applicants show a HIGHER default rate here, that's not\n"
        "verification causing default -- lenders often verify income specifically\n"
        "for riskier-looking applications in the first place.)"
    )


def two_sample_t_test(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STUDENT'S T-DISTRIBUTION & TWO-SAMPLE T-TEST")
    print("=" * 70)

    defaulted = df.loc[df["default_flag"] == 1, "annual_inc"]
    paid = df.loc[df["default_flag"] == 0, "annual_inc"]

    print(
        "H0: mean annual_inc is equal for defaulters and non-defaulters\n"
        "H1: the means differ\n\n"
        f"Defaulted (n={len(defaulted):,}): mean income = ${defaulted.mean():,.0f}\n"
        f"Paid off  (n={len(paid):,}): mean income = ${paid.mean():,.0f}"
    )

    t_stat, p_value = stats.ttest_ind(defaulted, paid, equal_var=False) 
    print(
        f"\nWelch's t-statistic = {t_stat:.3f}, p-value = {p_value:.2e}\n"
        f"{'Reject H0' if p_value < ALPHA else 'Fail to reject H0'}: income "
        f"{'does' if p_value < ALPHA else 'does not'} differ significantly by outcome.\n\n"
        "We used a T-TEST, not a Z-test, because we don't know the true\n"
        "population standard deviation -- it's estimated from the sample, which\n"
        "is exactly the situation the t-distribution exists for."
    )


def z_vs_t(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Z-TEST VS. T-TEST")
    print("=" * 70)

    n = len(df)
    t_crit_large_df = stats.t.ppf(0.975, df=n - 1)
    z_crit = stats.norm.ppf(0.975)
    t_crit_small_df = stats.t.ppf(0.975, df=10)

    print(
        f"With n={n:,}, the t-distribution's degrees of freedom are so high that\n"
        "it's numerically almost identical to the Normal distribution -- a t-test\n"
        "and a z-test give nearly the same p-value here. The distinction matters\n"
        "most with SMALL samples (rule of thumb: n < 30), where the t-distribution's\n"
        "fatter tails correctly account for the extra uncertainty of estimating\n"
        "the standard deviation from limited data.\n\n"
        f"Critical value at alpha=0.05 (two-tailed):\n"
        f"  Normal (z):                   {z_crit:.4f}\n"
        f"  t-distribution, df={n-1:,}:  {t_crit_large_df:.4f}  (essentially identical to z)\n"
        f"  t-distribution, df=10:         {t_crit_small_df:.4f}  (noticeably wider)"
    )


def type_i_ii_errors() -> None:
    print("\n" + "=" * 70)
    print("TYPE I & TYPE II ERRORS")
    print("=" * 70)
    print(
        "In the tests above, alpha=0.05 sets our Type I error rate: a 5% chance\n"
        "of concluding there's a real difference when there isn't one (a false\n"
        "positive). The unstated risk is Type II error: failing to detect a real\n"
        "difference that does exist (a false negative).\n\n"
        "Framed as a lending risk-flagging decision:\n"
        "  Type I error:  flagging a genuinely creditworthy applicant as risky\n"
        "                 -> lost business, an unnecessary decline\n"
        "  Type II error: failing to flag a genuinely risky applicant\n"
        "                 -> an approved loan that defaults, a direct financial loss\n"
        "A lender that cares more about avoiding defaults than approval volume\n"
        "would deliberately raise alpha (accept more Type I errors) to shrink\n"
        "Type II error -- that trade-off is a business decision, not a statistical one."
    )


def bayes_theorem(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BAYES' THEOREM")
    print("=" * 70)

    dti_threshold = df["dti"].median()
    high_dti = df["dti"] > dti_threshold

    p_default = df["default_flag"].mean()
    p_high_dti = high_dti.mean()
    p_high_dti_given_default = high_dti[df["default_flag"] == 1].mean()

    p_default_given_high_dti_bayes = (p_high_dti_given_default * p_default) / p_high_dti
    p_default_given_high_dti_direct = df.loc[high_dti, "default_flag"].mean()

    print(
        f"Question: given a borrower has above-median DTI (> {dti_threshold:.1f}),\n"
        f"what's P(default)?\n\n"
        f"P(default)             = {p_default:.4f}\n"
        f"P(high DTI)            = {p_high_dti:.4f}\n"
        f"P(high DTI | default)  = {p_high_dti_given_default:.4f}\n\n"
        f"Bayes: P(default | high DTI) = P(high DTI|default) * P(default) / P(high DTI)\n"
        f"                             = {p_default_given_high_dti_bayes:.4f}\n\n"
        f"Direct calculation from the data, for comparison: {p_default_given_high_dti_direct:.4f}\n"
        "(These match, as they should -- Bayes' theorem isn't a different answer,\n"
        "it's a different route to the same one, useful when you only have\n"
        "P(high DTI | default) and need to flip the conditioning around.)"
    )


def confidence_interval(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CONFIDENCE INTERVAL & MARGIN OF ERROR")
    print("=" * 70)

    p_hat = df["default_flag"].mean()
    n = len(df)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    z = stats.norm.ppf(0.975)
    margin_of_error = z * se
    ci_low, ci_high = p_hat - margin_of_error, p_hat + margin_of_error

    print(
        f"Point estimate p_hat = {p_hat:.4f}\n"
        f"Standard error = {se:.5f}\n"
        f"95% margin of error = +/-{margin_of_error:.4f}\n\n"
        f"95% CI for the true population default rate: [{ci_low:.4f}, {ci_high:.4f}]\n\n"
        "Correct interpretation: if this sampling process were repeated many\n"
        "times, 95% of such intervals would contain the true population default\n"
        "rate. It does NOT mean 'there's a 95% chance the true rate is in this range.'"
    )


def chi_square_tests(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CHI-SQUARE TEST OF INDEPENDENCE")
    print("=" * 70)

    contingency = pd.crosstab(df["grade"], df["default_flag"])
    print(contingency)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(
        f"\nH0: grade and default outcome are independent\n"
        f"H1: they are associated\n\n"
        f"chi2 = {chi2:.2f}, dof = {dof}, p-value = {p_value:.2e}\n"
        f"{'Reject H0' if p_value < ALPHA else 'Fail to reject H0'}: grade and default "
        f"{'are' if p_value < ALPHA else 'are not'} significantly associated."
    )

    print("\n" + "=" * 70)
    print("CHI-SQUARE GOODNESS OF FIT")
    print("=" * 70)

    observed_defaults = df.groupby("grade")["default_flag"].sum()
    n_per_grade = df.groupby("grade").size()
    overall_rate = df["default_flag"].mean()
    expected_defaults = n_per_grade * overall_rate

    gof_stat, gof_p = stats.chisquare(observed_defaults, expected_defaults)
    print(
        f"H0: every grade defaults at the same overall rate ({overall_rate:.4f})\n"
        f"H1: default rates differ by grade\n\n"
        f"chi2 = {gof_stat:.2f}, p-value = {gof_p:.2e}\n"
        f"{'Reject H0' if gof_p < ALPHA else 'Fail to reject H0'}: default rate "
        f"{'does' if gof_p < ALPHA else 'does not'} vary meaningfully by grade.\n"
    )


def anova(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ANOVA: WHAT, ASSUMPTIONS, TYPES, PARTITIONING")
    print("=" * 70)

    grades = sorted(df["grade"].unique())
    groups = [df.loc[df["grade"] == g, "int_rate"].dropna() for g in grades]

    print(
        "One-way ANOVA tests\n"
        "whether mean int_rate differs across all grades AT ONCE, instead of\n"
        "running 21 separate pairwise t-tests (which would inflate Type I error).\n\n"
        "H0: mean int_rate is equal across all 7 grades\n"
        "H1: at least one grade's mean differs"
    )

    print("\nAssumption checks:")
    levene_stat, levene_p = stats.levene(*groups)
    print(
        f"  Homogeneity of variance (Levene's test): p = {levene_p:.2e} "
        f"({'violated' if levene_p < ALPHA else 'holds'})\n"
        "  Independence: satisfied by the sampling design.\n"
        "  Normality within each group: with thousands of loans per grade, the\n"
        "  Central Limit Theorem makes this a non-issue for the F-test's\n"
        "  validity even though individual int_rate values aren't Normal."
    )

    f_stat, p_value = stats.f_oneway(*groups)
    print(
        f"\nF-statistic = {f_stat:.2f}, p-value = {p_value:.2e}\n"
        f"{'Reject H0' if p_value < ALPHA else 'Fail to reject H0'}: grade "
        f"{'does' if p_value < ALPHA else 'does not'} significantly affect mean int_rate."
    )

    print("\nPartitioning of variance: SST = SSB + SSW")
    all_values = df["int_rate"].dropna()
    grand_mean = all_values.mean()

    sst = ((all_values - grand_mean) ** 2).sum()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)

    print(
        f"  SST (total)         = {sst:,.1f}\n"
        f"  SSB (between-group) = {ssb:,.1f}  ({ssb / sst:.1%} of total variance -- explained by grade)\n"
        f"  SSW (within-group)  = {ssw:,.1f}  ({ssw / sst:.1%} of total variance -- unexplained)\n"
        f"  SSB + SSW           = {ssb + ssw:,.1f}  "
        f"({'matches SST' if abs((ssb + ssw) - sst) < 1 else 'MISMATCH vs SST'})"
    )


def main():
    print("Loading engineered data...")
    df = load_data()
    print(f"Ready: {len(df):,} rows")

    hypothesis_testing_mechanism()
    two_proportion_z_test(df)
    two_sample_t_test(df)
    z_vs_t(df)
    type_i_ii_errors()
    bayes_theorem(df)
    confidence_interval(df)
    chi_square_tests(df)
    anova(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
