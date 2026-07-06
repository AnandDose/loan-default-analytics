USE DATABASE LOAN_ANALYTICS;
USE SCHEMA RAW;
USE WAREHOUSE LOAN_WH;

-- 1. Row count sanity check
SELECT COUNT(*) AS total_rows FROM RAW_LOANS;

-- 2. Null audit on key columns
SELECT
  COUNT(*) - COUNT(loan_amnt)   AS null_loan_amnt,
  COUNT(*) - COUNT(annual_inc)  AS null_annual_inc,
  COUNT(*) - COUNT(dti)         AS null_dti,
  COUNT(*) - COUNT(emp_length)  AS null_emp_length,
  COUNT(*) - COUNT(revol_util)  AS null_revol_util,
  COUNT(*) - COUNT(mort_acc)    AS null_mort_acc
FROM RAW_LOANS;

-- 3. Distinct categorical values (sanity check on data quality)
SELECT DISTINCT grade FROM RAW_LOANS ORDER BY grade;
SELECT DISTINCT loan_status FROM RAW_LOANS;
SELECT DISTINCT home_ownership FROM RAW_LOANS;
SELECT DISTINCT purpose FROM RAW_LOANS ORDER BY purpose;

-- 4. Overall default rate
SELECT
  COUNT(*)                                        AS n_loans,
  ROUND(100.0 * SUM(default_flag) / COUNT(*), 2)  AS default_rate_pct
FROM RAW_LOANS;

-- 5. Default rate and average interest rate by grade
SELECT
  grade,
  COUNT(*)                                                          AS n_loans,
  ROUND(100.0 * SUM(default_flag) / COUNT(*), 2)                    AS default_rate_pct,
  ROUND(AVG(TRY_CAST(REPLACE(int_rate, '%', '') AS FLOAT)), 2)      AS avg_int_rate,
  ROUND(AVG(loan_amnt), 0)                                          AS avg_loan_amnt
FROM RAW_LOANS
GROUP BY grade
ORDER BY grade;

-- 6. Default rate by loan purpose
SELECT
  purpose,
  COUNT(*)                                        AS n_loans,
  ROUND(100.0 * SUM(default_flag) / COUNT(*), 2)  AS default_rate_pct
FROM RAW_LOANS
GROUP BY purpose
ORDER BY default_rate_pct DESC;

-- 7. Default rate and average income by home ownership
SELECT
  home_ownership,
  COUNT(*)                                        AS n_loans,
  ROUND(100.0 * SUM(default_flag) / COUNT(*), 2)  AS default_rate_pct,
  ROUND(AVG(annual_inc), 0)                       AS avg_annual_income
FROM RAW_LOANS
GROUP BY home_ownership
ORDER BY default_rate_pct DESC;

-- 8. Income and DTI: defaulted vs. non-defaulted (a preview of t-test)
SELECT
  default_flag,
  COUNT(*)                    AS n_loans,
  ROUND(AVG(annual_inc), 0)   AS avg_income,
  ROUND(AVG(dti), 2)          AS avg_dti
FROM RAW_LOANS
GROUP BY default_flag;

-- 9. Date range covered by this dataset
SELECT MIN(issue_d) AS earliest_issue, MAX(issue_d) AS latest_issue FROM RAW_LOANS;
