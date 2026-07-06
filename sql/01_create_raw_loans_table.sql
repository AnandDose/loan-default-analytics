USE DATABASE LOAN_ANALYTICS;
USE SCHEMA RAW;

CREATE OR REPLACE TABLE RAW_LOANS (
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
