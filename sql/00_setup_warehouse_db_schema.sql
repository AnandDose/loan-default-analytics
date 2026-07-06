
-- 1. Warehouse: X-Small is enough for this project, auto-suspends after 60s idle
--    to avoid burning trial credits.
CREATE WAREHOUSE IF NOT EXISTS LOAN_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Warehouse for the loan default analytics project';

-- 2. Database and schemas
--    RAW holds untouched loaded data. ANALYTICS holds cleaned/feature-engineered views.
CREATE DATABASE IF NOT EXISTS LOAN_ANALYTICS
  COMMENT = 'Bank loan default risk analytics project';

USE DATABASE LOAN_ANALYTICS;

CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;

-- 3. Set context and create an internal stage for the Day 2 data load
USE WAREHOUSE LOAN_WH;
USE SCHEMA RAW;

CREATE STAGE IF NOT EXISTS LOAN_STAGE
  COMMENT = 'Internal stage for uploading the raw loans CSV';

-- 4. Sanity check — should return LOAN_WH / LOAN_ANALYTICS / RAW
SELECT CURRENT_WAREHOUSE() AS warehouse,
       CURRENT_DATABASE()  AS database,
       CURRENT_SCHEMA()    AS schema;
