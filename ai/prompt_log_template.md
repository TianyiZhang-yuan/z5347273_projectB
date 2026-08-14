# Prompt log - <task name>

## What I wanted
## Prompt(s)
## What the assistant produced
## What was wrong or risky
## What I changed and why

---

**### Prepare： ETL migration from Part A: equity and crypto**
## What I wanted
Migrate the relevant Station 1 equity and crypto cleaning logic from my completed Part A project into the Part B starter
file, so Part B could reuse the same verified data foundation without depending directly on the Part A folder.
## Prompt(s)
I asked the assistant to inspect my Part A and Part B etl.py files, identify which Station 1 functions were relevant, 
and migrate only the equity and crypto cleaning logic into Part B while keeping the Part B project structure
and data-access interface unchanged.
## What the assistant produced
The assistant replaced the Part B TODO placeholders with working ETL logic for equity and crypto data. 
It migrated duplicate checks, missing-date audits, extreme-return screening, and OHLCV consistency checks. It also added the 2020–2023 crypto date filter and kept all data loading through Part B’s src.data_access helper.
### What was wrong or risky
The migrated functions returned (df, checks), while the original Part B starter functions returned only df. 
This creates a possible interface risk
## What I changed and why
I kept the migrated return structure:
return df, checks
This keeps both the cleaned data and useful integrity-check results, such as duplicates, missing dates, extreme returns, and OHLCV consistency.
I only migrated the equity and crypto cleaning logic required by Part B and excluded Part A-specific report and figure code to avoid duplication.
Downstream code will unpack both outputs correctly:
equity_df, equity_checks = load_clean_equities()
crypto_df, crypto_checks = load_clean_crypto()
Part B still loads data through its own src.data_access, so it remains independent and reproducible.

**### Prepare： ETL migration from Part A: news**
## What I wanted
Migrate the relevant Part A news-cleaning and headline-assembly logic into Part B so the headlines were correctly
aligned to the equity trading calendar and ready for the Station 3 sentiment model.
## Prompt(s)
I asked the assistant to inspect the Part A and Part B ETL and feature files, then migrate only 
the required news-processing logic while keeping sentiment scoring and signal lagging outside features.py.
## What the assistant produced
The assistant added logic to standardise news dates, remove exact duplicates using ticker, date, and title, 
and map weekend or holiday headlines to the next equity trading day. It also completed assemble_headline_panel(), 
which validates the required columns, preserves the raw headline text, and returns an organised headline-level panel.
## What was wrong or risky
nothing risk
## What I changed and why
nothing need change
****### Prepare： ETL migration from Part A：feature**
## What I wanted
Migrate the daily return calculation logic from Part A, along with the daily headline panel assembly logic, 
into src/features.py in Part B, replacing the original raise NotImplementedError placeholders in the daily_returns()
and assemble_headline_panel() functions, so that Station 2's feature engineering can run independently within the Part B project.
## Prompt(s)
I previously implemented the return calculation and headline-panel assembly in z5347273_projectA. 
Please migrate the relevant logic into z5347273_projectB/src/features.py by:
1. Finding the corresponding Part A code.
2. Implementing daily_returns() using adjClose by ticker in long format for both equities and crypto.
3. Implementing assemble_headline_panel() using cleaned news data, without sentiment scoring.
4. Running validation tests and printing sample outputs.
## What the assistant produced
1. daily_returns(): Calculates returns by ticker using pct_change(fill_method=None), 
keeps long format, works for both equity and crypto, and validates required columns.
2. assemble_headline_panel(): Validates cleaned-news fields, sorts by trading date, ticker, 
and sector, preserves raw headlines, performs no sentiment processing, and stores validation results in panel.attrs["headline_panel_checks"]
## What was wrong or risky
nothing risk
## What I changed and why
nothing need change

**### stage 0：Portfolio Optimisation Design Rules** 
## What I wanted
1. Establish clear portfolio design rules before writing any optimisation functions.
2. Define common constraints, including long-only weights, full investment and a 20% single-asset cap.
3. Separate the validation requirements for Min-Variance, Max-Sharpe, Min-CVaR and Equal Weight.
4. Prevent look-ahead bias by specifying that weights estimated using data through date t can only be applied from date t+1.
5. Clarify that the Min-CVaR extension is a pure downside-protection model with no expected-return constraint.
## Prompt(s)
I am starting the development of z5347273_projectB/src/portfolios.py, which is the core file for Station 3. 
Before implementing any portfolio-weight solvers, establish a clear design specification at the top of the file.
Define the following:
Define the following:
1. Common constraints for Max-Sharpe, Min-Variance and Min-CVaR: long-only weights, weights summing to one, and a default 
20% single-asset cap. Add a feasibility rule requiring the number of valid assets multiplied by the weight cap to be at least one.
Any constraint relaxation or fallback must be recorded rather than happening silently.
2. Method-specific checks: Min-Variance and Max-Sharpe should check covariance-matrix stability and use ridge 
regularisation when necessary. Min-CVaR should instead check the number of tail observations and record 
linear-programming solver status. Equal Weight only requires at least one valid asset.
3.A strict timing rule: weights estimated using data through the close of date t must only become effective from date t+1, 
and must never be applied to the return observed on date t.
4. Specify that the project uses pure Minimum CVaR, which minimises tail loss without an expected-return constraint.
Do not implement the optimisation functions yet. Only add the design specification and runtime configuration.
## What the assistant produced
The AI added a Phase 0 specification to src/portfolios.py, including:
1. LONG_ONLY = True
2. FULLY_INVESTED_WEIGHT = 1.0
3. DEFAULT_MAX_ASSET_WEIGHT = 0.20
4. A covariance condition-number threshold and ridge regularisation value
5. A default CVaR confidence level
6. A minimum number of tail observations
7. A reusable weight effective-date rule
8. A read-only PORTFOLIO_CONFIG structure containing method-specific validation requirements
## What was wrong or risky
The constant:
MIN_CVAR_CONFIDENCE_LEVEL = 0.95 was ambiguously named. It could be interpreted as the minimum permitted confidence level, 
although 95% was intended to be the project’s default setting.
The generated MIN_CVAR_WEIGHTS_DOCSTRING constant also duplicated information that would be more naturally 
placed directly inside the future min_cvar_weights() function docstring.
## What I changed and why
I also renamed:
MIN_CVAR_CONFIDENCE_LEVEL to:DEFAULT_CVAR_CONFIDENCE_LEVEL and updated the related configuration key. This makes it clear that 95% is the pre-specified main setting, while other 
confidence levels may later be used only for sensitivity analysis.

**### stage 1** 
## What I wanted
I wanted to implement four portfolio weight solvers in src/portfolios.py:
1. Equal Weight (transparent baseline)
2. Minimum Variance (minimise overall volatility)
3. Maximum Sharpe (maximise risk-adjusted return)
4. Pure Minimum CVaR (minimise historical tail losses, no expected-return constraint)
All four methods needed to share the same output structure (weights +
diagnostic metadata), enforce long-only and fully-invested portfolios,
apply a 20% single-asset cap on optimised methods, record solver outcomes,
and fall back to Equal Weight if optimisation failed.
Before moving to the walk-forward backtest, I wanted to verify that all
four solvers produced valid, meaningfully different allocations.
## Prompt(s)
Implement only the four Stage 1 portfolio-weight solver functions in src/portfolios.py: equal_weight_weights(),
min_variance_weights(), max_sharpe_weights() and min_cvar_weights().
All functions should accept a historical return DataFrame and return a consistent (weights, metadata) tuple. The weight 
output should be a pandas Series indexed by ticker, while metadata should record solver success, status, fallback use, effective weight cap and relevant diagnostics.
Equal Weight should divide the portfolio equally across all valid assets.
Minimum Variance and Maximum Sharpe should be long-only, fully invested and subject to the effective single-asset cap. 
They should check covariance-matrix stability and apply ridge regularisation when required.
Minimum CVaR should use a historical linear-programming formulation. It should minimise tail loss only,
without an expected-return constraint, and should record the number of complete observations, the number of tail observations and the LP solver status.
Optimisation failures must not occur silently. Failed or invalid optimisation results should explicitly fall 
back to Equal Weight and record the reason.
After implementation, validate all four methods using the same reproducible synthetic return data. Confirm that
weights sum to one, contain no negative or non-finite values, satisfy the relevant cap, and differ meaningfully across methods. Also test cap relaxation, missing columns, non-numeric input, insufficient covariance history and near-singular covariance matrices.

Do not implement the Stage 2 walk-forward backtest yet.
## What the assistant produced
The AI implemented the four weight solvers plus shared helper functions
covering: return-input validation, removal of all-missing asset columns,
non-numeric column rejection, complete-case handling, 20% single-asset
cap feasibility checks with transparent relaxation, final weight
validation, Equal Weight fallback, covariance condition-number checking
with ridge regularisation, and standardised solver metadata.

All four methods return the same structure: weights, metadata. The
metadata records: solver success/status/message, whether fallback was
used, valid asset/observation counts, effective max asset weight,
covariance condition numbers before/after regularisation, CVaR
confidence level and tail count, and weight-validation results with
diagnostic warnings.
## What was wrong or risky
The first version did not handle missing returns in the same way across all methods. 
Minimum Variance and Maximum Sharpe could use different dates for different assets, 
while Minimum CVaR only used dates with complete data for every asset. This made the comparison less fair.
It also did not clearly reject non-numeric columns or record the covariance condition number after ridge regularisation.
## What I changed and why
I retained the common (weights, metadata) return structure because it allows the future walk-forward function to call
all portfolio methods through a consistent interface.
I accepted complete-case handling across all optimised methods so that Minimum Variance, Maximum Sharpe and Minimum
CVaR use the same dates within each estimation window. This makes their results more directly comparable and avoids silently 
treating missing returns as zero.
I also retained explicit fallback and metadata recording. This ensures that a completed backtest cannot hide optimiser 
failures behind apparently valid Equal Weight results.
The validation confirmed that the four methods were not accidentally producing identical portfolios and that each method
followed its intended objective.

**stage2**
## What I wanted
I wanted to build Stage 2 of the portfolio module by implementing a reusable walk-forward out-of-sample backtesting framework in src/portfolios.py.
The framework needed to:
1. use a fixed historical estimation window;
2. rebalance portfolios monthly;
3. calculate portfolio weights using only past information;
4. apply newly estimated weights from the next trading date;
5. reuse the Stage 1 portfolio solvers;
6. calculate daily out-of-sample portfolio returns;
7. allow portfolio weights to drift between rebalances;
8. record target weights and pre-trade weights;
9. calculate realistic turnover;
10. record rebalance dates and effective dates;
11. calculate standard portfolio performance metrics
## Prompt(s)
Implement Stage 2 walk-forward out-of-sample backtesting in src/portfolios.py.
Create an oos_backtest() function that uses a fixed rolling estimation window and monthly rebalancing.
Reuse the Stage 1 portfolio solvers for Equal Weight, Minimum Variance, Maximum Sharpe and Minimum CVaR.
Weights estimated using data available through date t must not be applied to the return on date t. They must become effective from the next available trading date.
Track target weights, pre-trade drifted weights and turnover. Turnover must be calculated using the difference between the new target weights and the actual pre-trade weights:
0.5 × sum(abs(target weights - pre-trade weights))
Do not calculate turnover using two consecutive target-weight vectors.
Do not fill missing live returns for currently held assets with zero. Raise a clear error instead.
Add performance_metrics() to calculate annualised return, annualised volatility, Sharpe ratio, maximum drawdown and final Growth of $1.
Preserve the existing Stage 1 solver logic and keep the implementation reusable.
Extend oos_backtest() with the optional parameters crypto_assets, crypto_cap and crypto_cap_rule.
The Crypto cap must limit the combined weight of all assets identified in crypto_assets. It must be enforced inside the selected optimiser and checked again after solving.
Record the cap used in the rebalance log. Do not add test code, temporary data or unrelated later-stage functions to portfolios.py
## What the assistant produced
1. _validate_backtest_returns() checks that the return matrix has valid, unique asset columns and correctly ordered dates.
2. _solver_for_method() connects each portfolio method to its corresponding Stage 1 weight solver.
3. oos_backtest() runs the monthly walk-forward backtest, applies weights from the next trading day, tracks weight drift and turnover, and records the backtest results.
4. performance_metrics() calculates annualised return, volatility, Sharpe ratio, maximum drawdown and Growth of $1.
5. The existing solvers were slightly extended to support an optional total Crypto allocation cap without changing their original optimisation logic.
## What was wrong or risky
Missing returns being treated as zero
Filling a missing live return with zero would incorrectly assume that the asset price did not move.
This could distort portfolio returns and hide problems in the aligned return panel.
## What I changed and why
I kept missing held-asset returns as errors
The function raises a clear error rather than filling missing returns with zero.
I kept this behaviour because silently inserting zero returns would produce unsupported backtest results.

**### stage 4**
## What I wanted
I wanted to use the completed portfolio solvers and walk-forward backtest to generate all baseline funds across three asset universes:
1. Equity-only
2. Crypto-only
3. Combined Equity and Crypto
Each universe was tested using:
1. Equal Weight
2. Minimum Variance
3. Maximum Sharpe
4. Minimum CVaR
The goal was to run all 12 funds consistently, validate their outputs, 
and save consolidated results for later tables, charts and the Streamlit app.
## Prompt(s)
Implement Stage 4 baseline fund generation in scripts/run_part_b.py.
Reuse the existing ETL, return calculations, Stage 1 portfolio solvers and Stage 2 oos_backtest() function.
Construct Equity-only, Crypto-only and Combined return matrices. Calculate Crypto returns on the native Crypto calendar before aligning them to the equity calendar for the Combined funds.
Run Equal Weight, Minimum Variance, Maximum Sharpe and Minimum CVaR for each asset universe.
Use a 252-day window and annualisation for Equity and Combined funds, and a 365-day window and annualsation for Crypto funds.
Validate returns, weights, rebalance dates and asset-weight constraints.
Save consolidated fund returns, weights, performance metrics and run-status outputs. Do not rewrite the portfolio or backtesting functions.
## What the assistant produced
1. _wide_returns() converts long-format returns into a validated wide return matrix.
2. _build_return_universes() constructs the Equity, Crypto and Combined return panels while preserving the correct calendar logic.
3. _fund_configs() creates the 12 baseline fund configurations without repeating long code blocks.
4. _validate_fund_outputs() checks fund returns, weight sums, non-negative weights, timing and individual asset caps.
5. _tidy_fund_returns() combines all fund returns into one tidy table.
6. _tidy_target_weights() combines all target weights into one tidy table.
7. _run_one_fund() runs one configured fund, calculates performance metrics and returns its saved outputs.
main() runs all 12 funds and saves the consolidated result files.
The main outputs are:
results/data/fund_returns.csv
results/data/fund_weights.csv
results/tables/R1.performance_metrics.csv
results/tables/baseline_fund_run_status.csv
## What was wrong or risky
The first Stage 4 version saved seven separate outputs for each of the 12 funds, which created about 84 CSV files. 
Although the results were complete, the output structure was too fragmented and made the results folder difficult to manage.
This also increased the risk of duplicated information, 
inconsistent file naming and difficulty locating the files needed for later tables, charts and the Streamlit app.
## What I changed and why
I changed the output structure from many per-fund files to a small number of consolidated tables.
All daily fund returns were combined into fund_returns.csv, all target weights were combined into fund_weights.csv, 
and all performance results were combined into R1.performance_metrics.csv. I also kept one run-status table to confirm 
that all funds completed successfully and satisfied the weight and timing checks.
This made the output folder much cleaner while preserving the information needed for fund comparison, visualisation and app display.

**### stage 5**
## What I wanted
I wanted to turn the saved Stage 4 fund results into a clear, investor-friendly Fund Fact Sheet inside the existing Streamlit app.
The app needed to let users select a baseline fund and view its setup, performance, latest holdings, 
current Crypto allocation and Growth of $1 without rerunning portfolio optimisation or backtesting.
## Prompt(s)
Implement Stage 5 Fund Fact Sheet inside the existing Funds tab in streamlit_app.py.
Read the saved Stage 4 outputs:
1. results/tables/R1.performance_metrics.csv
2. results/data/fund_weights.csv
3. results/data/fund_returns.csv
Add a baseline-fund selector and display the selected fund’s setup, annualised return, volatility, 
Sharpe ratio, maximum drawdown, Growth of $1, latest target allocation, top five holdings and current total Crypto weight.
Use an explicit Crypto asset list rather than guessing from ticker names. 
Do not rerun portfolio optimisation or backtesting, and do not modify src/portfolios.py or scripts/run_part_b.py.
## What the assistant produced
_load_stage4_fund_results() loads and caches the three saved Stage 4 result files and reports missing files clearly.
_crypto_assets() loads the explicit Crypto ticker list used to calculate the selected fund’s total Crypto weight.
_title_case_label(), _method_label() and _fund_label() convert stored names into readable app labels.
_format_percent() formats performance values consistently as percentages.
render_fund_fact_sheet() creates the fund selector and displays the fund setup, five main performance metrics, 
latest rebalance date, top holdings, complete latest allocation, current Crypto weight and Growth of $1 chart.
The existing Sentiment and Data tabs were preserved, while the original Funds-tab placeholder was replaced with the Fund Fact Sheet.
## What was wrong or risky
The main risk was that the app could load mismatched data for the selected fund or accidentally rerun calculations 
instead of using the saved Stage 4 results. Crypto exposure also needed to be calculated from an explicit Crypto asset list rather than guessed from ticker names.
## What I changed and why
I linked the fund selector to the saved metrics, weights and returns files so all displayed information updates consistently. 
I also used the project’s Crypto ticker list to calculate total Crypto weight and added clear messages for missing files or invalid weights.

**### stage7 innovation**
## What I wanted
I wanted to establish a common portfolio optimisation foundation for my main innovation, the Personalised Explainable Adaptive Stock–Crypto Allocation system.
The innovation will later create six formal funds across three investor profiles:
1. Conservative Fixed 10%
2. Conservative Adaptive 0%–10%
3. Balanced Fixed 20%
4. Balanced Adaptive 5%–20%
5. Growth Fixed 30%
6. Growth Adaptive 10%–30%
For each investor profile, the Fixed fund acts as the comparison benchmark, while the Adaptive fund changes its
Crypto cap according to market risk. All six funds therefore need to use the same Pure 95% Minimum-CVaR method 
so that the Fixed and Adaptive versions can be compared fairly.
I also needed the existing optimiser to accept an optional aggregate Crypto cap, meaning that the total weight of 
all ten Crypto assets could not exceed a specified limit. The cap needed to remain an upper bound rather than a required allocation, while all existing baseline portfolio behaviour had to remain unchanged.
## Prompt(s)
I asked the AI agent to first inspect the existing src/portfolios.py implementation and understand how the current min_cvar_weights() optimiser, fallback logic, asset-weight constraints and metadata were already structured. The purpose was to extend the existing optimiser carefully, rather than creating a separate Stage 7 optimiser that duplicated baseline logic.
The prompt required the agent to:
preserve Pure 95% Minimum-CVaR as the only optimisation objective, so the optimiser would continue minimising the average loss in the worst 5% of historical return outcomes rather than introducing a blended or alternative objective;
retain all existing portfolio constraints, including long-only weights, full investment and the existing individual-asset cap;
add optional crypto_assets and crypto_cap inputs to the current min_cvar_weights() function, while ensuring that previous baseline calls without these inputs continued to work unchanged;
identify the positions of all Crypto assets in the supplied return matrix and impose an aggregate constraint requiring the total weight of those assets to remain less than or equal to the specified Crypto cap;
treat the Crypto cap strictly as an upper bound rather than a target allocation, meaning the optimiser could choose a lower Crypto allocation when this produced a lower-CVaR portfolio;
calculate the portfolio’s actual total Crypto allocation after optimisation and derive the actual Equity allocation as one minus the Crypto allocation;
record relevant metadata, including the requested Crypto cap, actual Crypto and Equity weights, whether the aggregate cap was binding, solver status and any fallback or cap-relaxation information;
preserve the existing covariance handling, feasibility checks, fallback policies and baseline outputs wherever they were already part of the optimiser;
avoid creating the six Conservative, Balanced and Growth funds at this stage, because this task was only intended to establish the common optimisation foundation needed by those later funds;
avoid generating new result files, figures, backtests, synthetic data or test code during this step.
This prompt was designed to keep Stage 7.1 narrowly focused. It established the technical ability to control the total Crypto exposure of a full stock–Crypto portfolio, while postponing the personalised Fixed and Adaptive fund construction until later stages.
## What the assistant produced
It extended the existing optimiser signature to include:
def min_cvar_weights(
    returns: pd.DataFrame,
    confidence_level: float = DEFAULT_CVAR_CONFIDENCE_LEVEL,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
It added an aggregate Crypto constraint equivalent to:
sum(weights[crypto_positions]) <= crypto_cap
It also added an asset_class_weights() helper to calculate:
actual_crypto_weight
actual_equity_weight
The optimiser metadata was extended to record information such as:
requested Crypto cap;
whether the Crypto-cap constraint was active;
actual Crypto and Equity weights;
whether the Crypto cap was binding;
solver and fallback information.
The assistant also created shared Stage 7 settings for Pure 95% Min-CVaR, a 252-observation estimation window, 
monthly rebalancing, long-only weights and the Combined universe of 50 Equity assets and 10 Crypto assets.
## What was wrong or risky
The initial implementation added the aggregate Crypto-cap constraint to the normal Minimum-CVaR optimisation, 
but it did not clearly confirm that the fallback portfolio would satisfy the same limit.
For example, an equal-weight fallback across 50 Equity and 10 Crypto assets would allocate:
10/60=16.67%
to Crypto. This would violate a 10% Conservative cap. Therefore, the main risk was that the cap could work during normal
optimisation but fail when fallback logic was triggered.
There was also a risk of treating the Crypto cap as a target. The cap should only be a maximum, 
so the optimiser must be allowed to allocate less than the permitted amount.
## What I changed and why
I revised the logic so that both the normal optimisation result and the fallback portfolio must satisfy the aggregate Crypto cap.
When a Crypto cap is provided, the fallback can no longer use unrestricted equal weights across all assets.
It must construct a feasible allocation with total Crypto weight below the cap while remaining fully invested.
I also kept the Crypto cap as a maximum limit rather than a required target. This means the combined weight of all 
Crypto assets must not exceed the specified Crypto cap, but the optimiser is allowed to allocate less than the maximum when this reduces portfolio risk.
rather than forcing Crypto weight to equal the cap. Metadata was added to record the requested cap, actual Crypto weight,
actual Equity weight, whether the cap was binding, and whether fallback was used.

**### stage 7.2**
## What I wanted
I wanted to complete Stage 7.2 by constructing a transparent Crypto market tail-risk indicator. The indicator 
uses an equal-weight return series across all ten Crypto assets and calculates a rolling 60-day 90% realised CVaR.
The purpose is to show that Crypto downside risk changes over time and therefore may not be well managed by one permanently fixed Crypto allocation rule.
## Prompt(s)
I asked the AI agent to:
1. use all ten Crypto assets rather than BTC alone;
2. calculate the equal-weight daily Crypto market return;
3. calculate rolling 60-day 90% realised CVaR using the worst six returns in each complete window;
4. convert the negative raw CVaR into a positive tail-risk value;
5. generate the daily risk dataset, summary table and risk-over-time figure;
6. preserve the existing Pure 95% Minimum-CVaR portfolio optimiser.
## What the assistant produced
The assistant implemented the Stage 7.2 functions in src/portfolios.py, including:
1. validation of the ten-Crypto return panel;
2. construction of the equal-weight Crypto market return;
3. calculation of rolling 60-day 90% realised CVaR;
4. conversion of raw CVaR into a positive tail-risk measure;
5. creation of summary statistics;
6. creation of the Crypto tail-risk time-series figure.
The execution logic was integrated into scripts/run_part_b.py, which now loads the cleaned Crypto data, calculates returns, calls the Stage 7.2 functions and saves the outputs.
Three Stage 7.2 outputs were produced:
results/data/stage7_crypto_tail_risk.csv
results/tables/stage7_crypto_tail_risk_summary.csv
results/figures/stage7_crypto_rolling_tail_risk.png
The main results were:
first valid tail-risk date: 1 March 2020;
minimum tail risk: 2.06%;
median tail risk: 7.00%;
mean tail risk: 7.33%;
maximum tail risk: 16.83%;
maximum tail-risk date: 21 June 2021.
## What was wrong or risky
The initial restructuring accidentally removed the original src/features.py file because it was incorrectly 
treated as a newly created Stage 7.2 module.
However, features.py was already part of the original Part B structure. It contains the Station 2 functions for
return-feature construction and headline-panel assembly. Removing it created a compatibility risk for later return and sentiment workflows.
To keep Stage 7.2 running, the assistant temporarily placed a duplicate _daily_returns() function inside run_part_b.py.
Although the calculation worked, this was not the intended structure because reusable feature logic should not remain inside the execution script.
There was also a risk that future code importing src.features.daily_returns() or assemble_headline_panel() would fail if the original module remained deleted.
## What I changed and why
I restored the original src/features.py module and asked the assistant to implement the working daily-return logic inside its existing daily_returns() function.
The duplicate _daily_returns() function was removed from scripts/run_part_b.py. 
The runner now imports and calls:
from src.features import daily_returns
The assemble_headline_panel() function was left unchanged because it belongs to the later news and sentiment workflow.
This correction restored the intended project structure without changing the Stage 7.2 methodology or numerical results. 
The Equity, Crypto and Combined return universes still built correctly, and the Stage 7.2 summary matched the previously saved results exactly.

**### 7.3** 
## What I wanted
1. Turn the Stage 7.2 Crypto tail-risk measure into a monthly risk score from 0 to 100.
2. Use the score later for the Adaptive Crypto-cap system.
3. Show how high the current Crypto risk is compared with the historical information available at that time.
4. A low score means the Crypto market is relatively calm.
5. A high score means current downside risk is close to its historical high.
6. Avoid look-ahead bias by using only past information.
7. Produce a score only after at least 60 valid daily tail-risk observations are available.
## Prompt(s)
1. reuse the completed Stage 7.2 positive Crypto tail-risk series;
2. identify the final valid Combined portfolio trading date in each month;
3. use the latest valid tail-risk observation available on or before each monthly score date;
4. compare the current tail risk with the available historical daily tail-risk observations;
5. convert the historical percentile into a score between 0 and 100;
6. require at least 60 valid historical observations before producing a formal score;
7. preserve early monthly rows with a missing score rather than replacing them with zero;
8. record the historical observation count and the exact tail-risk source date;
## What the assistant produced
1. The assistant added reusable functions for:
2. identifying monthly Combined portfolio decision dates
3. validating the Stage 7.2 tail-risk input
4. calculating the expanding historical percentile
5. building the monthly Crypto risk-score series
6. summarising and validating the Stage 7.3 outputs
The implementation produced 48 monthly rows from 31 January 2020 to 29 December 2023. The first three monthly scores
were unavailable because there were not yet 60 valid prior daily tail-risk observations.
The first formal score became available on 30 April 2020. Overall, 45 valid monthly scores were generated. 
The maximum score was 100 on 30 June 2021, while the minimum score was approximately 0.07 on 31 October 2023.
The final code uses only historical daily tail-risk observations dated before the current tail-risk source date, 
so the current observation is excluded from its own percentile ranking.
## What was wrong or risky
The initial method ranked each monthly tail-risk value against the full 2020–2023 sample. 
This caused look-ahead bias because early scores used future information.
There was also a risk of treating missing early scores as zero. A zero score means very low risk, 
while a missing score means there is not enough history.
## What I changed and why
I replaced the full-sample ranking with an expanding historical percentile.
For each month, the score now uses only earlier daily tail-risk observations. The current value is compared with past values only,
and no future data is included.
I also required at least 60 valid historical observations. Earlier months remain missing and are marked as unavailable instead of being assigned a zero score.
This made the Stage 7.3 risk score more realistic and free from look-ahead bias.
**### 7.4**
## What I wanted
1. Convert the Stage 7.3 monthly risk score into a personalised Crypto cap.
2. Create separate rules for Conservative, Balanced and Growth investors.
3. Make the Crypto cap fall continuously as market risk increases.
4. Keep the cap as a maximum limit, not a required Crypto allocation.
5. Produce a dataset, summary table and design figure without running the funds yet.
## Prompt(s)
I asked the AI agent to:
1. reuse the monthly 0–100 Crypto risk scores from Stage 7.3;
2. apply continuous linear cap rules for the three investor profiles;
3. map Conservative from 10% to 0%, Balanced from 20% to 5%, and Growth from 30% to 10%;
4. keep missing risk scores and caps as missing rather than treating them as zero;
5. save the long-format cap data and profile summary table;
6. create a figure showing Risk Score against Personalised Crypto Cap;
7. keep the Stage 7.1–7.3 logic and Pure 95% Minimum-CVaR optimiser unchanged;
8. avoid creating smoothing, portfolio weights, fund backtests or unnecessary validation code.
## What the assistant produced
Added reusable Stage 7.4 functions in src/portfolios.py.
Integrated the Stage 7.4 workflow into scripts/run_part_b.py.
Created:
1. results/data/personalised_crypto_caps.csv
2. results/tables/personalised_crypto_cap_summary.csv
3. results/figures/risk_score_vs_personalised_crypto_cap.png
Generated 45 valid monthly caps and 3 unavailable early caps for each profile.
The first valid cap date was 30 April 2020.
Confirmed the expected mappings:
Score 0: 10%, 20%, 30%
Score 50: 5%, 12.5%, 20%
Score 100: 0%, 5%, 10%
Confirmed that Conservative always had the lowest cap, Balanced remained in the middle and Growth had the highest cap.
## What was wrong or risky
The original output used the names minimum_crypto_cap and maximum_crypto_cap.
These names could be misunderstood as minimum and maximum actual Crypto holdings.
For example, Growth’s 10% lower cap could incorrectly suggest that the portfolio must always hold at least 10% Crypto.
In fact, these values only define the range of the Raw Crypto Cap. The future optimiser may choose an actual Crypto allocation below the cap.
## What I changed and why
I clarified that the values represent the minimum and maximum boundaries of the Raw Crypto Cap, not required portfolio allocations.
I changed the labels to minimum_raw_crypto_cap and maximum_raw_crypto_cap to make the meaning clearer.
I also clarified that raw_crypto_cap is an upper bound only.
This prevents the personalised cap rules from being confused with the actual Crypto weights that will later be selected by the Min-CVaR optimiser.
**###7.6**
## What I wanted
Convert the monthly Raw Crypto Caps into smoother Applied Crypto Caps.
Avoid unnecessary cap changes when the difference is very small.
Limit large monthly cap movements.
Keep early missing caps as missing.
Prepare the Applied Caps for the later Min-CVaR fund backtest.
## Prompt(s)
1. use the existing Stage 7.4 Raw Crypto Caps;
2. set the first valid Applied Cap equal to the first valid Raw Cap;
3. make no change when the difference is below 1 percentage point;
4. limit each monthly cap movement to 5 percentage points;
5. process Conservative, Balanced and Growth separately;
6. save the results in adaptive_cap_history.csv;
7. avoid creating fund weights, backtests or transaction costs at this stage;
8. keep the main code and validation concise.
## What the assistant produced
Added the smooth_crypto_cap_profile() function.
Integrated Stage 7.6 into scripts/run_part_b.py.
Created results/data/adaptive_cap_history.csv.
The first valid Applied Cap date was 30 April 2020 for all profiles.
The no-adjustment rule was triggered:
18 times for Conservative;
12 times for Balanced;
9 times for Growth.
The monthly movement limit was triggered:
2 times for Conservative;
11 times for Balanced;
15 times for Growth.
Missing early caps remained missing, and no portfolio weights were created.
## What was wrong or risky
A key risk was comparing the current Raw Cap with the previous Raw Cap instead of the previous Applied Cap.
This would make the smoothing process incorrect because each new Applied Cap should depend on the cap that was actually used in the previous month.
There was also a risk of treating the early missing caps as zero, which would incorrectly start the smoothing process before enough historical data was available.
## What I changed and why
Convert the monthly Raw Crypto Caps into smoother Applied Crypto Caps.
Avoid unnecessary cap changes when the difference is very small.
Limit large monthly cap movements.
Keep early missing caps as missing.
Prepare the Applied Caps for the later Min-CVaR fund backtest.

**### 7.7 Allocation History and Market Drift Tracking**

## What I wanted
* Combine the main allocation information into one monthly history.
* Show the difference between Raw Budget, Applied Budget, Target Budget, and actual portfolio allocation.
* Track how market movements changed the Crypto weight before each rebalance.
* Keep the output simple and useful for later analysis and reporting.
* Reuse the existing turnover and pre-trade drift logic instead of rebuilding the portfolio model.
## Prompt(s)
1. Merge the Stage 7.4 Raw Budget and Stage 7.6 Applied Budget with the Stage 7.5 fund results.
2. Keep one row for each fund and monthly rebalance date.
3. Include target Crypto and Equity budgets, pre-trade weights, and post-rebalance weights.
4. Use the existing drifted pre-trade weight calculation.
5. Keep Fixed and Adaptive funds separate.
6. Leave Raw and Applied Adaptive fields missing for Fixed funds.
7. Do not rerun the Min-CVaR optimiser.
8. Keep the Stage 7.7 code concise.
## What the assistant produced
* Added a monthly allocation-history output.
* Combined:
   Raw Crypto Budget;
    Applied Crypto Budget;
    Target Crypto Budget;
    pre-trade Crypto and Equity weights;
    post-rebalance Crypto and Equity weights;
    market drift;
    turnover.
* Created stage7_allocation_history.csv.
* Produced 216 monthly observations:
    6 funds;
   36 rebalance periods per fund.
* Confirmed that post-rebalance allocations matched the target budgets.
* Confirmed that market movements caused the pre-trade allocations to drift away from the previous targets.
However, the first version also created:
* build_stage7_daily_allocation_history(...);
* stage7_daily_allocation.csv;
* a full daily allocation panel;
* extra daily-drift and validation code.
## What was wrong or risky
The first implementation was too complicated for the purpose of Stage 7.7.
A full daily allocation history was not necessary because the monthly pre-trade weights already showed whether 
market movements caused the portfolio to drift away from its target.
The daily version also added a large amount of extra code and duplicated some of the drift logic that already
existed in the turnover calculation. This increased the risk of inconsistent calculations and made the project harder to maintain.
The issue was therefore mainly over-engineering, rather than an incorrect portfolio result.
## What I changed and why
I removed the daily allocation function and deleted the daily allocation CSV.
I kept only the monthly stage7_allocation_history.csv, which contains the information needed to explain the allocation process:
1. Raw Budget;
2. Applied Budget;
3. Target Budget;
4. pre-trade allocation;
5. post-rebalance allocation;
6. market drift;
7. turnover.
I also reused the existing pre-trade drift logic instead of creating a separate daily allocation system.
This made the code much shorter and kept the same economic information needed for the report. It also reduced duplication and made the Stage 7 workflow easier to maintain.

8. **### 7.8 Explainable Rebalance Decisions**
## What I wanted
1. Add explainability to the adaptive allocation process.
2. Generate a technical explanation for each rebalance.
3. Generate a simpler user-facing explanation for non-technical investors.
4. Explain why the Crypto allocation increased, decreased, or stayed unchanged.
5. Distinguish changes in the Risk Score from changes in the actual target allocation.
6. Keep Fixed and Adaptive funds clearly separated.
7. Avoid making unsupported causal claims about market events.
## Prompt(s)
1.Use the Stage 7.7 monthly allocation history as the input.
2. Generate one technical explanation and one simple user explanation for each rebalance.
3. For Adaptive funds, explain the Risk Score, Raw Budget, Applied Budget, smoothing action, and final Target Budget.
4. For Fixed funds, explain that the Crypto allocation stays fixed by profile and does not respond to the Risk Score.
5. Make sure the explanation matches the actual implemented allocation and does not make unsupported market-event claims.
6. Save the results in stage7_rebalance_explanations.csv without changing the existing 
## What the assistant produced
* Added automatic rebalance explanations for all six official funds.
* Created both:
    * technical explanations;
    * user-facing explanations.
* Generated one explanation for each monthly rebalance.
* Created stage7_rebalance_explanations.csv.
* The final output contained 216 rows:
    * 6 funds;
    * 36 rebalance dates per fund.
* Adaptive explanations included:
    * Risk Score;
    * Raw Budget;
    * Applied Budget;
    * smoothing action;
    * final Target Budget.
* Fixed fund explanations correctly described the allocation as profile-based and fixed rather than risk-responsive.
## What was wrong or risky
The first explanation logic was too simple. It mainly looked at whether the Risk Score increased or decreased
and then used this to describe the allocation decision.
This created several specific problems:
1. If the Risk Score increased, the explanation could say that Crypto exposure was reduced, even when the Applied Budget did not actually change because the difference was below the 1 percentage point threshold.
2. In some months, the Raw Budget moved sharply, but the 5 percentage point monthly limit meant that the Applied Budget only moved part of the way. The original wording did not clearly explain this difference.
3. The explanation could also describe a change as being caused by a market event, even though the model only observed changes in realised tail risk and did not identify the cause.
4. For Fixed funds, there was a risk of using the same adaptive explanation template even though their Crypto allocation does not respond to the Risk Score.
The main issue was therefore that the text could describe what the model “wanted to do” rather than what the portfolio actually implemented.
## What I changed and why
I changed the explanation logic to use the actual allocation fields rather than only the Risk Score.
The revised process checks the values in this order:
1. compare the current and previous Target Crypto Budget to decide whether the implemented allocation increased, decreased, or stayed unchanged;
2. compare the Raw Budget and Applied Budget to identify whether smoothing changed the requested adjustment;
3. check the smoothing trigger to distinguish between:
    * no adjustment because the difference was below 1 percentage point;
    * a limited adjustment because the 5 percentage point monthly limit was reached;
4. describe the Risk Score movement separately from the final allocation movement.
For example, if the Risk Score increased but the Applied Budget stayed unchanged, the explanation now says that risk 
increased but the Crypto allocation was maintained because the required change was too small.
If the Raw Budget fell by more than 5 percentage points, the explanation now says that the target Crypto allocation was reduced, 
but only by the maximum monthly adjustment allowed by the smoothing rule.
I also used a separate template for Fixed funds so that they are described as maintaining their profile-based Crypto 
Budget instead of reacting to the Risk Score.
This made the generated explanations match the actual rebalance decision more closely and reduced the risk of misleading users.
**### 7.9 Fixed vs Adaptive Performance Comparison**
## What I wanted
1. Compare each Adaptive fund with its matching Fixed fund.
2. Test whether the adaptive Crypto Budget improved portfolio risk.
3. Compare return, volatility, Sharpe ratio, maximum drawdown, CVaR, and worst 10-day return.
4. Keep Conservative, Balanced, and Growth as separate profile comparisons.
5. Avoid claiming that Adaptive “outperformed” unless both return and risk improved.
## Prompt(s)
1. Use the completed Stage 7 fund return and weight outputs.
2. Compare Conservative Fixed vs Adaptive, Balanced Fixed vs Adaptive, and Growth Fixed vs Adaptive.
3. Calculate annualised return, volatility, Sharpe ratio, maximum drawdown, realised 95% CVaR, worst 10-day return, and Growth of $1.
4. Summarise the differences between Fixed and Adaptive for each profile.
5. Clearly distinguish return improvements from downside-risk improvements.
6. Save the main comparison results in stage7_fixed_vs_adaptive_comparison.csv.
## What the assistant produced
* Created the Fixed vs Adaptive comparison table.
* Calculated the main performance and downside-risk metrics for all six funds.
* Showed that all three Adaptive funds had:
    * lower volatility;
    * smaller maximum drawdown;
    * less negative CVaR;
    * better worst 10-day returns.
* Also showed that all three Adaptive funds had:
    * lower annualised return;
    * lower Sharpe ratio.
* Created stage7_fixed_vs_adaptive_comparison.csv.
## What was wrong or risky
The original explanation logic could describe the Crypto allocation change based on the direction of the Risk Score.
This was not always correct because the Risk Score first changed the Raw Budget, and the smoothing rules could then keep the final Target Budget unchanged or limit the adjustment.
As a result, the explanation could say that Crypto allocation decreased even when the actual implemented Target Budget stayed the same.
## What I changed and why
I changed the explanation logic so that the allocation direction was based on the actual target_budget_change rather than the Risk Score.
The Risk Score movement and the allocation movement were then explained separately.
For example, if risk increased but smoothing kept the Target Budget unchanged, the explanation now says that 
risk increased but the Crypto allocation was maintained.
This made the explanation consistent with the portfolio decision that was actually implemented.
**### 7.10 Personalisation Validation**
## What I wanted
* Check whether the three investor profiles produced genuinely different Crypto allocations.
* Confirm that Conservative, Balanced and Growth were not only different labels.
* Compare their average Crypto Budgets and realised portfolio risk.
* Test whether higher-risk profiles actually held more Crypto and experienced more risk.
## Prompt(s)
1. Use the completed Adaptive fund results from Stage 7.9.
2. Compare Conservative, Balanced and Growth Adaptive funds.
3. Calculate the average Target Crypto Budget for each profile.
4. Compare annualised return, volatility, maximum drawdown, CVaR and worst 10-day return.
5. Check whether Crypto exposure increases from Conservative to Balanced to Growth.
6. Check whether realised portfolio risk also increases across the three profiles.
7. Save the comparison in stage7_personalisation_comparison.csv.
8. Keep the analysis focused on validating the personalisation design rather than comparing Fixed and Adaptive again.
## What the assistant produced
* Created stage7_personalisation_comparison.csv.
* Confirmed that average Adaptive Crypto Budgets were approximately:
    * Conservative: 6.03%
    * Balanced: 14.16%
    * Growth: 22.37%
* Confirmed that Crypto exposure increased clearly across the three profiles.
* Realised portfolio risk also increased from Conservative to Balanced to Growth.
* Growth produced the highest return, but also had higher volatility, deeper drawdown and worse downside-risk measures.
## What was wrong or risky
The main risk was only checking whether the three profiles had different names or different budget formulas.
That would not prove that the final portfolios were actually different.
I needed to check the realised average Crypto exposure and portfolio risk after the full backtest.
There was also a risk of treating the Growth profile as “better” simply because it had a higher return, even though it also carried more risk.
## What I changed and why
I validated the profiles using the actual backtest results instead of only the preset budget ranges.
I compared the average Crypto Budget and realised risk metrics across all three Adaptive funds.
This confirmed that:
Conservative < Balanced < Growth
for both Crypto exposure and overall portfolio risk.
I therefore described the result as successful personalisation, rather than saying that one profile was better than the others.

**### 7.11–7.12 Smoothing Validation and Robustness Review**
## What I wanted
* Check whether the smoothing rules actually made the Adaptive Crypto Budget more stable.
* Compare Raw Adaptive and Smoothed Adaptive portfolios.
* Test whether smoothing reduced large monthly changes, turnover and trading costs.
* Check whether this extra stability also improved portfolio performance.
* Review whether an additional sensitivity test was necessary before finalising the innovation.
## Prompt(s)
1. Use the existing Raw Adaptive and Smoothed Adaptive budget histories.
2. Compare average and maximum monthly Crypto Budget changes for Conservative, Balanced and Growth.
3. Compare turnover and estimated transaction costs between Raw and Smoothed versions.
4. Compare net annualised return, maximum drawdown and realised CVaR.
5. Do not rerun the Min-CVaR optimiser; reuse the existing sleeve weights.
6. Clearly separate implementation stability from investment performance.
7. Review whether extra 30/60/90-day sensitivity testing would add useful evidence or only increase complexity.
8. Save the main comparison in stage7_smoothing_comparison.csv.
## What the assistant produced
    Created stage7_raw_adaptive_results.csv.
* Created stage7_smoothing_comparison.csv.
* Compared Raw and Smoothed Adaptive results for all three profiles.
* Smoothing reduced the size of monthly Budget changes for all profiles.
* The largest improvement was for Growth, where the maximum monthly Budget movement fell from about 11.7% to 5%.
* Turnover and trading costs decreased for Balanced and Growth, but slightly increased for Conservative.
* Net performance did not improve consistently after smoothing.
* The additional 30/60/90-day sensitivity test was reviewed but was not kept as a formal Stage 7 output.
## What was wrong or risky
The original expectation was that smoothing would make the Adaptive strategy more stable and also reduce turnover and improve net performance.
However, the actual results were mixed.
* Smoothing reduced the size of monthly Crypto Budget changes for all three profiles.
* It reduced turnover for Balanced and Growth, but Conservative turnover slightly increased.
* Net annualised return became lower for all three Smoothed portfolios.
* Maximum drawdown also did not improve after smoothing.
* CVaR only improved slightly for Conservative and became worse for Balanced and Growth.

This meant I could not simply describe smoothing as an overall improvement.
## What I changed and why
I changed the comparison so that the Raw and Smoothed versions reused the same existing sleeve weights.
The main difference between them was therefore the Crypto Budget path rather than a new optimisation result.
I also changed the final interpretation after checking the actual results.
Instead of saying that smoothing improved the portfolio, I separated implementation stability from investment performance.
The final conclusion became:
Smoothing reduced large Budget movements and improved implementation stability, especially for Balanced and Growth, 
but it did not improve net performance.
I also kept the Conservative result in the analysis even though its turnover slightly increased, rather than only 
reporting the profiles where smoothing worked better.

**### Sentiment Baseline 2 — Sector Sentiment Index**

### What I wanted
Build a sector-level sentiment index from the ticker-day sentiment scores.
I wanted each stock to have equal influence within its sector, while also keeping no-news days in the dataset 
and avoiding look-ahead bias.
### Prompt(s)
1. use the existing ticker-day sentiment scores;
2. average stock sentiment equally within each sector;
3. treat ticker-days with no news as neutral sentiment of 0;
4. keep a complete stock-by-trading-day panel;
5. map weekend and non-trading-day news to the next trading day;
6. lag sentiment by at least one trading day before it can be used for portfolio decisions;
7. save the sector sentiment index and summary outputs.
### What the assistant produced
The assistant created the standalone sector sentiment index.
Ticker-day sentiment was first completed across all stocks and trading days. No-news observations were assigned a neutral value of 0.
The stock-level sentiment values were then equally averaged within each sector to create the daily sector sentiment index.
Weekend and non-trading-day news were mapped to the next trading day, and the usable sentiment signal was lagged before portfolio use.
Main outputs included:
* sector_sentiment_index.csv
* sector_sentiment_summary.csv
* sector_sentiment_index_timeseries.png
### What was wrong or risky
If I averaged all headlines directly at sector level, stocks with more news would have a bigger influence.
There was also a risk of using same-day sentiment too early, which could cause look-ahead bias.
### What I changed and why
I first averaged news within each stock-day, then averaged stocks equally within each sector.
I also used a one-trading-day lag so the portfolio only used sentiment that was already available.

**### Sentiment Baseline 3 — Rolling Signal and Naive Fusion**

### What I wanted
Create a simple sentiment signal and test whether it could improve the Equity Minimum-Variance portfolio.
The aim was to keep the baseline method simple before developing the later innovation.
### Prompt(s)
1. use Equity Minimum Variance as the no-sentiment benchmark;
2. use lagged ticker-day sentiment;
3. convert the sentiment signal into a 60-trading-day rolling z-score;
4. use the standardised sentiment signal to tilt the existing portfolio weights;
5. test:
    * \lambda=0 for Base;
    * \lambda=+1 for Momentum;
    * \lambda=-1 for Contrarian;
6. set negative tilted weights to 0;
7. renormalise weights so they sum to 1;
8. do not tune \lambda at the baseline stage.
### What the assistant produced
The assistant created a simple sentiment-fusion framework based on the Equity Minimum-Variance portfolio.
The sentiment signal was standardised using a 60-trading-day rolling z-score.
Three versions were created:
* Base Min-Variance;
* Naive Momentum;
* Naive Contrarian.
The adjusted weights were clipped at zero where needed and then normalised back to 100%.
### What was wrong or risky
The rolling z-score could accidentally use information from the current day instead of only past data.
There was also a problem when sentiment stayed at zero for a long period, because the rolling standard deviation could become zero.
### What I changed and why
I made sure the z-score only used lagged historical sentiment.
When the rolling window had no sentiment variation, I set the signal to 0 so the portfolio would not make an unnecessary adjustment.

### **Sentiment Baseline 4 — OOS Performance Comparison**

What I wanted
Compare the simple sentiment strategies with the original Equity Minimum-Variance portfolio and see whether 
sentiment added any useful information.
Prompt(s)
1. compare Base Min-Variance, Naive Momentum and Naive Contrarian over the same out-of-sample period;
2. calculate Annualised Return, Annualised Volatility, Sharpe Ratio and Maximum Drawdown;
3. create a Growth of $1 comparison;
4. use the same holding and backtest setup for all three strategies;
5. do not include transaction costs at this baseline stage;
6. save the returns, metrics and comparison figure.
### What the assistant produced
The assistant created:
* sentiment_fusion_returns.csv
* sentiment_fusion_baseline_metrics.csv
* sentiment_fusion_growth_of_1.png
The Base Min-Variance portfolio had approximately:
* Annualised Return: 5.44%
* Annualised Volatility: 12.69%
* Sharpe Ratio: 0.429
* Maximum Drawdown: -15.29%
The Naive Contrarian strategy increased annualised return to about 6.34% and Sharpe to 0.477.
However, volatility increased to about 13.29% and maximum drawdown worsened to around -16.73%.
### What was wrong or risky
At first, the higher return and Sharpe of the Contrarian strategy looked positive, but its maximum drawdown was actually 
worse than the Base fund.
So only looking at return would give an incomplete result.
### What I changed and why
I compared return, volatility, Sharpe and maximum drawdown together instead of focusing on one metric.
I then kept the naive sentiment strategy as a baseline, because it showed some improvement but still had clear weaknesses that could be improved later.
**### Sentiment Baseline 5 — Benchmark**
### What I wanted
Use the Equity Minimum-Variance portfolio as the no-sentiment benchmark.
I wanted a simple reference point so I could later compare whether adding sentiment actually improved the portfolio.
### Prompt(s)
1. use the existing Equity Minimum-Variance portfolio;
2. keep the original portfolio construction unchanged;
3. use it as the no-sentiment benchmark;
4. do not add sentiment information at this stage;
5. keep the benchmark consistent with the later sentiment strategies.
### What the assistant produced
The assistant used the existing Equity Minimum-Variance portfolio as the baseline fund.
This gave me a clean benchmark with no sentiment adjustment, so the later Momentum and Contrarian strategies 
6. could be compared fairly against the same portfolio.
### What was wrong or risky
The main risk was changing the benchmark setup when adding sentiment later.
If the base portfolio or backtest settings changed at the same time, I would not know whether 
the performance difference came from sentiment or from another change.
### What I changed and why
I kept the original Equity Minimum-Variance fund unchanged and used it as the fixed benchmark.
This made the later comparison cleaner because sentiment was the main thing being changed.

### Innovation Attempt — Sector-Relative Signal with Fixed Tilt
### What I wanted
Improve the naive sentiment baseline by removing broad sector-wide sentiment.
The idea was to focus more on stock-specific sentiment rather than reacting to news that affected the whole sector.
### Prompt(s)
1. calculate relative sentiment as stock sentiment minus sector sentiment;
2. lag the relative signal by one trading day;
3. use the same 60-day rolling z-score framework;
4. keep the original contrarian tilt with \lambda=-1;
5. compare the result with Base Min-Variance and Naive Contrarian;
6. keep all other settings unchanged.
### What the assistant produced
The assistant created the sector-relative sentiment signal and applied it to the same portfolio framework.
The fixed \lambda=-1 Sector-Relative strategy had a Sharpe ratio of about 0.353, which was lower than 
both the Base Min-Variance and Naive Contrarian strategies.
### What was wrong or risky
The new signal itself was more targeted, but using the old fixed \lambda=-1 did not work well with it.
This showed that a better signal does not automatically mean the same tilt strength and direction will still be suitable.
### What I changed and why
I did not treat this result as the final innovation.
Instead, I kept the sector-relative signal but moved to a more disciplined process for choosing the tilt strength 
and then added risk control.
### sentiment Baseline 2.2 — Rolling Sentiment Signal
### What I wanted
Convert the ticker-day sentiment into a standardised signal that could be compared across stocks.
I also wanted the signal to use only information that was already available.
### Prompt(s)
1. use the existing ticker-day sentiment;
2. lag sentiment by one trading day;
3. calculate a 60-trading-day rolling mean and standard deviation;
4. convert sentiment into a rolling z-score;
5. use only past observations in the rolling window;
6. keep the method simple for the baseline.
### What the assistant produced
The assistant created a 60-trading-day rolling sentiment z-score for each stock.
The signal measured whether current sentiment was unusually positive or negative compared with that stock’s own recent history.
The sentiment was lagged before calculation so that current-day portfolio decisions did not use information from the same day.
### What was wrong or risky
The main risk was accidentally using current-day information in the rolling calculation.
There was also a problem when sentiment stayed unchanged for a long period, because the rolling standard deviation 
could become zero.
### What I changed and why
I made sure the rolling signal only used lagged historical sentiment.
When there was no variation in the rolling window, I treated the signal as 0 so the portfolio did not make an 
unnecessary adjustment.

### Sentiment Baseline 2.3 — Naive Weight Tilt
### What I wanted
Apply the sentiment signal directly to the existing Minimum-Variance weights using a simple tilt.
The aim was to test a basic sentiment-fusion method before doing any more advanced innovation.
### Prompt(s)
1. use the existing Min-Variance weights as base weights;
2. apply the formula
    \tilde{w}_{i,t}=w^{base}_{i,t}(1+\lambda z_{i,t})
3. test:
    * \lambda=0 for Base;
    * \lambda=+1 for Momentum;
    * \lambda=-1 for Contrarian;
4. clip negative weights to 0;
5. renormalise the remaining weights so they sum to 1;
6. do not tune \lambda in the baseline.
### What the assistant produced
The assistant created three versions of the strategy:
* Base Min-Variance;
* Naive Momentum;
* Naive Contrarian.
The sentiment signal directly adjusted the original portfolio weights.
Any negative weights created by the tilt were set to 0, and the remaining weights were rescaled to sum to 100%.
### What was wrong or risky
A key risk was choosing a \lambda value only because it gave the best result.
That would make the baseline less fair and could lead to overfitting.
There was also a risk that the tilt could create negative portfolio weights.
### What I changed and why
I kept \lambda fixed at simple values of 0, +1 and -1 and did not tune them.
I also clipped negative weights to 0 and renormalised the portfolio so the strategy remained long-only and fully invested.
⸻
### sentiment Baseline 2.4 — OOS Performance Comparison
## What I wanted
Compare the Base, Momentum and Contrarian strategies over the same out-of-sample period and check whether sentiment
actually improved the fund.
### Prompt(s)
1. compare Base Min-Variance, Naive Momentum and Naive Contrarian;
2. use the same OOS period for all strategies;
3. calculate:
    * Annualised Return;
    * Annualised Volatility;
    * Sharpe Ratio;
    * Maximum Drawdown;
4. create a Growth of $1 comparison;
5. do not include transaction costs at this baseline stage;
6. save the results clearly.
### What the assistant produced
The assistant created:
* sentiment_fusion_returns.csv
* sentiment_fusion_baseline_metrics.csv
* sentiment_fusion_growth_of_1.png
The Base Min-Variance strategy had approximately:
* Return: 5.44%
* Volatility: 12.69%
* Sharpe: 0.429
* Maximum Drawdown: -15.29%
The Naive Contrarian strategy improved return to about 6.34% and Sharpe to 0.477.
However, volatility increased to about 13.29% and maximum drawdown became worse at around -16.73%.
### What was wrong or risky
The Contrarian strategy looked better if I only focused on return and Sharpe.
But its drawdown was worse, so the improvement was not clear in every risk measure.
### What I changed and why
I compared all four performance measures together instead of only looking at return.
I kept the naive sentiment strategy as a baseline because it showed that sentiment had some value,
but it also had clear weaknesses that needed to be improved in the innovation stage.

###  1 — App Structure
### What I wanted
I wanted a clear investor journey.
I designed five main pages.
They were Home, Explore Funds, Build Portfolio, Adaptive Allocation, and Sentiment Signals.
I wanted each page to have a clear purpose.
### Prompt(s)
“Build the Streamlit app based on my five-page structure. 
Keep the same sidebar across all pages. Use my existing research results.”
### What the assistant produced
The assistant created the main navigation.
It connected the five pages.
It also added the common sidebar.
### What was wrong or risky
Explore Funds had too much information on one page.
Fund comparison and fund details had different purposes.
### What I changed and why
I split Explore Funds into Compare Funds and Fund Fact Sheet.
Compare Funds supports fund selection.
Fund Fact Sheet gives more detail about one fund.
This made the investor journey clearer.

**###  2 — Compare Funds**
### What I wanted
I wanted users to compare funds before building a portfolio.
I wanted three funds on the same page.
I wanted both numbers and charts.
### Prompt(s)
“Create a Compare Funds page for three selected model funds. 
Show return, volatility, Sharpe ratio, and max drawdown. Add Risk vs Return and Growth of $1 charts. 
Add buttons for Fact Sheet and Build Portfolio.”
### What the assistant produced
The assistant created three fund cards.
It added the main performance metrics.
It added the comparison charts.
It also added portfolio selection buttons.
### What was wrong or risky
The first layout was too spread out.
Some charts had too much empty space.
The comparison order was not clear.
### What I changed and why
I moved the fund cards to the top.
I placed the two main charts below them.
I placed the detailed comparison at the bottom.
This made the page easier to scan.
###  3 — Fund Fact Sheet
### What I wanted
I wanted a fund page based on the course example.
I did not want to copy the example design.
I wanted a PulseAlloc version.
### Prompt(s)
“Create a Fund Fact Sheet page. Keep the four main metrics. Add Growth of $1, 
Drawdown, and Current Holdings. Add fund information and interactive period controls.”
### What the assistant produced
The assistant added return, volatility, Sharpe ratio, and max drawdown.
It added Growth of $1.
It added Drawdown.
It added a holdings chart.
### What was wrong or risky
The page first looked too similar to a simple report.
It did not feel like part of the app.
### What I changed and why
I added Fund, Universe, Method, and As of information.
I added Performance Window and Top Holdings controls.
I added Key Facts.
I added a short Portfolio Insight.
This made the page more useful for investors.

**### 4 — Build Portfolio**
### What I wanted
I wanted users to build their own portfolio.
They could use funds selected from Explore Funds.
They could change the fund weights.
They could also choose a risk profile.
### Prompt(s)
“Create a Build Portfolio page using the selected model funds. Add weight controls. Add Conservative, 
Balanced, and Growth profiles. Show portfolio allocation, asset exposure, performance, and risk metrics.”
### What the assistant produced
The assistant created the fund weight controls.
It added the three risk profiles.
It added portfolio metrics.
It added allocation charts.
It also added a save function.
### What was wrong or risky
The allocation could go above 100%.
One test showed a portfolio total above 100%.
This was not valid.
### What I changed and why
I asked the assistant to fix the allocation logic.
The total allocation must stay at 100%.
I also added a remaining-allocation message.
This reduced input errors.
**###  5 — Adaptive Allocation**
### What I wanted
I wanted this page to show my main portfolio innovation.
The portfolio starts from a saved baseline.
The allocation then changes with downside risk.
The investor profile also affects the crypto budget.
### Prompt(s)
“Build an Adaptive Allocation page from my saved baseline portfolio. 
Use Investor Profile and Historical Risk Observation as inputs. Show the base crypto budget, risk score, 
downside risk, risk scaler, and adaptive crypto budget. Show the allocation change and historical evidence.”
### What the assistant produced
The assistant created the adaptive allocation controls.
It connected the saved baseline portfolio.
It showed the risk information.
It also showed fixed and adaptive performance.
### What was wrong or risky
The page became too large.
It had nine separate sections.
My main innovation was hard to see.
### What I changed and why
I reduced the page to four main modules.
I made the Adaptive Allocation Engine the main section，used the flow: Baseline → Risk Engine → Adaptive.
And added Allocation Response，and added Why Did PulseAlloc Change It?
I placed the historical evidence in one interactive section.
This made the innovation much clearer.
**###  6 — Sentiment Signals**
### What I wanted
I wanted sentiment to affect portfolio decisions.
I did not want to use raw sentiment alone.
I wanted a more useful investment signal.
### Prompt(s)
“Use the saved sentiment research in the app. 
Show sector-relative sentiment, signal strength, tuned response, tail-risk control, and portfolio action.
Add an interactive scenario section.”
### What the assistant produced
The assistant created Market Pulse and Sentiment Strategy.
It showed sentiment signals.
It showed risk information.
It also added interactive scenario controls.
### What was wrong or risky
A sentiment score alone did not explain the investment decision.
Users could not see how risk changed the final response.
### What I changed and why
I showed the full signal process.
It starts with relative sentiment.
It then uses historical signal strength.
It applies a tuned response.
It then applies the tail-risk scaler.
The final result is a portfolio action.
I also kept transaction-cost testing.
This made the strategy more realistic.
**###  7 — App Design**
### What I wanted
I wanted one visual style across the whole app.
I wanted it to look like an investment platform.
I designed the PulseAlloc colour system and page style.
### Prompt(s)
“Use the same PulseAlloc design across all pages. Keep the dark navy sidebar. Use white cards. 
Use blue as the main colour. Use green for positive or adaptive results. Keep the same spacing and typography.”
### What the assistant produced
The assistant applied the design to the pages.
It created cards, buttons, charts, and navigation styles.
### What was wrong or risky
Some pages had different spacing.
Some cards had different sizes.
Some headings did not match.
### What I changed and why
I used Home and Compare Funds as the main visual reference.
I asked the assistant to match the other pages to them.
I kept the same colours and card style.
This improved consistency.
**### 8 — Historical Data Communication**
### What I wanted
I wanted users to know that the app uses historical data.
I did not want the app to look like a live trading platform.
### Prompt(s)
“Show the latest available historical date clearly. Add Historical Research Mode where needed. 
Do not present the results as live market data.”
### What the assistant produced
The assistant added historical date labels.
It also added research-mode labels.
### What was wrong or risky
Some wording could look like a current market recommendation.
This could confuse users.
### What I changed and why
I changed the wording to historical research language.
I kept the data date visible.
I avoided direct buy or sell advice.
This made the app more transparent.
**### 9 — Testing and Code Cleanup**
### What I wanted
I wanted to check the final code.
The main file was very long.
I wanted to remove unused or repeated code.
### Prompt(s)
“Audit the final Streamlit app. Identify unused code and repeated code. 
Do not change the investment calculations or session state.”
### What the assistant produced
The assistant checked the active functions.
It found that most functions were still used.
It also found repeated CSS and UI code.
It later tried to simplify some of this code.
### What was wrong or risky
The cleanup changed the working UI.
Raw HTML appeared on the Home page.
The app no longer matched the final design.
### What I changed and why
I stopped the cleanup.
I used Rollback to restore the working version.
I checked the app again.
I kept the longer code because it was stable.
App reliability was more important than a smaller file.





## What I wanted
## Prompt(s)
## What the assistant produced
## What was wrong or risky
## What I changed and why

### **Sentiment Baseline**
## What I wanted

## Prompt(s)
## What the assistant produced
## What was wrong or risky
## What I changed and why
## Example (one filled-in entry - delete before you hand in)

### What I wanted
A function to compute simple daily returns per ticker from the equity panel.

### Prompt(s)
"Write a function that pivots equity_prices to wide adjClose and returns daily simple
returns per ticker."

### What the assistant produced
A `daily_returns()` that pivoted on `close` (not `adjClose`) and used `.diff()`
instead of `.pct_change()`.

### What was wrong or risky
Two bugs: it used raw close (ignores splits and dividends) and `.diff()` gives price
changes, not returns.

### What I changed and why
Switched to `adjClose` and `.pct_change()`, and confirmed the first row is NaN per
ticker. Checked one value by hand for AAPL on a single date.
