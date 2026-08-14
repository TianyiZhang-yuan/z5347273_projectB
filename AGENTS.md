# PulseAlloc Project Instructions
This project is FINS3645 Project B.
The product is called PulseAlloc.
PulseAlloc is an investment app for comparing model funds, reviewing fund fact sheets, 
building a portfolio, using adaptive allocation, and exploring sentiment signals.
## General Rules
- Read PROJECT_BRIEF.md before making major changes.
- Work only inside this project folder.
- Do not edit files in context/.
- Do not delete research outputs unless I clearly ask.
- Keep code changes focused on the task.
- Do not change working calculations during UI changes.
- Do not create unnecessary new files.
- Explain which files were changed after each task.
## Portfolio Construction Rules
General constraints for Max-Sharpe, Min-Variance, and pure Min-CVaR:
- Long-only: weights must be greater than or equal to zero.
- Fully invested: weights must sum to 1.
- Default single-asset cap: each asset weight is capped at 20%.
- Feasibility must be checked before every solve:
  valid_asset_count * DEFAULT_MAX_ASSET_WEIGHT must be at least 1.
- If too few valid assets make the 20% cap infeasible, the actual cap for that
  solve must be dynamically relaxed to 1 / valid_asset_count. This relaxation
  must record a warning and metadata including the valid asset count, default
  cap, adjusted cap, method, and rebalance date. Constraint relaxation must never
  happen silently.
- If a later optimiser fails because of solver or numerical problems, the future
  implementation must fall back to Equal Weight and record the failure reason,
  original method, solver status/message when available, and fallback status.
  This module currently specifies that rule only; it does not implement fallback.
### Method-specific validation:
- Min-Variance and Max-Sharpe must validate the covariance matrix condition
  number. If it exceeds COVARIANCE_CONDITION_NUMBER_THRESHOLD, future solvers
  must add COVARIANCE_RIDGE_EPSILON to the covariance diagonal and record that
  ridge regularisation was applied.
- Min-CVaR must not run covariance-matrix checks because historical CVaR does
  not depend on a covariance matrix. It must check tail scenario coverage:
  tail_count = ceil((1 - confidence_level) * valid_observation_count). If
  tail_count is below MIN_TAIL_OBSERVATIONS, future solvers must record a warning
  or refuse to solve. Future Min-CVaR code must also record linear-programming
  solver status and message.
- Equal Weight does not require covariance checks or tail scenario checks. It
  only requires valid_asset_count > 0.
Weight effective-date rule:
Weights estimated using data available through the close of date t must become
effective no earlier than date t+1. Newly estimated weights must never be applied
to the return observed on date t.
Min-CVaR version:
This project uses pure Minimum CVaR only. It minimises historical tail loss with
no expected-return constraint and no return-constrained CVaR variant.
"""
## Sentiment Rules
- Sentiment applies to equity data only.
- Preserve the original headline structure where possible.
- Lag sentiment before using it in an investment decision.
- Do not use future headlines.
- Keep the sector-neutral sentiment logic.
- Keep rolling signal standardisation.
- Keep the tuned sentiment response.
- Keep the downside tail-risk scaler.
- Keep transaction-cost testing for the final sentiment strategy.
- Do not change the final sentiment methodology unless I ask.
## Streamlit App Rules
- The app must use precomputed files from results/.
- Do not run VADER inside the deployed app.
- Do not rerun portfolio backtests inside the deployed app.
- Keep the app lightweight.
- Keep historical-data dates visible.
- Do not present historical results as live market data.
The investor journey is:
Home
→ Explore Funds
→ Compare Funds / Fund Fact Sheet
→ Build Portfolio
→ Adaptive Allocation
→ Sentiment Signals
Keep this journey unless I ask for a structural change.
## UI Rules
- Keep the PulseAlloc visual style consistent.
- Use the dark navy sidebar.
- Use white content cards.
- Use blue as the main interface colour.
- Use green for positive or adaptive results.
- Keep spacing, typography, and chart styling consistent.
- Do not redesign an approved page unless I ask.
- Do not change calculations when changing layout.
## Testing and Checking
After important code changes:
- check that streamlit_app.py compiles;
- run the app;
- test the affected page;
- test navigation;
- check that no raw HTML is visible;
- check that session state still works;
- check that portfolio weights remain valid.
For large refactors, test all main pages.
Do not delete code only because it looks unused.
First check whether it is used through callbacks, session state, dynamic keys, pandas mapping, or page routing.
If a cleanup or refactor breaks the final app, stop the change and restore the last working version.
Reliability is more important than reducing line count.
## AI Output Rules
Do not assume your first answer is correct.
For calculations:
- state the logic;
- check the result;
- flag uncertainty.
For code changes:
- summarise what changed;
- state any risk;
- report test results.
If the task is unclear, inspect the existing implementation before changing it.
Preserve working behaviour unless the requested change requires otherwise.
