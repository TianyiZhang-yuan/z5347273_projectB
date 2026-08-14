"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B."""
import warnings
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

LONG_ONLY = True
FULLY_INVESTED_WEIGHT = 1.0
DEFAULT_MAX_ASSET_WEIGHT = 0.20

COVARIANCE_CONDITION_NUMBER_THRESHOLD = 1e8
COVARIANCE_RIDGE_EPSILON = 1e-6

DEFAULT_CVAR_CONFIDENCE_LEVEL = 0.95
MIN_TAIL_OBSERVATIONS = 10
MIN_COVARIANCE_OBSERVATIONS = 2

WEIGHT_EFFECTIVE_DATE_RULE = (
    "Weights estimated using data available through the close of date t must "
    "become effective no earlier than date t+1. Newly estimated weights must "
    "never be applied to the return observed on date t."
)

MIN_CVAR_WEIGHTS_DOCSTRING = (
    "This is a pure downside-protection specification. It minimises tail loss "
    "only, with no expected-return constraint. It is not designed to maximise "
    "or match any target return."
)

PORTFOLIO_CONFIG = MappingProxyType(
    {
        "long_only": LONG_ONLY,
        "fully_invested_weight": FULLY_INVESTED_WEIGHT,
        "default_max_asset_weight": DEFAULT_MAX_ASSET_WEIGHT,
        "covariance_condition_number_threshold": (
            COVARIANCE_CONDITION_NUMBER_THRESHOLD
        ),
        "covariance_ridge_epsilon": COVARIANCE_RIDGE_EPSILON,
        "default_cvar_confidence_level": DEFAULT_CVAR_CONFIDENCE_LEVEL,
        "min_tail_observations": MIN_TAIL_OBSERVATIONS,
        "min_covariance_observations": MIN_COVARIANCE_OBSERVATIONS,
        "weight_effective_date_rule": WEIGHT_EFFECTIVE_DATE_RULE,
        "min_cvar_weights_docstring": MIN_CVAR_WEIGHTS_DOCSTRING,
        "methods": MappingProxyType(
            {
                "equal_weight": MappingProxyType(
                    {
                        "requires_covariance_check": False,
                        "requires_tail_check": False,
                        "requires_positive_asset_count": True,
                    }
                ),
                "min_variance": MappingProxyType(
                    {
                        "requires_long_only_cap_feasibility": True,
                        "requires_covariance_check": True,
                        "requires_tail_check": False,
                    }
                ),
                "max_sharpe": MappingProxyType(
                    {
                        "requires_long_only_cap_feasibility": True,
                        "requires_covariance_check": True,
                        "requires_tail_check": False,
                    }
                ),
                "min_cvar": MappingProxyType(
                    {
                        "requires_long_only_cap_feasibility": True,
                        "requires_covariance_check": False,
                        "requires_tail_check": True,
                        "requires_lp_solver_status_message": True,
                        "pure_downside_protection_only": True,
                    }
                ),
            }
        ),
    }
)

# ============================================================================
# STAGE 7.1: UNIFIED MIN-CVAR RISK-CONTROL FOUNDATION
# Purpose:
# Provide one reusable Pure 95% Minimum-CVaR optimisation foundation for the
# six Combined stock–Crypto innovation funds. The optimiser retains the common
# baseline constraints and optionally accepts an aggregate Crypto allocation
# cap. Adaptive risk scoring and personalised cap rules are implemented in
# later Stage 7 sections.
# ============================================================================
STAGE_7_COMMON_SETTINGS = MappingProxyType(
    {
        "portfolio_method": "min_cvar",
        "cvar_confidence_level": DEFAULT_CVAR_CONFIDENCE_LEVEL,
        "pure_min_cvar": True,
        "estimation_window_observations": 252,
        "rebalance_frequency": "monthly",
        "long_only": LONG_ONLY,
        "fully_invested_weight": FULLY_INVESTED_WEIGHT,
        "single_asset_max_weight": DEFAULT_MAX_ASSET_WEIGHT,
        "single_asset_cap_policy": (
            "use the existing single-asset cap and relax it only when the "
            "long-only full-investment constraint is otherwise infeasible"
        ),
        "solver": "scipy.optimize.linprog(method='highs')",
        "fallback_policy": "existing capped equal-weight fallback on optimiser failure",
        "weight_effective_date_rule": WEIGHT_EFFECTIVE_DATE_RULE,
        "universe": "combined_equity_plus_crypto",
        "asset_count_equity": 50,
        "asset_count_crypto": 10,
        "asset_count_total": 60,
    }
)

# Min-CVaR is the common portfolio method, not the main product innovation.
# Later stages determine each personalised Crypto cap; this optimiser chooses
# the individual 50 Equity and 10 Crypto weights within that upper bound.

### stage 1
_WEIGHT_SUM_TOLERANCE = 1e-6
_WEIGHT_BOUND_TOLERANCE = 1e-8
_NEAR_ZERO_VOLATILITY = 1e-12


def _base_metadata(method: str) -> dict:
    """Create the shared metadata schema used by every weight solver."""
    return {
        "method": method,
        "requested_method": method,
        "optimisation_method": method,
        "solver_success": None,
        "solver_status": None,
        "solver_message": None,
        "fallback_used": False,
        "fallback_from_method": None,
        "fallback_reason": None,
        "valid_asset_count": None,
        "valid_observation_count": None,
        "complete_case_observation_count": None,
        "complete_case_rule": None,
        "dropped_assets_all_missing": [],
        "non_numeric_columns": [],
        "default_max_asset_weight": DEFAULT_MAX_ASSET_WEIGHT,
        "effective_max_asset_weight": None,
        "cap_feasible_at_default": None,
        "cap_relaxed": False,
        "cap_relaxation_warning": None,
        "min_covariance_observations": None,
        "covariance_condition_number": None,
        "covariance_condition_number_before_ridge": None,
        "covariance_condition_number_after_ridge": None,
        "ridge_regularisation_used": False,
        "ridge_epsilon": None,
        "cvar_confidence_level": None,
        "cvar_tail_count": None,
        "cvar_tail_warning": None,
        "pure_min_cvar": method == "min_cvar",
        "crypto_assets": None,
        "crypto_cap": None,
        "crypto_cap_active": False,
        "requested_crypto_cap": None,
        "crypto_cap_constraint_active": False,
        "actual_crypto_weight": None,
        "actual_equity_weight": None,
        "crypto_cap_binding": None,
        "lp_solver_status": None,
        "lp_solver_message": None,
        "single_asset_cap": None,
        "single_asset_cap_relaxed": False,
        "weights_sum": None,
        "minimum_weight": None,
        "maximum_weight": None,
        "weights_valid": None,
        "weights_validation_message": None,
        "diagnostic_warnings": [],
    }


def _validate_returns_input(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a non-empty wide return matrix."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame")
    return returns.copy()


def _clean_returns(returns: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Validate numeric returns, replace infinities, and remove all-missing assets."""
    clean = _validate_returns_input(returns)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    all_missing = clean.columns[clean.notna().sum(axis=0).eq(0)].tolist()
    clean = clean.drop(columns=all_missing)
    if clean.shape[1] == 0:
        raise ValueError("returns has no asset columns with usable observations")

    non_numeric_columns = [
        column
        for column in clean.columns
        if not pd.api.types.is_numeric_dtype(clean[column])
    ]
    metadata["non_numeric_columns"] = non_numeric_columns
    if non_numeric_columns:
        raise TypeError(
            "all asset return columns must be numeric; "
            f"non-numeric columns: {non_numeric_columns}"
        )

    metadata["dropped_assets_all_missing"] = all_missing
    metadata["valid_asset_count"] = int(clean.shape[1])
    metadata["valid_observation_count"] = int(clean.dropna(how="all").shape[0])
    return clean


def _complete_case_returns(clean_returns: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Use one complete-case matrix for all optimised portfolio methods."""
    complete_returns = clean_returns.dropna(how="any")
    metadata["complete_case_rule"] = (
        "drop any historical row with a missing return for any retained asset"
    )
    metadata["complete_case_observation_count"] = len(complete_returns)
    metadata["valid_observation_count"] = len(complete_returns)
    return complete_returns


def _effective_asset_cap(
    valid_asset_count: int,
    metadata: dict,
    method: str,
    apply_cap: bool,
    rebalance_date=None,
):
    """Return the effective cap, relaxing it only when Stage 0 requires it."""
    if not apply_cap:
        metadata["cap_feasible_at_default"] = None
        metadata["effective_max_asset_weight"] = None
        metadata["single_asset_cap"] = None
        metadata["single_asset_cap_relaxed"] = False
        return None

    cap_feasible = valid_asset_count * DEFAULT_MAX_ASSET_WEIGHT >= FULLY_INVESTED_WEIGHT
    metadata["cap_feasible_at_default"] = bool(cap_feasible)
    if cap_feasible:
        metadata["effective_max_asset_weight"] = DEFAULT_MAX_ASSET_WEIGHT
        metadata["single_asset_cap"] = DEFAULT_MAX_ASSET_WEIGHT
        metadata["single_asset_cap_relaxed"] = False
        return DEFAULT_MAX_ASSET_WEIGHT

    adjusted_cap = FULLY_INVESTED_WEIGHT / valid_asset_count
    warning_message = (
        f"{method}: default max asset weight {DEFAULT_MAX_ASSET_WEIGHT:.2%} is "
        f"infeasible with {valid_asset_count} valid assets; relaxing effective "
        f"cap to {adjusted_cap:.6f} for this solve."
    )
    metadata["effective_max_asset_weight"] = adjusted_cap
    metadata["cap_relaxed"] = True
    metadata["single_asset_cap"] = adjusted_cap
    metadata["single_asset_cap_relaxed"] = True
    metadata["cap_relaxation_warning"] = warning_message
    metadata["cap_relaxation_metadata"] = {
        "method": method,
        "rebalance_date": rebalance_date,
        "valid_asset_count": valid_asset_count,
        "default_max_asset_weight": DEFAULT_MAX_ASSET_WEIGHT,
        "effective_max_asset_weight": adjusted_cap,
    }
    metadata["diagnostic_warnings"].append(warning_message)
    warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
    return adjusted_cap


def _validate_crypto_cap_value(crypto_cap, name: str = "crypto_cap") -> float:
    """Validate a fixed or dynamically supplied total crypto allocation cap."""
    if not isinstance(crypto_cap, int | float | np.number) or not np.isfinite(crypto_cap):
        raise ValueError(f"{name} must be a finite numeric value")
    crypto_cap = float(crypto_cap)
    if not 0.0 <= crypto_cap <= 1.0:
        raise ValueError(f"{name} must be in the closed interval [0, 1]")
    return crypto_cap


def _normalise_crypto_assets(columns, crypto_assets, crypto_cap):
    """Validate explicit crypto asset labels and return their positional indexes."""
    if crypto_cap is None:
        return [], []
    if crypto_assets is None:
        raise ValueError("crypto_assets must be provided when a crypto cap is active")

    assets = list(crypto_assets)
    if not assets:
        raise ValueError("crypto_assets must contain at least one asset label")
    if len(set(assets)) != len(assets):
        raise ValueError("crypto_assets must not contain duplicate labels")

    columns = pd.Index(columns)
    missing = sorted(set(assets).difference(columns))
    if missing:
        raise ValueError(f"crypto_assets not found in returns columns: {missing}")
    return assets, [columns.get_loc(asset) for asset in assets]


def _record_crypto_cap(metadata: dict, crypto_assets, crypto_cap):
    """Store crypto cap diagnostics in solver metadata."""
    metadata["crypto_cap"] = crypto_cap
    metadata["crypto_cap_active"] = crypto_cap is not None
    metadata["requested_crypto_cap"] = crypto_cap
    metadata["crypto_cap_constraint_active"] = crypto_cap is not None
    metadata["crypto_assets"] = list(crypto_assets) if crypto_assets is not None else None


def _crypto_cap_constraint(columns, crypto_assets, crypto_cap):
    """Build a scipy inequality constraint for the total crypto weight cap."""
    assets, positions = _normalise_crypto_assets(columns, crypto_assets, crypto_cap)
    if crypto_cap is None:
        return None

    def constraint(weights):
        return crypto_cap - float(np.sum(weights[positions]))

    return {"type": "ineq", "fun": constraint}, assets, positions


def asset_class_weights(weights: pd.Series, crypto_assets) -> dict[str, float]:
    """Return actual total Crypto and Equity weights using explicit Crypto labels."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if weights.index.has_duplicates:
        raise ValueError("weights index must not contain duplicate asset labels")
    if crypto_assets is None:
        raise ValueError("crypto_assets must be supplied explicitly")
    crypto_list = list(crypto_assets)
    if len(set(crypto_list)) != len(crypto_list):
        raise ValueError("crypto_assets must not contain duplicate labels")

    numeric_weights = weights.astype(float)
    if not np.isfinite(numeric_weights.to_numpy(dtype=float)).all():
        raise ValueError("weights must contain only finite values")

    crypto_index = numeric_weights.index.intersection(pd.Index(crypto_list))
    actual_crypto_weight = float(numeric_weights.loc[crypto_index].sum())
    actual_equity_weight = float(numeric_weights.sum() - actual_crypto_weight)
    return {
        "actual_crypto_weight": actual_crypto_weight,
        "actual_equity_weight": actual_equity_weight,
    }


def _record_asset_class_metadata(
    metadata: dict,
    weights: pd.Series,
    crypto_assets,
    crypto_cap,
) -> None:
    """Record actual asset-class allocations when explicit Crypto labels exist."""
    values = weights.to_numpy(dtype=float)
    metadata["weights_sum"] = float(values.sum())
    metadata["minimum_weight"] = float(values.min())
    metadata["maximum_weight"] = float(values.max())

    if crypto_assets is None:
        metadata["actual_crypto_weight"] = None
        metadata["actual_equity_weight"] = None
        metadata["crypto_cap_binding"] = None
        return

    allocations = asset_class_weights(weights, crypto_assets)
    metadata.update(allocations)
    if crypto_cap is None:
        metadata["crypto_cap_binding"] = None
    else:
        metadata["crypto_cap_binding"] = bool(
            np.isclose(
                allocations["actual_crypto_weight"],
                crypto_cap,
                atol=_WEIGHT_SUM_TOLERANCE,
            )
        )


def _equal_weights_for_columns(
    columns,
    crypto_assets=None,
    crypto_cap=None,
) -> pd.Series:
    """Create equal weights indexed by the supplied asset columns."""
    n_assets = len(columns)
    if n_assets <= 0:
        raise ValueError("equal weights require at least one valid asset")
    if crypto_cap is not None:
        crypto_assets, _positions = _normalise_crypto_assets(columns, crypto_assets, crypto_cap)
        crypto_set = set(crypto_assets)
        crypto_columns = [column for column in columns if column in crypto_set]
        noncrypto_columns = [column for column in columns if column not in crypto_set]
        uncapped_crypto_weight = len(crypto_columns) / n_assets
        if uncapped_crypto_weight <= crypto_cap + _WEIGHT_BOUND_TOLERANCE:
            return pd.Series(
                np.repeat(FULLY_INVESTED_WEIGHT / n_assets, n_assets),
                index=pd.Index(columns),
                name="weight",
            )
        if crypto_cap < FULLY_INVESTED_WEIGHT and not noncrypto_columns:
            raise ValueError("crypto cap is infeasible because all valid assets are crypto")

        weights = pd.Series(0.0, index=pd.Index(columns), name="weight")
        if crypto_columns:
            weights.loc[crypto_columns] = crypto_cap / len(crypto_columns)
        if noncrypto_columns:
            weights.loc[noncrypto_columns] = (
                (FULLY_INVESTED_WEIGHT - crypto_cap) / len(noncrypto_columns)
            )
        return weights
    return pd.Series(
        np.repeat(FULLY_INVESTED_WEIGHT / n_assets, n_assets),
        index=pd.Index(columns),
        name="weight",
    )


def _validate_final_weights(
    weights: pd.Series,
    effective_cap: float | None,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[bool, str]:
    """Validate final portfolio weights against shared constraints."""
    if not isinstance(weights, pd.Series):
        return False, "weights must be a pandas Series"
    if weights.empty:
        return False, "weights must not be empty"
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False, "weights contain NaN or infinity"
    if (values < -_WEIGHT_BOUND_TOLERANCE).any():
        return False, "weights contain negative values"
    if not np.isclose(
        values.sum(),
        FULLY_INVESTED_WEIGHT,
        atol=_WEIGHT_SUM_TOLERANCE,
    ):
        return False, f"weights sum to {values.sum()}, not 1"
    if effective_cap is not None and (values > effective_cap + _WEIGHT_BOUND_TOLERANCE).any():
        return False, "weights exceed the effective asset cap"
    if crypto_cap is not None:
        crypto_assets, _positions = _normalise_crypto_assets(
            weights.index,
            crypto_assets,
            crypto_cap,
        )
        crypto_weight = float(weights.loc[crypto_assets].sum())
        if crypto_weight > crypto_cap + _WEIGHT_BOUND_TOLERANCE:
            return False, "weights exceed the total crypto allocation cap"
    return True, "valid"


def _finalise_weights(
    weights: pd.Series,
    metadata: dict,
    effective_cap: float | None,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Store final validation diagnostics and return the standard output tuple."""
    weights = weights.astype(float)
    valid, message = _validate_final_weights(
        weights,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    metadata["weights_valid"] = valid
    metadata["weights_validation_message"] = message
    if not valid:
        raise ValueError(message)
    _record_asset_class_metadata(
        metadata,
        weights,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    weights.name = "weight"
    return weights, metadata


def _fallback_equal_weight(
    clean_returns: pd.DataFrame,
    metadata: dict,
    reason: str,
    effective_cap: float | None,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Use Equal Weight after an optimiser failure and record why."""
    metadata["fallback_used"] = True
    metadata["fallback_from_method"] = metadata["requested_method"]
    metadata["fallback_reason"] = reason
    metadata["solver_success"] = False
    if metadata["solver_status"] is None:
        metadata["solver_status"] = "fallback"
    if metadata["solver_message"] is None:
        metadata["solver_message"] = reason
    metadata["diagnostic_warnings"].append(f"fallback_to_equal_weight: {reason}")
    weights = _equal_weights_for_columns(
        clean_returns.columns,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    return _finalise_weights(
        weights,
        metadata,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )


def _covariance_with_regularisation(
    complete_returns: pd.DataFrame,
    metadata: dict,
) -> np.ndarray | None:
    """Estimate covariance and apply ridge regularisation when required."""
    metadata["min_covariance_observations"] = MIN_COVARIANCE_OBSERVATIONS
    if len(complete_returns) < MIN_COVARIANCE_OBSERVATIONS:
        metadata["solver_message"] = (
            "insufficient complete observations for covariance estimation: "
            f"{len(complete_returns)} < {MIN_COVARIANCE_OBSERVATIONS}"
        )
        return None

    covariance = complete_returns.cov().to_numpy(dtype=float)
    if covariance.shape[0] == 0 or not np.isfinite(covariance).all():
        metadata["solver_message"] = "covariance matrix contains NaN or infinity"
        return None

    condition_number = float(np.linalg.cond(covariance))
    metadata["covariance_condition_number"] = condition_number
    metadata["covariance_condition_number_before_ridge"] = condition_number
    if (
        not np.isfinite(condition_number)
        or condition_number > COVARIANCE_CONDITION_NUMBER_THRESHOLD
    ):
        covariance = covariance.copy()
        covariance += np.eye(covariance.shape[0]) * COVARIANCE_RIDGE_EPSILON
        metadata["ridge_regularisation_used"] = True
        metadata["ridge_epsilon"] = COVARIANCE_RIDGE_EPSILON
        metadata["covariance_condition_number_after_ridge"] = float(
            np.linalg.cond(covariance)
        )
    else:
        metadata["covariance_condition_number_after_ridge"] = condition_number
    return covariance


def _optimiser_bounds(n_assets: int, effective_cap: float):
    """Build long-only capped bounds for scipy optimisers."""
    return [(0.0, effective_cap) for _ in range(n_assets)]


def _sum_to_one_constraint() -> dict:
    """Build the fully invested equality constraint for scipy minimizers."""
    return {"type": "eq", "fun": lambda weights: np.sum(weights) - FULLY_INVESTED_WEIGHT}


def equal_weight_weights(
    returns: pd.DataFrame,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Return the transparent Equal Weight baseline."""
    method = "equal_weight"
    if crypto_cap is not None:
        crypto_cap = _validate_crypto_cap_value(crypto_cap)
    metadata = _base_metadata(method)
    clean_returns = _clean_returns(returns, metadata)
    _record_crypto_cap(metadata, crypto_assets, crypto_cap)
    _effective_asset_cap(
        metadata["valid_asset_count"],
        metadata,
        method=method,
        apply_cap=False,
    )
    weights = _equal_weights_for_columns(
        clean_returns.columns,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    metadata["solver_success"] = True
    metadata["solver_status"] = 0
    metadata["solver_message"] = "equal weights assigned; no optimiser used"
    return _finalise_weights(
        weights,
        metadata,
        effective_cap=None,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )


def min_variance_weights(
    returns: pd.DataFrame,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Return long-only, fully invested Minimum Variance weights."""
    method = "min_variance"
    if crypto_cap is not None:
        crypto_cap = _validate_crypto_cap_value(crypto_cap)
    metadata = _base_metadata(method)
    clean_returns = _clean_returns(returns, metadata)
    _record_crypto_cap(metadata, crypto_assets, crypto_cap)
    complete_returns = _complete_case_returns(clean_returns, metadata)
    effective_cap = _effective_asset_cap(
        metadata["valid_asset_count"],
        metadata,
        method=method,
        apply_cap=True,
    )
    covariance = _covariance_with_regularisation(complete_returns, metadata)
    if covariance is None:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            "cannot optimise with invalid covariance matrix",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    n_assets = clean_returns.shape[1]
    initial_weights = _equal_weights_for_columns(
        clean_returns.columns,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    ).to_numpy()

    def objective(weights):
        return float(weights @ covariance @ weights)

    constraints = [_sum_to_one_constraint()]
    crypto_constraint = _crypto_cap_constraint(clean_returns.columns, crypto_assets, crypto_cap)
    if crypto_constraint is not None:
        constraints.append(crypto_constraint[0])

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=_optimiser_bounds(n_assets, effective_cap),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    metadata["solver_success"] = bool(result.success)
    metadata["solver_status"] = int(result.status)
    metadata["solver_message"] = str(result.message)
    if not result.success:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"min_variance optimiser failed: {result.message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    weights = pd.Series(result.x, index=clean_returns.columns, name="weight")
    valid, message = _validate_final_weights(
        weights,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    if not valid:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"min_variance optimiser produced invalid weights: {message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )
    return _finalise_weights(
        weights,
        metadata,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )


def max_sharpe_weights(
    returns: pd.DataFrame,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Return long-only, fully invested Maximum Sharpe weights."""
    method = "max_sharpe"
    if crypto_cap is not None:
        crypto_cap = _validate_crypto_cap_value(crypto_cap)
    metadata = _base_metadata(method)
    clean_returns = _clean_returns(returns, metadata)
    _record_crypto_cap(metadata, crypto_assets, crypto_cap)
    complete_returns = _complete_case_returns(clean_returns, metadata)
    effective_cap = _effective_asset_cap(
        metadata["valid_asset_count"],
        metadata,
        method=method,
        apply_cap=True,
    )
    covariance = _covariance_with_regularisation(complete_returns, metadata)
    mean_returns = complete_returns.mean(skipna=True).to_numpy(dtype=float)
    if covariance is None or not np.isfinite(mean_returns).all():
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            "cannot optimise with invalid covariance matrix or mean returns",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    n_assets = clean_returns.shape[1]
    initial_weights = _equal_weights_for_columns(
        clean_returns.columns,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    ).to_numpy()

    def objective(weights):
        portfolio_return = float(weights @ mean_returns)
        variance = float(weights @ covariance @ weights)
        volatility = np.sqrt(max(variance, 0.0))
        if volatility <= _NEAR_ZERO_VOLATILITY:
            return 1e12
        return -portfolio_return / volatility

    constraints = [_sum_to_one_constraint()]
    crypto_constraint = _crypto_cap_constraint(clean_returns.columns, crypto_assets, crypto_cap)
    if crypto_constraint is not None:
        constraints.append(crypto_constraint[0])

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=_optimiser_bounds(n_assets, effective_cap),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    metadata["solver_success"] = bool(result.success)
    metadata["solver_status"] = int(result.status)
    metadata["solver_message"] = str(result.message)
    if not result.success:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"max_sharpe optimiser failed: {result.message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    weights = pd.Series(result.x, index=clean_returns.columns, name="weight")
    valid, message = _validate_final_weights(
        weights,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    if not valid:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"max_sharpe optimiser produced invalid weights: {message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )
    return _finalise_weights(
        weights,
        metadata,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )


def min_cvar_weights(
    returns: pd.DataFrame,
    confidence_level: float = DEFAULT_CVAR_CONFIDENCE_LEVEL,
    crypto_assets=None,
    crypto_cap=None,
) -> tuple[pd.Series, dict]:
    """Return pure historical Minimum CVaR weights.

    This is a pure downside-protection specification. It minimises tail loss
    only, with no expected-return constraint. It is not designed to maximise
    or match any target return.
    """
    method = "min_cvar"
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1")
    if crypto_cap is not None:
        crypto_cap = _validate_crypto_cap_value(crypto_cap)

    metadata = _base_metadata(method)
    metadata["cvar_confidence_level"] = confidence_level
    clean_returns = _clean_returns(returns, metadata)
    _record_crypto_cap(metadata, crypto_assets, crypto_cap)
    complete_returns = _complete_case_returns(clean_returns, metadata)
    effective_cap = _effective_asset_cap(
        metadata["valid_asset_count"],
        metadata,
        method=method,
        apply_cap=True,
    )

    valid_observation_count = len(complete_returns)
    if valid_observation_count == 0:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            "min_cvar has no complete return observations",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    tail_count = int(np.ceil((1 - confidence_level) * valid_observation_count))
    metadata["cvar_tail_count"] = tail_count
    if tail_count < MIN_TAIL_OBSERVATIONS:
        warning_message = (
            f"min_cvar tail sample is small: tail_count={tail_count}, "
            f"minimum={MIN_TAIL_OBSERVATIONS}"
        )
        metadata["cvar_tail_warning"] = warning_message
        metadata["diagnostic_warnings"].append(warning_message)
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    return_matrix = complete_returns.to_numpy(dtype=float)
    if not np.isfinite(return_matrix).all():
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            "min_cvar complete return matrix contains NaN or infinity",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    n_obs, n_assets = return_matrix.shape
    n_variables = n_assets + 1 + n_obs
    alpha_index = n_assets
    u_start = n_assets + 1

    objective = np.zeros(n_variables)
    objective[alpha_index] = 1.0
    objective[u_start:] = 1.0 / ((1 - confidence_level) * n_obs)

    a_ub = np.zeros((n_obs, n_variables))
    a_ub[:, :n_assets] = -return_matrix
    a_ub[:, alpha_index] = -1.0
    a_ub[np.arange(n_obs), u_start + np.arange(n_obs)] = -1.0
    b_ub = np.zeros(n_obs)

    a_eq = np.zeros((1, n_variables))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([FULLY_INVESTED_WEIGHT])

    crypto_assets, crypto_positions = _normalise_crypto_assets(
        clean_returns.columns,
        crypto_assets,
        crypto_cap,
    )
    if crypto_cap is not None:
        crypto_row = np.zeros((1, n_variables))
        crypto_row[0, crypto_positions] = 1.0
        a_ub = np.vstack([a_ub, crypto_row])
        b_ub = np.append(b_ub, crypto_cap)

    bounds = (
        [(0.0, effective_cap) for _ in range(n_assets)]
        + [(None, None)]
        + [(0.0, None) for _ in range(n_obs)]
    )

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    metadata["solver_success"] = bool(result.success)
    metadata["solver_status"] = int(result.status)
    metadata["solver_message"] = str(result.message)
    metadata["lp_solver_status"] = int(result.status)
    metadata["lp_solver_message"] = str(result.message)
    if not result.success:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"min_cvar LP solver failed: {result.message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )

    weights = pd.Series(result.x[:n_assets], index=clean_returns.columns, name="weight")
    valid, message = _validate_final_weights(
        weights,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )
    if not valid:
        return _fallback_equal_weight(
            clean_returns,
            metadata,
            f"min_cvar LP produced invalid weights: {message}",
            effective_cap,
            crypto_assets=crypto_assets,
            crypto_cap=crypto_cap,
        )
    return _finalise_weights(
        weights,
        metadata,
        effective_cap,
        crypto_assets=crypto_assets,
        crypto_cap=crypto_cap,
    )

# ============================================================================
# STAGE 2: WALK-FORWARD OUT-OF-SAMPLE BACKTESTING
# ============================================================================


def _validate_backtest_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate the wide return panel required by the walk-forward backtest."""
    if not returns.columns.is_unique:
        raise ValueError("returns columns must contain unique asset labels")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing date order")
    if returns.index.has_duplicates:
        raise ValueError("returns index must contain unique dates")
    return _clean_returns(returns, _base_metadata("oos_backtest"))


def _solver_for_method(method: str):
    """Return the Stage 1 solver for a supported method name."""
    solvers = {
        "equal_weight": equal_weight_weights,
        "min_variance": min_variance_weights,
        "max_sharpe": max_sharpe_weights,
        "min_cvar": min_cvar_weights,
    }
    if method not in solvers:
        raise ValueError(f"unsupported portfolio method: {method}")
    return solvers[method]


def oos_backtest(
    returns: pd.DataFrame,
    method: str,
    window_size: int,
    periods_per_year: int,
    confidence_level: float = DEFAULT_CVAR_CONFIDENCE_LEVEL,
    risk_free_rate: float = 0.0,
    crypto_assets=None,
    crypto_cap=None,
    crypto_cap_rule=None,
    start_decision_date=None,
    end_decision_date=None,
) -> dict:
    """Run a fixed-window monthly walk-forward portfolio backtest.

    Missing live returns for currently held assets raise a clear error rather
    than being filled with zero.
    """
    del risk_free_rate  # Reserved for future method variants; solvers use rf=0.
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if crypto_cap is not None and crypto_cap_rule is not None:
        raise ValueError("crypto_cap and crypto_cap_rule cannot both be supplied")
    if (crypto_cap is not None or crypto_cap_rule is not None) and crypto_assets is None:
        raise ValueError("crypto_assets must be supplied when a crypto cap is active")
    crypto_cap_active = crypto_cap is not None or crypto_cap_rule is not None
    if crypto_cap is not None:
        crypto_cap = _validate_crypto_cap_value(crypto_cap)

    clean_returns = _validate_backtest_returns(returns)
    if crypto_cap_active:
        crypto_assets, _crypto_positions = _normalise_crypto_assets(
            clean_returns.columns,
            crypto_assets,
            1.0,
        )
    if len(clean_returns) <= window_size:
        raise ValueError("returns must contain more rows than window_size")

    solver = _solver_for_method(method)
    dates = clean_returns.index
    assets = clean_returns.columns
    month_end_positions = (
        pd.Series(range(len(dates)), index=dates)
        .groupby(dates.to_period("M"))
        .max()
        .astype(int)
    )
    if start_decision_date is not None:
        start_decision_date = pd.Timestamp(start_decision_date)
    if end_decision_date is not None:
        end_decision_date = pd.Timestamp(end_decision_date)
    rebalance_positions = []
    for position in month_end_positions:
        decision_date = dates[position]
        if position < window_size - 1 or position + 1 >= len(clean_returns):
            continue
        if start_decision_date is not None and decision_date < start_decision_date:
            continue
        if end_decision_date is not None and decision_date > end_decision_date:
            continue
        rebalance_positions.append(position)
    if not rebalance_positions:
        raise ValueError("no rebalance dates available after the initial window")

    effective_to_estimation = {
        position + 1: position
        for position in rebalance_positions
    }
    first_live_position = min(effective_to_estimation)

    current_weights = None
    daily_returns = {}
    target_rows = {}
    pretrade_rows = {}
    turnover_values = {}
    rebalance_records = []
    previous_crypto_cap = None

    for position in range(first_live_position, len(clean_returns)):
        date = dates[position]
        if position in effective_to_estimation:
            estimation_position = effective_to_estimation[position]
            estimation_end_date = dates[estimation_position]
            estimation_start_date = dates[estimation_position - window_size + 1]
            assert estimation_end_date < date, WEIGHT_EFFECTIVE_DATE_RULE

            window = clean_returns.iloc[
                estimation_position - window_size + 1: estimation_position + 1
            ]
            if crypto_cap_rule is not None:
                current_crypto_cap = _validate_crypto_cap_value(
                    crypto_cap_rule(
                        estimation_returns=window,
                        rebalance_date=date,
                        previous_cap=previous_crypto_cap,
                    ),
                    name="crypto_cap_rule result",
                )
            else:
                current_crypto_cap = crypto_cap

            if method == "min_cvar":
                solved_weights, metadata = solver(
                    window,
                    confidence_level=confidence_level,
                    crypto_assets=crypto_assets,
                    crypto_cap=current_crypto_cap,
                )
            else:
                solved_weights, metadata = solver(
                    window,
                    crypto_assets=crypto_assets,
                    crypto_cap=current_crypto_cap,
                )

            target = solved_weights.reindex(assets).fillna(0.0)
            valid, message = _validate_final_weights(
                target,
                metadata["effective_max_asset_weight"],
                crypto_assets=crypto_assets,
                crypto_cap=current_crypto_cap,
            )
            if not valid:
                raise ValueError(f"invalid target weights at {date.date()}: {message}")
            excluded_assets = target.index[target.eq(0.0)].tolist()
            pretrade = current_weights.copy() if current_weights is not None else target.copy()
            turnover = (
                np.nan
                if current_weights is None
                else 0.5 * float((target - pretrade).abs().sum())
            )

            target_rows[date] = target
            pretrade_rows[date] = pretrade
            turnover_values[date] = turnover
            rebalance_records.append(
                {
                    "estimation_start_date": estimation_start_date,
                    "estimation_end_date": estimation_end_date,
                    "effective_date": date,
                    "method": method,
                    "solver_success": metadata["solver_success"],
                    "fallback_used": metadata["fallback_used"],
                    "effective_max_asset_weight": metadata["effective_max_asset_weight"],
                    "excluded_assets": excluded_assets,
                    "turnover": turnover,
                    "crypto_cap": current_crypto_cap,
                }
            )
            if current_crypto_cap is not None:
                rebalance_records[-1].update(
                    {
                        "optimisation_method": metadata["optimisation_method"],
                        "cvar_confidence_level": metadata["cvar_confidence_level"],
                        "pure_min_cvar": metadata["pure_min_cvar"],
                        "requested_crypto_cap": metadata["requested_crypto_cap"],
                        "crypto_cap_constraint_active": metadata[
                            "crypto_cap_constraint_active"
                        ],
                        "actual_crypto_weight": metadata["actual_crypto_weight"],
                        "actual_equity_weight": metadata["actual_equity_weight"],
                        "crypto_cap_binding": metadata["crypto_cap_binding"],
                        "solver_status": metadata["solver_status"],
                        "single_asset_cap": metadata["single_asset_cap"],
                        "single_asset_cap_relaxed": metadata[
                            "single_asset_cap_relaxed"
                        ],
                        "weights_sum": metadata["weights_sum"],
                        "minimum_weight": metadata["minimum_weight"],
                        "maximum_weight": metadata["maximum_weight"],
                    }
                )
            current_weights = target
            previous_crypto_cap = current_crypto_cap

        held_assets = current_weights.index[current_weights.gt(_WEIGHT_BOUND_TOLERANCE)]
        day_returns = clean_returns.loc[date, held_assets]
        if day_returns.isna().any():
            missing_assets = day_returns.index[day_returns.isna()].tolist()
            raise ValueError(
                f"missing live returns on {date.date()} for held assets: {missing_assets}"
            )

        portfolio_return = float((day_returns * current_weights.loc[held_assets]).sum())
        daily_returns[date] = portfolio_return
        gross_return = 1.0 + portfolio_return
        if gross_return <= 0:
            raise ValueError(f"portfolio gross return is non-positive on {date.date()}")
        drifted = current_weights.copy()
        drifted.loc[held_assets] = (
            current_weights.loc[held_assets] * (1.0 + day_returns) / gross_return
        )
        current_weights = drifted / drifted.sum()

    return {
        "daily_returns": pd.Series(daily_returns, name="portfolio_return"),
        "target_weights": pd.DataFrame.from_dict(target_rows, orient="index")
        .sort_index()
        .rename_axis("date"),
        "pretrade_weights": pd.DataFrame.from_dict(pretrade_rows, orient="index")
        .sort_index()
        .rename_axis("date"),
        "turnover": pd.Series(turnover_values, name="turnover").sort_index(),
        "rebalance_log": pd.DataFrame(rebalance_records),
        "first_live_date": dates[first_live_position],
        "method": method,
        "window_size": window_size,
        "periods_per_year": periods_per_year,
    }


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Calculate compact performance metrics from finite daily returns."""
    if not isinstance(daily_returns, pd.Series):
        raise TypeError("daily_returns must be a pandas Series")
    if daily_returns.empty:
        raise ValueError("daily_returns must not be empty")
    returns = daily_returns.astype(float)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("daily_returns must contain only finite values")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")

    growth = (1.0 + returns).cumprod()
    observation_count = len(returns)
    final_growth = float(growth.iloc[-1])
    annualised_return = final_growth ** (periods_per_year / observation_count) - 1.0
    annualised_volatility = (
        float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        if observation_count > 1
        else np.nan
    )
    sharpe_ratio = (
        (annualised_return - risk_free_rate) / annualised_volatility
        if annualised_volatility and np.isfinite(annualised_volatility)
        else np.nan
    )
    drawdown = growth / growth.cummax() - 1.0
    return {
        "observation_count": observation_count,
        "start_date": returns.index.min(),
        "end_date": returns.index.max(),
        "annualised_return": float(annualised_return),
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": float(drawdown.min()),
        "final_growth_of_1": final_growth,
    }


# ============================================================================
# STAGE 7.2: CRYPTO MARKET ROLLING TAIL-RISK FOUNDATION
# Purpose:
# Show that realised Crypto downside risk changes over time. This layer builds
# an equal-weight ten-Crypto market return series and a backward-looking
# rolling 60-day 90% realised CVaR loss measure. It does not replace the Pure
# 95% Minimum-CVaR portfolio optimiser above.
# ============================================================================
STAGE_7_CRYPTO_ASSET_COUNT = 10
STAGE_7_TAIL_RISK_WINDOW = 60
STAGE_7_TAIL_RISK_CONFIDENCE_LEVEL = 0.90


def _validate_crypto_return_panel(
    returns: pd.DataFrame,
    return_col: str,
    expected_asset_count: int,
) -> pd.DataFrame:
    """Validate a long-form Crypto return panel before Stage 7.2 aggregation."""
    required = {"ticker", "date", return_col}
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"returns is missing required columns: {sorted(missing)}")
    if returns.duplicated(["ticker", "date"]).any():
        raise ValueError("returns must contain unique ticker-date observations")

    panel = returns[["ticker", "date", return_col]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    panel = panel.sort_values(["date", "ticker"])
    if not panel["date"].is_monotonic_increasing:
        raise ValueError("return dates must be sorted in chronological order")

    asset_count = int(panel["ticker"].nunique())
    if asset_count != expected_asset_count:
        raise ValueError(
            "Stage 7.2 Crypto market proxy must use the ten-asset Crypto "
            f"universe; found {asset_count} assets"
        )
    if asset_count <= 1:
        raise ValueError("Crypto market proxy must not be based on BTC alone")
    if not pd.api.types.is_numeric_dtype(panel[return_col]):
        raise TypeError(f"{return_col} must contain numeric returns")
    return panel


def build_crypto_market_return(
    crypto_returns: pd.DataFrame,
    expected_asset_count: int = STAGE_7_CRYPTO_ASSET_COUNT,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build the equal-weight ten-Crypto market return series.

    The market return is the simple average of individual Crypto returns on
    dates where all ten Crypto returns are available. Missing returns are not
    filled with zero; the observation count is retained for transparency.
    """
    panel = _validate_crypto_return_panel(
        crypto_returns,
        return_col=return_col,
        expected_asset_count=expected_asset_count,
    )
    grouped = panel.groupby("date", sort=True)[return_col]
    market = grouped.agg(
        crypto_market_return="mean",
        number_of_crypto_assets="count",
    ).reset_index()
    incomplete = market["number_of_crypto_assets"].ne(expected_asset_count)
    market.loc[incomplete, "crypto_market_return"] = np.nan

    if not market["date"].is_monotonic_increasing:
        raise ValueError("Crypto market return dates must be sorted")
    if market["date"].duplicated().any():
        raise ValueError("Crypto market return dates must be unique")
    return market


def rolling_realised_cvar(
    market_returns: pd.DataFrame,
    window: int = STAGE_7_TAIL_RISK_WINDOW,
    confidence_level: float = STAGE_7_TAIL_RISK_CONFIDENCE_LEVEL,
    return_col: str = "crypto_market_return",
) -> pd.DataFrame:
    """Calculate rolling realised CVaR and express it as a positive loss.

    Each valid date uses the most recent 60 valid market returns, sorts them
    from lowest to highest, and averages the worst ceiling(60 * 10%) returns.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1")
    required = {"date", return_col}
    missing = required.difference(market_returns.columns)
    if missing:
        raise ValueError(f"market_returns is missing required columns: {sorted(missing)}")

    tail_count = int(np.ceil(window * (1.0 - confidence_level)))
    if tail_count < 1:
        raise ValueError("tail_count must be at least one observation")

    out = market_returns.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out = out.sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("market return dates must be unique")
    if not pd.api.types.is_numeric_dtype(out[return_col]):
        raise TypeError(f"{return_col} must be numeric")

    valid_returns = out.loc[out[return_col].notna(), ["date", return_col]].reset_index()
    out["rolling_window_observations"] = 0
    out.loc[valid_returns["index"], "rolling_window_observations"] = np.minimum(
        np.arange(1, len(valid_returns) + 1),
        window,
    )
    out["tail_observations"] = np.nan
    out["raw_cvar_60d_90"] = np.nan
    out["crypto_tail_risk_60d_90"] = np.nan

    values = valid_returns[return_col].to_numpy(dtype=float)
    for valid_position in range(window - 1, len(valid_returns)):
        window_values = values[valid_position - window + 1: valid_position + 1]
        raw_cvar = float(np.sort(window_values)[:tail_count].mean())
        row_index = int(valid_returns.loc[valid_position, "index"])
        out.loc[row_index, "tail_observations"] = tail_count
        out.loc[row_index, "raw_cvar_60d_90"] = raw_cvar
        out.loc[row_index, "crypto_tail_risk_60d_90"] = -raw_cvar

    valid_tail = out["crypto_tail_risk_60d_90"].dropna()
    if not valid_tail.empty and (valid_tail < -1e-12).any():
        raise ValueError("positive Crypto tail-risk values must be non-negative")
    check = out["raw_cvar_60d_90"].notna()
    if check.any() and not np.allclose(
        out.loc[check, "crypto_tail_risk_60d_90"],
        -out.loc[check, "raw_cvar_60d_90"],
    ):
        raise ValueError("positive tail risk must equal negative raw CVaR")
    if check.any() and not out.loc[check, "rolling_window_observations"].eq(window).all():
        raise ValueError("valid rolling CVaR dates must use a full 60-observation window")
    if check.any() and not out.loc[check, "tail_observations"].eq(tail_count).all():
        raise ValueError("valid rolling CVaR dates must use the deterministic tail count")
    return out


def build_crypto_tail_risk_series(
    crypto_returns: pd.DataFrame,
    expected_asset_count: int = STAGE_7_CRYPTO_ASSET_COUNT,
    window: int = STAGE_7_TAIL_RISK_WINDOW,
    confidence_level: float = STAGE_7_TAIL_RISK_CONFIDENCE_LEVEL,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build the Stage 7.2 equal-weight Crypto market tail-risk series."""
    market = build_crypto_market_return(
        crypto_returns,
        expected_asset_count=expected_asset_count,
        return_col=return_col,
    )
    tail_risk = rolling_realised_cvar(
        market,
        window=window,
        confidence_level=confidence_level,
        return_col="crypto_market_return",
    )
    first_valid = tail_risk["crypto_tail_risk_60d_90"].first_valid_index()
    if (
        first_valid is not None
        and int(tail_risk.loc[first_valid, "rolling_window_observations"]) < window
    ):
        raise ValueError("first valid tail-risk date occurs before enough observations exist")
    return tail_risk


def summarize_crypto_tail_risk(
    tail_risk: pd.DataFrame,
    window: int = STAGE_7_TAIL_RISK_WINDOW,
    confidence_level: float = STAGE_7_TAIL_RISK_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Return one compact summary row for the Stage 7.2 tail-risk output."""
    required = {
        "date",
        "crypto_market_return",
        "number_of_crypto_assets",
        "rolling_window_observations",
        "tail_observations",
        "raw_cvar_60d_90",
        "crypto_tail_risk_60d_90",
    }
    missing = required.difference(tail_risk.columns)
    if missing:
        raise ValueError(f"tail_risk is missing required columns: {sorted(missing)}")

    data = tail_risk.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    if not data["date"].is_monotonic_increasing or data["date"].duplicated().any():
        raise ValueError("tail-risk dates must be sorted and unique")

    valid_market = data.loc[data["crypto_market_return"].notna()]
    valid_tail = data.loc[data["crypto_tail_risk_60d_90"].notna()]
    if valid_market.empty or valid_tail.empty:
        raise ValueError("tail-risk summary requires valid market and tail-risk rows")

    max_idx = valid_tail["crypto_tail_risk_60d_90"].idxmax()
    tail_observations = int(valid_tail["tail_observations"].iloc[0])
    return pd.DataFrame(
        [
            {
                "crypto_market_series_start_date": valid_market["date"].min().date(),
                "crypto_market_series_end_date": valid_market["date"].max().date(),
                "total_valid_crypto_market_return_observations": int(len(valid_market)),
                "rolling_window_length": int(window),
                "cvar_confidence_level": float(confidence_level),
                "tail_observations_per_window": tail_observations,
                "first_valid_tail_risk_date": valid_tail["date"].min().date(),
                "minimum_positive_tail_risk_value": float(
                    valid_tail["crypto_tail_risk_60d_90"].min()
                ),
                "median_positive_tail_risk_value": float(
                    valid_tail["crypto_tail_risk_60d_90"].median()
                ),
                "mean_positive_tail_risk_value": float(
                    valid_tail["crypto_tail_risk_60d_90"].mean()
                ),
                "maximum_positive_tail_risk_value": float(
                    valid_tail["crypto_tail_risk_60d_90"].max()
                ),
                "date_of_maximum_tail_risk": data.loc[max_idx, "date"].date(),
            }
        ]
    )


def plot_crypto_tail_risk(tail_risk: pd.DataFrame, output_path) -> object:
    """Plot the positive rolling 60-day 90% realised Crypto tail-risk series."""
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plot_data = tail_risk.loc[tail_risk["crypto_tail_risk_60d_90"].notna()].copy()
    if plot_data.empty:
        raise ValueError("tail-risk figure requires at least one valid rolling value")
    plot_data["date"] = pd.to_datetime(plot_data["date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        plot_data["date"],
        plot_data["crypto_tail_risk_60d_90"],
        color="tab:red",
        linewidth=1.8,
    )
    ax.fill_between(
        plot_data["date"],
        0.0,
        plot_data["crypto_tail_risk_60d_90"].to_numpy(dtype=float),
        color="tab:red",
        alpha=0.14,
    )

    y_max = float(plot_data["crypto_tail_risk_60d_90"].max())
    label_y = y_max * 0.92
    markers = [
        (
            "2020-03-01",
            "2020-03-31",
            "March 2020\npandemic crash",
            "0.4",
            0.10,
            "2020-03-08",
        ),
        (
            "2021-01-01",
            "2021-12-31",
            "2021\nbull market + volatility",
            "tab:orange",
            0.07,
            "2021-03-01",
        ),
        (
            "2022-01-01",
            "2022-12-31",
            "2022\ncrypto winter",
            "tab:blue",
            0.06,
            "2022-02-01",
        ),
    ]
    x_min = plot_data["date"].min()
    x_max = plot_data["date"].max()
    for start, end, label, color, alpha, text_date in markers:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        if end_date >= x_min and start_date <= x_max:
            ax.axvspan(
                max(start_date, x_min),
                min(end_date, x_max),
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            ax.text(
                pd.Timestamp(text_date),
                label_y,
                label,
                fontsize=8,
                color="0.3",
                va="top",
            )

    ftx_date = pd.Timestamp("2022-11-11")
    if x_min <= ftx_date <= x_max:
        ax.axvline(ftx_date, color="0.25", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.annotate(
            "Nov 2022\nFTX stress",
            xy=(ftx_date, y_max * 0.78),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7},
        )

    ax.set_title("Crypto Market Rolling 60-Day 90% Tail Risk", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Loss in Worst 10% of Days")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max * 1.08)
    ax.grid(True, alpha=0.24)
    fig.text(
        0.01,
        0.01,
        "Context markers are reference periods only. The measure is backward-looking realised downside risk, not a crash forecast.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 7.3: LAYER 3A - CONTINUOUS CRYPTO RISK SCORE
# Purpose:
# Convert the completed Stage 7.2 rolling Crypto tail-risk series into a
# no-look-ahead 0-100 score on monthly Combined portfolio decision dates. This
# stage only measures risk; it does not map scores into Crypto caps or generate
# adaptive fund weights.
# ============================================================================
STAGE_7_RISK_SCORE_MIN_HISTORY = 60


def identify_monthly_rebalance_dates(
    returns,
    start_date=None,
    end_date=None,
) -> pd.DatetimeIndex:
    """Return each month latest valid portfolio trading date.

    The rule matches the monthly decision-date logic used inside the
    walk-forward backtest before applying the next-trading-day effective-date
    rule. Passing a Combined return matrix therefore produces the monthly score
    dates used by Stage 7.3.
    """
    if isinstance(returns, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(returns)
    elif isinstance(returns, pd.Series | pd.DataFrame):
        if isinstance(returns.index, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(returns.index)
        elif "date" in returns:
            dates = pd.DatetimeIndex(pd.to_datetime(returns["date"]))
        else:
            raise TypeError("returns must have a DatetimeIndex or a date column")
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(returns))

    dates = pd.DatetimeIndex(pd.Series(dates).dropna().drop_duplicates().sort_values())
    if dates.empty:
        raise ValueError("at least one portfolio trading date is required")

    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if dates.empty:
        raise ValueError("no portfolio trading dates remain after date filtering")

    monthly = (
        pd.DataFrame({"date": dates})
        .groupby(dates.to_period("M"), sort=True)["date"]
        .max()
        .sort_values()
    )
    out = pd.DatetimeIndex(monthly.to_numpy(), name="date")
    if not out.is_monotonic_increasing or out.has_duplicates:
        raise ValueError("monthly rebalance dates must be sorted and unique")
    return out


def calculate_expanding_tail_risk_percentile(
    current_tail_risk: float,
    prior_tail_risk_observations,
) -> float:
    """Return the expanding historical percentile score on a 0-100 scale.

    Ties are handled deterministically by counting prior observations less than
    or equal to the current tail-risk value.
    """
    prior = pd.Series(prior_tail_risk_observations, dtype="float64").dropna()
    if prior.empty:
        raise ValueError("prior_tail_risk_observations must contain at least one value")
    if not np.isfinite(float(current_tail_risk)):
        raise ValueError("current_tail_risk must be finite")
    return float((prior.le(float(current_tail_risk)).sum() / len(prior)) * 100.0)


def _validate_tail_risk_input_for_scores(
    tail_risk: pd.DataFrame,
    tail_risk_col: str,
    return_col: str,
) -> pd.DataFrame:
    """Validate the completed Stage 7.2 tail-risk series for Stage 7.3."""
    required = {"date", return_col, tail_risk_col}
    missing = required.difference(tail_risk.columns)
    if missing:
        raise ValueError(f"tail_risk is missing required columns: {sorted(missing)}")

    data = tail_risk.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    data = data.sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError("tail-risk dates must be unique")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("tail-risk dates must be sorted")
    for column in [return_col, tail_risk_col]:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"{column} must be numeric")
    valid_tail = data[tail_risk_col].dropna()
    if valid_tail.empty:
        raise ValueError("tail_risk must contain valid daily tail-risk observations")
    if (valid_tail < -1e-12).any():
        raise ValueError("positive Crypto tail-risk values must be non-negative")
    return data


def build_crypto_risk_score_series(
    tail_risk: pd.DataFrame,
    rebalance_dates,
    min_history: int = STAGE_7_RISK_SCORE_MIN_HISTORY,
    tail_risk_col: str = "crypto_tail_risk_60d_90",
    return_col: str = "crypto_market_return",
) -> pd.DataFrame:
    """Build monthly no-look-ahead Crypto risk scores from Stage 7.2 output.

    For each monthly score date, the current tail-risk value is the latest valid
    Stage 7.2 tail-risk observation dated on or before that score date. The
    percentile history contains only valid daily tail-risk observations dated
    before that source observation, so the current observation is not included
    in its own ranking.
    """
    if min_history < 1:
        raise ValueError("min_history must be positive")

    data = _validate_tail_risk_input_for_scores(tail_risk, tail_risk_col, return_col)
    score_dates = pd.DatetimeIndex(pd.to_datetime(rebalance_dates)).drop_duplicates()
    score_dates = pd.DatetimeIndex(score_dates.sort_values(), name="date")
    if score_dates.empty:
        raise ValueError("rebalance_dates must contain at least one date")
    if score_dates.has_duplicates or not score_dates.is_monotonic_increasing:
        raise ValueError("rebalance_dates must be sorted and unique")

    valid_daily = data.loc[data[tail_risk_col].notna(), ["date", return_col, tail_risk_col]]
    valid_daily = valid_daily.sort_values("date").reset_index(drop=True)
    valid_dates = valid_daily["date"].to_numpy(dtype="datetime64[ns]")

    records = []
    for score_date in score_dates:
        source_position = np.searchsorted(valid_dates, np.datetime64(score_date), side="right") - 1
        if source_position < 0:
            source_date = pd.NaT
            current_return = np.nan
            current_tail_risk = np.nan
            prior_values = np.array([], dtype=float)
            prior_dates = pd.Series(dtype="datetime64[ns]")
        else:
            source_row = valid_daily.iloc[int(source_position)]
            source_date = pd.Timestamp(source_row["date"])
            current_return = float(source_row[return_col])
            current_tail_risk = float(source_row[tail_risk_col])
            prior = valid_daily.loc[valid_daily["date"] < source_date]
            prior_values = prior[tail_risk_col].to_numpy(dtype=float)
            prior_dates = prior["date"]

        valid_history_count = len(prior_values)
        score_available = bool(
            valid_history_count >= min_history and np.isfinite(current_tail_risk)
        )
        risk_score = (
            calculate_expanding_tail_risk_percentile(current_tail_risk, prior_values)
            if score_available
            else np.nan
        )

        records.append(
            {
                "date": score_date,
                "crypto_market_return": current_return,
                "rolling_tail_risk": current_tail_risk,
                "risk_score": risk_score,
                "valid_history_count": valid_history_count,
                "risk_score_available": score_available,
                "risk_score_ready": score_available,
                "history_start_date": (
                    pd.Timestamp(prior_dates.min()) if valid_history_count else pd.NaT
                ),
                "history_end_date": (
                    pd.Timestamp(prior_dates.max()) if valid_history_count else pd.NaT
                ),
                "rebalance_month": score_date.to_period("M").strftime("%Y-%m"),
                "tail_risk_source_date": source_date,
            }
        )

    out = pd.DataFrame(records)
    if not out["date"].is_monotonic_increasing or out["date"].duplicated().any():
        raise ValueError("monthly Crypto risk-score dates must be sorted and unique")
    available = out.loc[out["risk_score_available"]]
    if not available.empty and not available["risk_score"].between(0.0, 100.0).all():
        raise ValueError("available Crypto risk scores must be between 0 and 100")
    unavailable = out.loc[~out["risk_score_available"]]
    if not unavailable["risk_score"].isna().all():
        raise ValueError("unavailable Crypto risk scores must remain missing")
    return out


def summarize_crypto_risk_scores(
    risk_scores: pd.DataFrame,
    min_history: int = STAGE_7_RISK_SCORE_MIN_HISTORY,
) -> pd.DataFrame:
    """Return a compact one-row summary for the Stage 7.3 risk-score output."""
    required = {
        "date",
        "risk_score",
        "valid_history_count",
        "risk_score_available",
    }
    missing = required.difference(risk_scores.columns)
    if missing:
        raise ValueError(f"risk_scores is missing required columns: {sorted(missing)}")

    data = risk_scores.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    data = data.sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("risk_scores must contain at least one monthly row")
    if data["date"].duplicated().any():
        raise ValueError("risk-score dates must be unique")

    available = data.loc[data["risk_score_available"].astype(bool)].copy()
    if available.empty:
        first_available_date = pd.NaT
        min_score = median_score = mean_score = max_score = np.nan
        min_date = max_date = pd.NaT
    else:
        min_idx = available["risk_score"].idxmin()
        max_idx = available["risk_score"].idxmax()
        first_available_date = available["date"].min()
        min_score = float(available["risk_score"].min())
        median_score = float(available["risk_score"].median())
        mean_score = float(available["risk_score"].mean())
        max_score = float(available["risk_score"].max())
        min_date = data.loc[min_idx, "date"]
        max_date = data.loc[max_idx, "date"]

    return pd.DataFrame(
        [
            {
                "first_monthly_score_date": data["date"].min().date(),
                "last_monthly_score_date": data["date"].max().date(),
                "total_monthly_score_date_rows": len(data),
                "minimum_required_history": int(min_history),
                "first_risk_score_available_date": (
                    first_available_date.date()
                    if pd.notna(first_available_date)
                    else pd.NaT
                ),
                "number_of_available_scores": len(available),
                "number_of_unavailable_early_scores": int(
                    (~data["risk_score_available"].astype(bool)).sum()
                ),
                "minimum_risk_score": min_score,
                "median_risk_score": median_score,
                "mean_risk_score": mean_score,
                "maximum_risk_score": max_score,
                "date_of_maximum_risk_score": (
                    max_date.date() if pd.notna(max_date) else pd.NaT
                ),
                "date_of_minimum_risk_score": (
                    min_date.date() if pd.notna(min_date) else pd.NaT
                ),
            }
        ]
    )


# ============================================================================
# STAGE 7.4: LAYER 3B - CONTINUOUS PERSONALISED CRYPTO BUDGETS
# Purpose:
# Convert Stage 7.3 monthly Crypto risk scores into target Crypto and Equity
# asset-class budgets. These are target sleeve budgets, not upper-bound caps.
# ============================================================================
PERSONALISED_CRYPTO_BUDGET_RANGES = MappingProxyType(
    {
        "conservative": MappingProxyType(
            {"minimum_raw_crypto_budget": 0.00, "maximum_raw_crypto_budget": 0.10}
        ),
        "balanced": MappingProxyType(
            {"minimum_raw_crypto_budget": 0.05, "maximum_raw_crypto_budget": 0.20}
        ),
        "growth": MappingProxyType(
            {"minimum_raw_crypto_budget": 0.10, "maximum_raw_crypto_budget": 0.30}
        ),
    }
)


def map_risk_score_to_raw_crypto_budget(risk_score: float, profile: str) -> float:
    """Map one 0-100 score to a target Raw Crypto Budget."""
    if profile not in PERSONALISED_CRYPTO_BUDGET_RANGES:
        raise ValueError(f"unknown investor profile: {profile}")
    if not isinstance(risk_score, int | float | np.number) or not np.isfinite(risk_score):
        raise ValueError("risk_score must be a finite numeric value")
    risk_score = float(risk_score)
    if not 0.0 <= risk_score <= 100.0:
        raise ValueError("risk_score must be in the closed interval [0, 100]")

    settings = PERSONALISED_CRYPTO_BUDGET_RANGES[profile]
    minimum_budget = settings["minimum_raw_crypto_budget"]
    maximum_budget = settings["maximum_raw_crypto_budget"]
    raw_budget = maximum_budget - (
        maximum_budget - minimum_budget
    ) * (risk_score / 100.0)
    return float(np.clip(raw_budget, minimum_budget, maximum_budget))


def build_personalised_crypto_budget_series(risk_scores: pd.DataFrame) -> pd.DataFrame:
    """Build monthly target Crypto and Equity budgets for each profile."""
    required = {"date", "rolling_tail_risk", "risk_score", "risk_score_available"}
    missing = required.difference(risk_scores.columns)
    if missing:
        raise ValueError(f"risk_scores is missing required columns: {sorted(missing)}")

    data = risk_scores.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    data = data.sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("risk_scores must contain at least one monthly row")
    if data["date"].duplicated().any():
        raise ValueError("risk-score dates must be unique")

    available = data["risk_score_available"].astype(bool) & data["risk_score"].notna()
    if not data.loc[available, "risk_score"].between(0.0, 100.0).all():
        raise ValueError("available Crypto risk scores must be between 0 and 100")

    records = []
    for row in data.itertuples(index=False):
        budget_available = bool(row.risk_score_available) and pd.notna(row.risk_score)
        for profile, settings in PERSONALISED_CRYPTO_BUDGET_RANGES.items():
            raw_crypto_budget = (
                map_risk_score_to_raw_crypto_budget(row.risk_score, profile)
                if budget_available
                else np.nan
            )
            records.append(
                {
                    "date": row.date,
                    "profile": profile,
                    "rolling_tail_risk": row.rolling_tail_risk,
                    "risk_score": row.risk_score,
                    "risk_score_available": bool(row.risk_score_available),
                    "minimum_raw_crypto_budget": settings[
                        "minimum_raw_crypto_budget"
                    ],
                    "maximum_raw_crypto_budget": settings[
                        "maximum_raw_crypto_budget"
                    ],
                    "raw_crypto_budget": raw_crypto_budget,
                    "raw_equity_budget": (
                        1.0 - raw_crypto_budget if budget_available else np.nan
                    ),
                    "raw_budget_available": budget_available,
                }
            )

    out = pd.DataFrame(records)
    out["profile"] = pd.Categorical(
        out["profile"],
        categories=list(PERSONALISED_CRYPTO_BUDGET_RANGES),
        ordered=True,
    )
    return out.sort_values(["date", "profile"]).reset_index(drop=True)


def summarize_personalised_crypto_budgets(budgets: pd.DataFrame) -> pd.DataFrame:
    """Summarise valid and unavailable Raw Crypto Budgets by profile."""
    required = {
        "date",
        "profile",
        "minimum_raw_crypto_budget",
        "maximum_raw_crypto_budget",
        "raw_crypto_budget",
        "raw_budget_available",
    }
    missing = required.difference(budgets.columns)
    if missing:
        raise ValueError(f"budgets is missing required columns: {sorted(missing)}")

    data = budgets.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    invalid_profiles = sorted(
        set(data["profile"].astype(str)).difference(PERSONALISED_CRYPTO_BUDGET_RANGES)
    )
    if invalid_profiles:
        raise ValueError(f"unknown investor profiles: {invalid_profiles}")

    records = []
    for profile in PERSONALISED_CRYPTO_BUDGET_RANGES:
        panel = data.loc[data["profile"].astype(str).eq(profile)].copy()
        valid = panel.loc[panel["raw_budget_available"].astype(bool)].copy()
        if valid.empty:
            first_valid_date = min_idx = max_idx = pd.NaT
            minimum_observed = median_observed = mean_observed = maximum_observed = np.nan
        else:
            min_idx = valid["raw_crypto_budget"].idxmin()
            max_idx = valid["raw_crypto_budget"].idxmax()
            first_valid_date = valid["date"].min()
            minimum_observed = float(valid["raw_crypto_budget"].min())
            median_observed = float(valid["raw_crypto_budget"].median())
            mean_observed = float(valid["raw_crypto_budget"].mean())
            maximum_observed = float(valid["raw_crypto_budget"].max())

        records.append(
            {
                "profile": profile,
                "minimum_permitted_raw_crypto_budget": float(
                    panel["minimum_raw_crypto_budget"].iloc[0]
                ),
                "maximum_permitted_raw_crypto_budget": float(
                    panel["maximum_raw_crypto_budget"].iloc[0]
                ),
                "number_of_valid_raw_budgets": len(valid),
                "number_of_unavailable_raw_budgets": int(
                    (~panel["raw_budget_available"].astype(bool)).sum()
                ),
                "first_valid_raw_budget_date": (
                    first_valid_date.date() if pd.notna(first_valid_date) else pd.NaT
                ),
                "minimum_observed_raw_crypto_budget": minimum_observed,
                "median_observed_raw_crypto_budget": median_observed,
                "mean_observed_raw_crypto_budget": mean_observed,
                "maximum_observed_raw_crypto_budget": maximum_observed,
                "date_of_minimum_observed_raw_crypto_budget": (
                    data.loc[min_idx, "date"].date() if pd.notna(min_idx) else pd.NaT
                ),
                "date_of_maximum_observed_raw_crypto_budget": (
                    data.loc[max_idx, "date"].date() if pd.notna(max_idx) else pd.NaT
                ),
            }
        )

    return pd.DataFrame(records)


def plot_personalised_crypto_budget_curves(output_path) -> object:
    """Plot the Stage 7.4 risk-score-to-target-budget design curves."""
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    risk_scores = np.linspace(0.0, 100.0, 101)
    labels = {
        "conservative": "Conservative",
        "balanced": "Balanced",
        "growth": "Growth",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for profile in PERSONALISED_CRYPTO_BUDGET_RANGES:
        budgets = [
            map_risk_score_to_raw_crypto_budget(score, profile)
            for score in risk_scores
        ]
        ax.plot(risk_scores, budgets, linewidth=2.0, label=labels[profile])

    ax.set_title("Personalised Crypto Budgets by Risk Score", pad=14)
    ax.set_xlabel("Crypto Risk Score")
    ax.set_ylabel("Target Crypto Allocation")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.32)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 7.6: PERSONALISED CRYPTO BUDGET SMOOTHING
# Purpose:
# Convert one profile's monthly Raw Crypto Budget target series into a smoother
# Applied Crypto and Equity Budget series. This stage does not create funds,
# portfolio weights or realised sleeve allocations.
# ============================================================================
STAGE_7_BUDGET_NO_ADJUSTMENT_THRESHOLD = 0.01
STAGE_7_BUDGET_MONTHLY_MOVEMENT_LIMIT = 0.05


def smooth_crypto_budget_profile(
    profile_budgets: pd.DataFrame,
    no_adjustment_threshold: float = STAGE_7_BUDGET_NO_ADJUSTMENT_THRESHOLD,
    monthly_movement_limit: float = STAGE_7_BUDGET_MONTHLY_MOVEMENT_LIMIT,
) -> pd.DataFrame:
    """Apply Stage 7.6 smoothing to one profile's Raw Crypto Budget series."""
    required = {
        "date",
        "profile",
        "rolling_tail_risk",
        "risk_score",
        "minimum_raw_crypto_budget",
        "maximum_raw_crypto_budget",
        "raw_crypto_budget",
        "raw_equity_budget",
        "raw_budget_available",
    }
    missing = required.difference(profile_budgets.columns)
    if missing:
        raise ValueError(f"profile_budgets is missing columns: {sorted(missing)}")
    if no_adjustment_threshold < 0 or monthly_movement_limit <= 0:
        raise ValueError("smoothing thresholds must be non-negative and positive")

    data = profile_budgets.copy()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    if data.empty:
        raise ValueError("profile_budgets must contain at least one row")
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("profile budget dates must be sorted and unique")

    profiles = data["profile"].dropna().astype(str).unique()
    if len(profiles) != 1 or profiles[0] not in PERSONALISED_CRYPTO_BUDGET_RANGES:
        raise ValueError("profile_budgets must contain one valid investor profile")
    profile = profiles[0]
    settings = PERSONALISED_CRYPTO_BUDGET_RANGES[profile]
    minimum_budget = settings["minimum_raw_crypto_budget"]
    maximum_budget = settings["maximum_raw_crypto_budget"]
    available = (
        data["raw_budget_available"].astype(bool)
        & data["raw_crypto_budget"].notna()
    )
    if not data.loc[available, "raw_crypto_budget"].between(
        minimum_budget,
        maximum_budget,
    ).all():
        raise ValueError("available Raw Crypto Budgets must stay inside profile range")

    previous_applied = np.nan
    records = []
    for row in data.itertuples(index=False):
        raw_available = bool(row.raw_budget_available) and pd.notna(
            row.raw_crypto_budget
        )
        no_adjustment = False
        monthly_limit = False

        if not raw_available:
            applied_budget = np.nan
            previous_value = np.nan
        elif pd.isna(previous_applied):
            applied_budget = float(row.raw_crypto_budget)
            previous_value = np.nan
        else:
            previous_value = float(previous_applied)
            desired_change = float(row.raw_crypto_budget) - previous_value
            no_adjustment = abs(desired_change) < no_adjustment_threshold
            monthly_limit = abs(desired_change) > monthly_movement_limit
            if no_adjustment:
                applied_budget = previous_value
            elif monthly_limit:
                applied_budget = (
                    previous_value + np.sign(desired_change) * monthly_movement_limit
                )
            else:
                applied_budget = float(row.raw_crypto_budget)

        budget_change = (
            applied_budget - previous_value
            if pd.notna(applied_budget) and pd.notna(previous_value)
            else np.nan
        )
        if pd.notna(applied_budget):
            previous_applied = float(
                np.clip(applied_budget, minimum_budget, maximum_budget)
            )
            applied_budget = previous_applied

        records.append(
            {
                "date": row.date,
                "profile": profile,
                "rolling_tail_risk": row.rolling_tail_risk,
                "risk_score": row.risk_score,
                "raw_crypto_budget": row.raw_crypto_budget,
                "raw_equity_budget": row.raw_equity_budget,
                "previous_applied_crypto_budget": previous_value,
                "applied_crypto_budget": applied_budget,
                "applied_equity_budget": (
                    1.0 - applied_budget if pd.notna(applied_budget) else np.nan
                ),
                "budget_change": budget_change,
                "no_adjustment_triggered": no_adjustment,
                "monthly_limit_triggered": monthly_limit,
                "applied_budget_available": pd.notna(applied_budget),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# STAGE 7.5: SIX OFFICIAL TWO-SLEEVE STOCK-CRYPTO FUNDS
# Purpose:
# Build the six official funds by optimising Equity and Crypto sleeves
# separately with Pure 95% Min-CVaR, then scaling them by target budgets.
# ============================================================================
STAGE_7_FIXED_CRYPTO_BUDGETS = MappingProxyType(
    {
        "conservative": 0.10,
        "balanced": 0.20,
        "growth": 0.30,
    }
)


def stage7_two_sleeve_fund_configs() -> list[dict]:
    """Return the six official Stage 7.5 Budget-based fund configurations."""
    configs = []
    for profile in STAGE_7_FIXED_CRYPTO_BUDGETS:
        configs.extend(
            [
                {
                    "fund": f"{profile}_fixed",
                    "profile": profile,
                    "fund_type": "fixed",
                    "fixed_crypto_budget": STAGE_7_FIXED_CRYPTO_BUDGETS[profile],
                },
                {
                    "fund": f"{profile}_adaptive",
                    "profile": profile,
                    "fund_type": "adaptive",
                    "fixed_crypto_budget": None,
                },
            ]
        )
    fund_names = [config["fund"] for config in configs]
    if len(set(fund_names)) != len(fund_names):
        raise ValueError("Stage 7.5 fund names must be unique")
    return configs


def run_stage7_two_sleeve_backtest(
    returns: pd.DataFrame,
    equity_assets,
    crypto_assets,
    adaptive_budget_history: pd.DataFrame,
    window_size: int = 252,
    confidence_level: float = DEFAULT_CVAR_CONFIDENCE_LEVEL,
) -> dict[str, pd.DataFrame]:
    """Run the six official two-sleeve Budget-based Stage 7.5 funds."""
    required_budget_columns = {
        "date",
        "profile",
        "applied_crypto_budget",
        "applied_equity_budget",
        "applied_budget_available",
    }
    missing = required_budget_columns.difference(adaptive_budget_history.columns)
    if missing:
        raise ValueError(f"adaptive_budget_history is missing columns: {sorted(missing)}")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    clean_returns = _validate_backtest_returns(returns)
    equity_assets = list(equity_assets)
    crypto_assets = list(crypto_assets)
    if len(equity_assets) != 50 or len(crypto_assets) != 10:
        raise ValueError("Stage 7.5 requires 50 Equity assets and 10 Crypto assets")
    if set(equity_assets).intersection(crypto_assets):
        raise ValueError("Equity and Crypto sleeve assets must not overlap")
    missing_assets = sorted(set(equity_assets + crypto_assets).difference(clean_returns.columns))
    if missing_assets:
        raise ValueError(f"Stage 7.5 returns are missing assets: {missing_assets}")

    budgets = adaptive_budget_history.copy()
    budgets["date"] = pd.to_datetime(budgets["date"]).astype("datetime64[ns]")
    budgets = budgets.sort_values(["date", "profile"]).reset_index(drop=True)
    if budgets.duplicated(["date", "profile"]).any():
        raise ValueError("adaptive budgets must contain unique date-profile rows")
    invalid_profiles = sorted(
        set(budgets["profile"].astype(str)).difference(STAGE_7_FIXED_CRYPTO_BUDGETS)
    )
    if invalid_profiles:
        raise ValueError(f"unknown investor profiles: {invalid_profiles}")
    available_budgets = budgets.loc[budgets["applied_budget_available"].astype(bool)]
    if not available_budgets["applied_crypto_budget"].between(0.0, 1.0).all():
        raise ValueError("Applied Crypto Budgets must be in [0, 1]")
    if not available_budgets["applied_equity_budget"].between(0.0, 1.0).all():
        raise ValueError("Applied Equity Budgets must be in [0, 1]")

    crypto_budget_lookup = available_budgets.pivot(
        index="date",
        columns="profile",
        values="applied_crypto_budget",
    )
    equity_budget_lookup = available_budgets.pivot(
        index="date",
        columns="profile",
        values="applied_equity_budget",
    )
    common_budget_dates = crypto_budget_lookup.dropna(how="any").index.intersection(
        equity_budget_lookup.dropna(how="any").index
    )

    dates = clean_returns.index
    month_end_positions = (
        pd.Series(range(len(dates)), index=dates)
        .groupby(dates.to_period("M"))
        .max()
        .astype(int)
    )
    rebalance_positions = []
    sleeve_cache = {}
    for position in month_end_positions:
        decision_date = dates[position]
        if position < window_size - 1 or position + 1 >= len(clean_returns):
            continue
        if decision_date not in common_budget_dates:
            continue
        window = clean_returns.iloc[position - window_size + 1: position + 1]
        equity_window = window[equity_assets]
        crypto_window = window[crypto_assets]
        if len(equity_window.dropna(how="any")) < window_size:
            continue
        if len(crypto_window.dropna(how="any")) < window_size:
            continue

        equity_weights, equity_metadata = min_cvar_weights(
            equity_window,
            confidence_level=confidence_level,
        )
        crypto_weights, crypto_metadata = min_cvar_weights(
            crypto_window,
            confidence_level=confidence_level,
        )
        equity_weights = equity_weights.reindex(equity_assets).fillna(0.0)
        crypto_weights = crypto_weights.reindex(crypto_assets).fillna(0.0)
        sleeve_cache[position] = {
            "decision_date": decision_date,
            "effective_date": dates[position + 1],
            "equity_weights": equity_weights,
            "crypto_weights": crypto_weights,
            "equity_metadata": equity_metadata,
            "crypto_metadata": crypto_metadata,
        }
        rebalance_positions.append(position)

    if not rebalance_positions:
        raise ValueError("no valid Stage 7.5 two-sleeve rebalance dates are available")

    configs = stage7_two_sleeve_fund_configs()
    assets = pd.Index([*equity_assets, *crypto_assets])
    asset_class = pd.Series("Equity", index=assets)
    asset_class.loc[crypto_assets] = "Crypto"
    effective_positions = {position + 1: position for position in rebalance_positions}
    first_live_position = min(effective_positions)
    target_by_fund = {}
    weight_records = []
    log_records = []

    for position in rebalance_positions:
        sleeve = sleeve_cache[position]
        decision_date = sleeve["decision_date"]
        effective_date = sleeve["effective_date"]
        for config in configs:
            profile = config["profile"]
            if config["fund_type"] == "fixed":
                target_crypto_budget = float(config["fixed_crypto_budget"])
                target_equity_budget = 1.0 - target_crypto_budget
            else:
                target_crypto_budget = float(crypto_budget_lookup.loc[decision_date, profile])
                target_equity_budget = float(equity_budget_lookup.loc[decision_date, profile])
            if not np.isclose(target_crypto_budget + target_equity_budget, 1.0):
                raise ValueError("target Equity and Crypto budgets must sum to one")

            target = pd.Series(0.0, index=assets, name="portfolio_weight")
            target.loc[equity_assets] = sleeve["equity_weights"] * target_equity_budget
            target.loc[crypto_assets] = sleeve["crypto_weights"] * target_crypto_budget
            valid, message = _validate_final_weights(target, effective_cap=None)
            if not valid:
                raise ValueError(
                    f"invalid Stage 7.5 weights for {config['fund']} "
                    f"on {decision_date.date()}: {message}"
                )

            target_by_fund[(config["fund"], dates[position + 1])] = target
            for ticker in assets:
                if ticker in equity_assets:
                    sleeve_weight = float(sleeve["equity_weights"].loc[ticker])
                else:
                    sleeve_weight = float(sleeve["crypto_weights"].loc[ticker])
                weight_records.append(
                    {
                        "decision_date": decision_date,
                        "effective_date": effective_date,
                        "fund": config["fund"],
                        "profile": profile,
                        "fund_type": config["fund_type"],
                        "ticker": ticker,
                        "asset_class": asset_class.loc[ticker],
                        "sleeve_weight": sleeve_weight,
                        "portfolio_weight": float(target.loc[ticker]),
                    }
                )

            post_crypto_weight = float(target.loc[crypto_assets].sum())
            post_equity_weight = float(target.loc[equity_assets].sum())
            log_records.append(
                {
                    "decision_date": decision_date,
                    "effective_date": effective_date,
                    "fund": config["fund"],
                    "profile": profile,
                    "fund_type": config["fund_type"],
                    "target_crypto_budget": target_crypto_budget,
                    "target_equity_budget": target_equity_budget,
                    "post_rebalance_crypto_weight": post_crypto_weight,
                    "post_rebalance_equity_weight": post_equity_weight,
                    "equity_solver_status": sleeve["equity_metadata"]["solver_status"],
                    "crypto_solver_status": sleeve["crypto_metadata"]["solver_status"],
                    "equity_fallback_used": sleeve["equity_metadata"]["fallback_used"],
                    "crypto_fallback_used": sleeve["crypto_metadata"]["fallback_used"],
                }
            )

    returns_records = []
    turnover_records = []
    for config in configs:
        fund = config["fund"]
        current_weights = None
        for position in range(first_live_position, len(clean_returns)):
            date = dates[position]
            if position in effective_positions:
                decision_position = effective_positions[position]
                sleeve = sleeve_cache[decision_position]
                target = target_by_fund[(fund, date)]
                pretrade = target if current_weights is None else current_weights.copy()
                turnover = (
                    np.nan
                    if current_weights is None
                    else 0.5 * float((target - pretrade).abs().sum())
                )
                turnover_records.append(
                    {
                        "decision_date": sleeve["decision_date"],
                        "effective_date": date,
                        "fund": fund,
                        "profile": config["profile"],
                        "fund_type": config["fund_type"],
                        "turnover": turnover,
                    }
                )
                current_weights = target.copy()

            held_assets = current_weights.index[current_weights.gt(_WEIGHT_BOUND_TOLERANCE)]
            day_returns = clean_returns.loc[date, held_assets]
            if day_returns.isna().any():
                missing_live = day_returns.index[day_returns.isna()].tolist()
                raise ValueError(f"missing live returns on {date.date()}: {missing_live}")
            portfolio_return = float((day_returns * current_weights.loc[held_assets]).sum())
            gross_return = 1.0 + portfolio_return
            if gross_return <= 0:
                raise ValueError(f"portfolio gross return is non-positive on {date.date()}")
            returns_records.append(
                {
                    "date": date,
                    "fund": fund,
                    "profile": config["profile"],
                    "fund_type": config["fund_type"],
                    "gross_return": gross_return,
                }
            )
            drifted = current_weights.copy()
            drifted.loc[held_assets] = (
                current_weights.loc[held_assets] * (1.0 + day_returns) / gross_return
            )
            current_weights = drifted / drifted.sum()

    return {
        "returns": pd.DataFrame(returns_records),
        "weights": pd.DataFrame(weight_records),
        "turnover": pd.DataFrame(turnover_records),
        "rebalance_log": pd.DataFrame(log_records),
    }


# ============================================================================
# STAGE 7.7: RAW, APPLIED, TARGET AND MONTHLY PRE-TRADE ALLOCATION HISTORY
# Purpose:
# Organise existing Stage 7 Budget and fund outputs into monthly target/pretrade
# allocation history.
# ============================================================================
STAGE_7_ALLOCATION_HISTORY_COLUMNS = [
    "decision_date",
    "effective_date",
    "fund",
    "profile",
    "fund_type",
    "rolling_tail_risk",
    "risk_score",
    "raw_crypto_budget",
    "raw_equity_budget",
    "previous_applied_crypto_budget",
    "applied_crypto_budget",
    "applied_equity_budget",
    "previous_target_crypto_budget",
    "target_crypto_budget",
    "target_equity_budget",
    "pretrade_crypto_weight",
    "pretrade_equity_weight",
    "market_drift_from_previous_target",
    "post_rebalance_crypto_weight",
    "post_rebalance_equity_weight",
    "crypto_rebalance_change",
    "target_budget_change",
    "no_adjustment_triggered",
    "monthly_limit_triggered",
    "turnover",
]


def _drift_weights_for_stage7_allocation(
    current_weights: pd.Series,
    day_returns: pd.Series,
    date,
) -> pd.Series:
    """Apply realised asset returns and renormalise portfolio weights."""
    held_assets = current_weights.index[current_weights.gt(_WEIGHT_BOUND_TOLERANCE)]
    live_returns = day_returns.loc[held_assets]
    if live_returns.isna().any():
        missing_live = live_returns.index[live_returns.isna()].tolist()
        raise ValueError(f"missing live returns on {pd.Timestamp(date).date()}: {missing_live}")
    portfolio_return = float((live_returns * current_weights.loc[held_assets]).sum())
    gross_return = 1.0 + portfolio_return
    if gross_return <= 0:
        raise ValueError(
            f"portfolio gross return is non-positive on {pd.Timestamp(date).date()}"
        )
    drifted = current_weights.copy()
    drifted.loc[held_assets] = (
        current_weights.loc[held_assets] * (1.0 + live_returns) / gross_return
    )
    return drifted / drifted.sum()


def build_stage7_allocation_history(
    rebalance_log: pd.DataFrame,
    fund_weights: pd.DataFrame,
    fund_turnover: pd.DataFrame,
    adaptive_budget_history: pd.DataFrame,
    returns: pd.DataFrame,
    crypto_assets,
) -> pd.DataFrame:
    """Build the monthly Stage 7.7 allocation history."""
    required_log_columns = {
        "decision_date",
        "effective_date",
        "fund",
        "profile",
        "fund_type",
        "target_crypto_budget",
        "target_equity_budget",
        "post_rebalance_crypto_weight",
        "post_rebalance_equity_weight",
    }
    required_budget_columns = {
        "date",
        "profile",
        "rolling_tail_risk",
        "risk_score",
        "raw_crypto_budget",
        "raw_equity_budget",
        "previous_applied_crypto_budget",
        "applied_crypto_budget",
        "applied_equity_budget",
        "budget_change",
        "no_adjustment_triggered",
        "monthly_limit_triggered",
        "previous_target_crypto_budget",
    }
    required_weight_columns = {
        "effective_date",
        "fund",
        "ticker",
        "asset_class",
        "portfolio_weight",
    }
    required_turnover_columns = {"effective_date", "fund", "turnover"}
    missing_log = required_log_columns.difference(rebalance_log.columns)
    missing_weights = required_weight_columns.difference(fund_weights.columns)
    missing_budget = required_budget_columns.difference(adaptive_budget_history.columns)
    missing_turnover = required_turnover_columns.difference(fund_turnover.columns)
    if missing_log:
        raise ValueError(f"rebalance_log is missing columns: {sorted(missing_log)}")
    if missing_weights:
        raise ValueError(f"fund_weights is missing columns: {sorted(missing_weights)}")
    if missing_budget:
        raise ValueError(
            f"adaptive_budget_history is missing columns: {sorted(missing_budget)}"
        )
    if missing_turnover:
        raise ValueError(f"fund_turnover is missing columns: {sorted(missing_turnover)}")

    log = rebalance_log.copy()
    log["decision_date"] = pd.to_datetime(log["decision_date"])
    log["effective_date"] = pd.to_datetime(log["effective_date"])
    for column in [
        "target_crypto_budget",
        "target_equity_budget",
        "post_rebalance_crypto_weight",
        "post_rebalance_equity_weight",
    ]:
        log[column] = pd.to_numeric(log[column], errors="raise")
    clean_returns = _validate_backtest_returns(returns)
    weights = fund_weights.copy()
    weights["effective_date"] = pd.to_datetime(weights["effective_date"])
    weights["portfolio_weight"] = pd.to_numeric(
        weights["portfolio_weight"],
        errors="raise",
    )
    crypto_assets = list(crypto_assets)
    if not {"Equity", "Crypto"}.issubset(set(weights["asset_class"].astype(str))):
        raise ValueError("fund_weights asset_class must contain Equity and Crypto")

    pretrade_records = []
    assets = pd.Index(weights["ticker"].drop_duplicates())
    for fund, fund_weights_panel in weights.groupby("fund", sort=True):
        target_panel = (
            fund_weights_panel.pivot_table(
                index="effective_date",
                columns="ticker",
                values="portfolio_weight",
                aggfunc="first",
            )
            .reindex(columns=assets)
            .fillna(0.0)
            .sort_index()
        )
        current_weights = None
        target_dates = set(target_panel.index)
        live_dates = clean_returns.index[clean_returns.index >= target_panel.index.min()]
        for date in live_dates:
            if date in target_dates:
                target = target_panel.loc[date].astype(float)
                pretrade = target if current_weights is None else current_weights.copy()
                pretrade_crypto = float(pretrade.loc[crypto_assets].sum())
                pretrade_records.append(
                    {
                        "effective_date": date,
                        "fund": fund,
                        "pretrade_crypto_weight": pretrade_crypto,
                        "pretrade_equity_weight": 1.0 - pretrade_crypto,
                    }
                )
                current_weights = target.copy()
            if current_weights is not None:
                current_weights = _drift_weights_for_stage7_allocation(
                    current_weights,
                    clean_returns.loc[date],
                    date,
                )
    pretrade = pd.DataFrame(pretrade_records)
    history = log.merge(
        pretrade,
        on=["fund", "effective_date"],
        how="left",
        validate="one_to_one",
    )

    budgets = adaptive_budget_history.copy()
    budgets["decision_date"] = pd.to_datetime(budgets["date"])
    budget_columns = [
        "decision_date",
        "profile",
        "rolling_tail_risk",
        "risk_score",
        "raw_crypto_budget",
        "raw_equity_budget",
        "previous_applied_crypto_budget",
        "applied_crypto_budget",
        "applied_equity_budget",
        "budget_change",
        "no_adjustment_triggered",
        "monthly_limit_triggered",
    ]
    history = history.merge(
        budgets[budget_columns],
        on=["decision_date", "profile"],
        how="left",
        validate="many_to_one",
    )

    turnover = fund_turnover.copy()
    turnover["effective_date"] = pd.to_datetime(turnover["effective_date"])
    turnover["turnover"] = pd.to_numeric(turnover["turnover"], errors="coerce")
    history = history.merge(
        turnover[["effective_date", "fund", "turnover"]],
        on=["fund", "effective_date"],
        how="left",
        validate="one_to_one",
    )

    history = history.sort_values(["fund", "effective_date"]).reset_index(drop=True)
    trigger_columns = ["no_adjustment_triggered", "monthly_limit_triggered"]
    history[trigger_columns] = history[trigger_columns].astype("object")
    history["previous_target_crypto_budget"] = history.groupby("fund")[
        "target_crypto_budget"
    ].shift(1)
    history["market_drift_from_previous_target"] = (
        history["pretrade_crypto_weight"] - history["previous_target_crypto_budget"]
    )
    history["crypto_rebalance_change"] = (
        history["post_rebalance_crypto_weight"] - history["pretrade_crypto_weight"]
    )

    fixed = history["fund_type"].astype(str).eq("fixed")
    history["target_budget_change"] = history["budget_change"]
    history.loc[fixed, "target_budget_change"] = (
        history.loc[fixed, "target_crypto_budget"]
        - history.loc[fixed, "previous_target_crypto_budget"]
    )
    adaptive_budget_columns = [
        "rolling_tail_risk",
        "risk_score",
        "raw_crypto_budget",
        "raw_equity_budget",
        "previous_applied_crypto_budget",
        "applied_crypto_budget",
        "applied_equity_budget",
        "budget_change",
    ]
    history.loc[fixed, adaptive_budget_columns] = np.nan
    history.loc[fixed, trigger_columns] = np.nan

    return history[STAGE_7_ALLOCATION_HISTORY_COLUMNS].sort_values(
        ["fund", "effective_date"]
    ).reset_index(drop=True)


# ============================================================================
# STAGE 7.8: AUTOMATIC EXPLAINABLE REBALANCE NARRATIVES
# Purpose:
# Convert the monthly Stage 7.7 allocation history into deterministic technical
# and user-facing rebalance explanations.
# ============================================================================
STAGE_7_REBALANCE_EXPLANATION_COLUMNS = [
    "decision_date",
    "effective_date",
    "fund",
    "profile",
    "fund_type",
    "rolling_tail_risk",
    "previous_rolling_tail_risk",
    "risk_score",
    "previous_risk_score",
    "raw_crypto_budget",
    "applied_crypto_budget",
    "target_crypto_budget",
    "target_equity_budget",
    "post_rebalance_crypto_weight",
    "post_rebalance_equity_weight",
    "no_adjustment_triggered",
    "monthly_limit_triggered",
    "technical_explanation",
    "user_explanation",
]


def build_stage7_rebalance_explanations(
    allocation_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build deterministic Stage 7.8 technical and user rebalance explanations."""
    required = {
        "decision_date",
        "effective_date",
        "fund",
        "profile",
        "fund_type",
        "rolling_tail_risk",
        "risk_score",
        "raw_crypto_budget",
        "applied_crypto_budget",
        "target_crypto_budget",
        "target_equity_budget",
        "post_rebalance_crypto_weight",
        "post_rebalance_equity_weight",
        "no_adjustment_triggered",
        "monthly_limit_triggered",
        "target_budget_change",
    }
    missing = required.difference(allocation_history.columns)
    if missing:
        raise ValueError(f"allocation_history is missing columns: {sorted(missing)}")

    data = allocation_history.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["effective_date"] = pd.to_datetime(data["effective_date"])
    data = data.sort_values(["fund", "decision_date"]).reset_index(drop=True)
    valid_funds = {config["fund"] for config in stage7_two_sleeve_fund_configs()}
    if set(data["fund"].astype(str)).difference(valid_funds):
        raise ValueError("allocation_history contains unknown Stage 7 fund labels")

    for column in [
        "rolling_tail_risk",
        "risk_score",
        "raw_crypto_budget",
        "applied_crypto_budget",
        "target_crypto_budget",
        "target_equity_budget",
        "post_rebalance_crypto_weight",
        "post_rebalance_equity_weight",
        "target_budget_change",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    market_risk = data.groupby(["decision_date", "profile"])[
        ["rolling_tail_risk", "risk_score"]
    ].transform("max")
    data[["rolling_tail_risk", "risk_score"]] = data[
        ["rolling_tail_risk", "risk_score"]
    ].fillna(market_risk)
    data["previous_rolling_tail_risk"] = data.groupby("fund")[
        "rolling_tail_risk"
    ].shift(1)
    data["previous_risk_score"] = data.groupby("fund")["risk_score"].shift(1)

    adaptive = data["fund_type"].astype(str).eq("adaptive")
    fixed = data["fund_type"].astype(str).eq("fixed")
    if data.loc[adaptive, ["raw_crypto_budget", "applied_crypto_budget"]].isna().any().any():
        raise ValueError("Adaptive rows require Raw and Applied Crypto Budgets")
    if data.loc[fixed, ["raw_crypto_budget", "applied_crypto_budget"]].notna().any().any():
        raise ValueError("Fixed rows must not contain Raw or Applied Adaptive Budgets")

    def pct(value, decimals=1):
        return "unavailable" if pd.isna(value) else f"{float(value):.{decimals}%}"

    def score(value):
        return "unavailable" if pd.isna(value) else f"{float(value):.1f}"

    def direction(current, previous, up, down, tolerance):
        if pd.isna(current) or pd.isna(previous):
            return None
        change = float(current) - float(previous)
        if abs(change) < tolerance:
            return "was broadly unchanged"
        return up if change > 0 else down

    def budget_movement(change, tolerance=1e-6):
        if pd.isna(change):
            return None
        if float(change) < -tolerance:
            return "decreased"
        if float(change) > tolerance:
            return "increased"
        return "unchanged"

    def flag(value):
        return bool(value) if pd.notna(value) else False

    technical = []
    user = []
    for row in data.itertuples(index=False):
        profile = str(row.profile).title()
        fund_type = str(row.fund_type).title()
        fund_label = f"{profile} {fund_type}"
        crypto = pct(row.target_crypto_budget)
        equity = pct(row.target_equity_budget)
        no_adjust = flag(row.no_adjustment_triggered)
        monthly_limit = flag(row.monthly_limit_triggered)
        movement = budget_movement(row.target_budget_change)

        if row.fund_type == "adaptive":
            if pd.isna(row.previous_risk_score) or pd.isna(row.previous_rolling_tail_risk):
                tech = (
                    f"This is the first live rebalance for the {fund_label} fund. "
                    f"The current Crypto risk score was {score(row.risk_score)} and "
                    f"the rolling tail-loss estimate was {pct(row.rolling_tail_risk, 2)}. "
                    f"For the {profile} profile, the Raw Crypto Budget was "
                    f"{pct(row.raw_crypto_budget)}; after smoothing, the Applied "
                    f"Crypto Budget was {pct(row.applied_crypto_budget)}. "
                )
                simple = (
                    f"PulseAlloc used the current Crypto risk reading for the "
                    f"{profile} portfolio and set the target after safeguards. "
                )
            else:
                risk_direction = direction(
                    row.risk_score,
                    row.previous_risk_score,
                    "moved up",
                    "moved down",
                    0.05,
                )
                tail_direction = direction(
                    row.rolling_tail_risk,
                    row.previous_rolling_tail_risk,
                    "rose",
                    "fell",
                    0.00005,
                )
                tech = (
                    f"The Crypto risk score {risk_direction} from "
                    f"{score(row.previous_risk_score)} to {score(row.risk_score)}, "
                    f"while the rolling tail-loss estimate {tail_direction} from "
                    f"{pct(row.previous_rolling_tail_risk, 2)} to "
                    f"{pct(row.rolling_tail_risk, 2)}. For the {profile} profile, "
                    f"the Raw Crypto Budget was {pct(row.raw_crypto_budget)}. "
                    f"After smoothing, the Applied Crypto Budget was "
                    f"{pct(row.applied_crypto_budget)}. "
                )
                risk_word = (
                    "rose"
                    if tail_direction == "rose"
                    else "fell"
                    if tail_direction == "fell"
                    else "was broadly unchanged"
                )
                if no_adjust:
                    tech += (
                        "The proposed Budget change was below the "
                        "1-percentage-point threshold, so the previous Applied "
                        "Crypto Budget was maintained. "
                    )
                    simple = (
                        f"Recent Crypto downside risk {risk_word}. The change was "
                        f"small, so the {profile} portfolio's target Crypto "
                        "allocation was kept unchanged. "
                    )
                elif monthly_limit:
                    if movement == "increased":
                        tech += (
                            "The Applied Crypto Budget increased toward the Raw "
                            "Budget, but the monthly adjustment was limited to "
                            "5 percentage points. "
                        )
                        simple = (
                            f"Recent Crypto downside risk {risk_word}. The target "
                            "Crypto allocation was increased gradually to avoid a "
                            "sudden portfolio change. "
                        )
                    elif movement == "decreased":
                        tech += (
                            "The Applied Crypto Budget decreased toward the Raw "
                            "Budget, but the monthly adjustment was limited to "
                            "5 percentage points. "
                        )
                        simple = (
                            f"Recent Crypto downside risk {risk_word}. The target "
                            "Crypto allocation was reduced gradually to avoid a "
                            "sudden portfolio change. "
                        )
                    else:
                        tech += (
                            "The Applied Crypto Budget moved toward the Raw Budget, "
                            "but the monthly adjustment was limited to "
                            "5 percentage points. "
                        )
                        simple = (
                            f"Recent Crypto downside risk {risk_word}. The target "
                            "Crypto allocation moved gradually to avoid a sudden "
                            "portfolio change. "
                        )
                elif movement == "increased":
                    simple = (
                        f"Recent Crypto downside risk {risk_word}. PulseAlloc "
                        f"increased the {profile} portfolio's target Crypto "
                        "allocation. "
                    )
                elif movement == "decreased":
                    simple = (
                        f"Recent Crypto downside risk {risk_word}. PulseAlloc "
                        f"reduced the {profile} portfolio's target Crypto "
                        "allocation. "
                    )
                else:
                    simple = (
                        f"Recent Crypto downside risk {risk_word}, but the "
                        f"{profile} portfolio's target Crypto allocation was kept "
                        "unchanged. "
                    )
        else:
            risk_text = (
                f"The Crypto risk score was {score(row.risk_score)}, but "
                if pd.notna(row.risk_score)
                else "The current Crypto risk score was unavailable, but "
            )
            tech = (
                f"{risk_text}the {fund_label} fund maintained its pre-set "
                f"{crypto} Crypto Budget. "
            )
            simple = (
                f"This Fixed fund keeps {crypto} in Crypto regardless of changes "
                f"in market risk. "
            )

        tech += (
            f"The portfolio was rebalanced for the next holding period to "
            f"{crypto} Crypto and {equity} Equity, while Minimum-CVaR selected "
            f"the assets within each sleeve."
        )
        simple += (
            f"The next-period allocation is {crypto} Crypto and {equity} stocks."
        )
        technical.append(" ".join(tech.split()))
        user.append(" ".join(simple.split()))

    data["technical_explanation"] = technical
    data["user_explanation"] = user
    return data[STAGE_7_REBALANCE_EXPLANATION_COLUMNS].reset_index(drop=True)


# ============================================================================
# STAGE 7.9-7.10: FIXED/ADAPTIVE AND PERSONALISATION COMPARISON
# Purpose:
# Summarise official Stage 7 fund performance and compare Fixed versus Adaptive
# outcomes plus the three Adaptive personalised Budget profiles.
# ============================================================================
STAGE_7_PERFORMANCE_METRIC_COLUMNS = [
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "realised_95_cvar",
    "worst_10_day_return",
    "ending_growth_of_1",
    "average_target_crypto_budget",
    "average_pretrade_crypto_weight",
    "average_turnover",
]


def calculate_stage7_fund_performance_metrics(
    fund_returns: pd.DataFrame,
    fund_turnover: pd.DataFrame,
    allocation_history: pd.DataFrame,
    periods_per_year: int = 252,
    return_column: str = "gross_return",
    return_values_are_growth_factors: bool = True,
) -> pd.DataFrame:
    """Calculate Stage 7.9 performance metrics for the six official funds."""
    required_returns = {"date", "fund", "profile", "fund_type", return_column}
    required_turnover = {"fund", "turnover"}
    required_allocation = {
        "fund",
        "target_crypto_budget",
        "pretrade_crypto_weight",
    }
    missing_returns = required_returns.difference(fund_returns.columns)
    missing_turnover = required_turnover.difference(fund_turnover.columns)
    missing_allocation = required_allocation.difference(allocation_history.columns)
    if missing_returns:
        raise ValueError(f"fund_returns is missing columns: {sorted(missing_returns)}")
    if missing_turnover:
        raise ValueError(f"fund_turnover is missing columns: {sorted(missing_turnover)}")
    if missing_allocation:
        raise ValueError(
            f"allocation_history is missing columns: {sorted(missing_allocation)}"
        )

    returns = fund_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns[return_column] = pd.to_numeric(returns[return_column], errors="raise")
    expected_funds = {config["fund"] for config in stage7_two_sleeve_fund_configs()}
    observed_funds = set(returns["fund"].astype(str))
    if observed_funds != expected_funds:
        raise ValueError("fund_returns must contain the six official Stage 7 funds")

    date_sets = returns.groupby("fund")["date"].apply(lambda series: tuple(series))
    if len(set(date_sets)) != 1:
        raise ValueError("all Stage 7 funds must use identical return dates")

    turnover = fund_turnover.copy()
    turnover["turnover"] = pd.to_numeric(turnover["turnover"], errors="coerce")
    allocation = allocation_history.copy()
    allocation["target_crypto_budget"] = pd.to_numeric(
        allocation["target_crypto_budget"],
        errors="raise",
    )
    allocation["pretrade_crypto_weight"] = pd.to_numeric(
        allocation["pretrade_crypto_weight"],
        errors="raise",
    )

    rows = []
    for fund, panel in returns.sort_values(["fund", "date"]).groupby("fund", sort=True):
        profile = str(panel["profile"].iloc[0])
        fund_type = str(panel["fund_type"].iloc[0])
        daily_returns = panel.set_index("date")[return_column].astype(float).sort_index()
        if return_values_are_growth_factors:
            daily_returns = daily_returns - 1.0
        metrics = performance_metrics(
            daily_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=0.0,
        )
        tail_cutoff = daily_returns.quantile(0.05)
        worst_tail = daily_returns.loc[daily_returns <= tail_cutoff]
        rolling_10_day = (1.0 + daily_returns).rolling(10).apply(np.prod, raw=True) - 1.0
        fund_allocation = allocation.loc[allocation["fund"].astype(str).eq(str(fund))]
        fund_turnover = turnover.loc[turnover["fund"].astype(str).eq(str(fund))]
        rows.append(
            {
                "fund": fund,
                "profile": profile,
                "fund_type": fund_type,
                "annualised_return": metrics["annualised_return"],
                "annualised_volatility": metrics["annualised_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "realised_95_cvar": float(worst_tail.mean()),
                "worst_10_day_return": float(rolling_10_day.min()),
                "ending_growth_of_1": metrics["final_growth_of_1"],
                "average_target_crypto_budget": float(
                    fund_allocation["target_crypto_budget"].mean()
                ),
                "average_pretrade_crypto_weight": float(
                    fund_allocation["pretrade_crypto_weight"].mean()
                ),
                "average_turnover": float(fund_turnover["turnover"].dropna().mean()),
            }
        )

    profile_order = {"conservative": 0, "balanced": 1, "growth": 2}
    type_order = {"fixed": 0, "adaptive": 1}
    metrics_table = pd.DataFrame(rows)
    return metrics_table.sort_values(
        ["profile", "fund_type"],
        key=lambda column: column.map(
            profile_order if column.name == "profile" else type_order
        ),
    ).reset_index(drop=True)


def apply_stage7_transaction_costs(
    fund_returns: pd.DataFrame,
    fund_turnover: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Apply proportional transaction costs to saved Stage 7 gross fund returns."""
    required_returns = {"date", "fund", "profile", "fund_type", "gross_return"}
    required_turnover = {"effective_date", "fund", "turnover"}
    missing_returns = required_returns.difference(fund_returns.columns)
    missing_turnover = required_turnover.difference(fund_turnover.columns)
    if missing_returns:
        raise ValueError(f"fund_returns is missing columns: {sorted(missing_returns)}")
    if missing_turnover:
        raise ValueError(f"fund_turnover is missing columns: {sorted(missing_turnover)}")
    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate must be non-negative")

    returns = fund_returns.copy()
    original_row_count = len(returns)
    returns["date"] = pd.to_datetime(returns["date"])
    returns["gross_return"] = pd.to_numeric(returns["gross_return"], errors="raise") - 1.0

    expected_funds = {config["fund"] for config in stage7_two_sleeve_fund_configs()}
    if set(returns["fund"].astype(str)) != expected_funds:
        raise ValueError("fund_returns must contain the six official Stage 7 funds")

    turnover = fund_turnover.copy()
    turnover["effective_date"] = pd.to_datetime(turnover["effective_date"])
    turnover["turnover"] = pd.to_numeric(turnover["turnover"], errors="coerce")
    observed_turnover_funds = set(turnover["fund"].astype(str))
    if observed_turnover_funds != expected_funds:
        raise ValueError("fund_turnover must contain the six official Stage 7 funds")
    if turnover["turnover"].dropna().lt(0).any():
        raise ValueError("turnover must be non-negative")

    turnover_daily = turnover.rename(columns={"effective_date": "date"})[
        ["fund", "date", "turnover"]
    ]
    matched = turnover_daily.merge(
        returns[["fund", "date"]],
        on=["fund", "date"],
        how="left",
        indicator=True,
    )
    if matched["_merge"].ne("both").any():
        raise ValueError("all Stage 7 turnover effective dates must match return dates")

    out = returns.merge(turnover_daily, on=["fund", "date"], how="left")
    out["turnover"] = out["turnover"].fillna(0.0)
    out["transaction_cost_rate"] = float(transaction_cost_rate)
    out["transaction_cost"] = out["turnover"] * float(transaction_cost_rate)
    out["net_return"] = out["gross_return"] - out["transaction_cost"]
    out = out.sort_values(["fund", "date"]).reset_index(drop=True)
    grouped = out.groupby("fund", sort=False)
    out["gross_growth_of_1"] = grouped["gross_return"].transform(
        lambda returns_: (1.0 + returns_).cumprod()
    )
    out["net_growth_of_1"] = grouped["net_return"].transform(
        lambda returns_: (1.0 + returns_).cumprod()
    )
    out["cumulative_transaction_cost"] = grouped["transaction_cost"].cumsum()
    if len(out) != original_row_count:
        raise ValueError("daily return row count changed after applying costs")
    return out[
        [
            "date",
            "fund",
            "profile",
            "fund_type",
            "gross_return",
            "turnover",
            "transaction_cost_rate",
            "transaction_cost",
            "net_return",
            "gross_growth_of_1",
            "net_growth_of_1",
            "cumulative_transaction_cost",
        ]
    ]


def build_stage7_raw_adaptive_results(
    fund_weights: pd.DataFrame,
    raw_budgets: pd.DataFrame,
    smoothed_net_returns: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Build diagnostic Raw Adaptive fund returns from saved Stage 7 sleeve weights."""
    profiles = ["conservative", "balanced", "growth"]
    required_weights = {
        "decision_date",
        "effective_date",
        "fund",
        "profile",
        "ticker",
        "asset_class",
        "sleeve_weight",
    }
    required_budgets = {"date", "profile", "raw_crypto_budget"}
    required_smoothed = {"date", "fund", "profile"}
    missing_weights = required_weights.difference(fund_weights.columns)
    missing_budgets = required_budgets.difference(raw_budgets.columns)
    missing_smoothed = required_smoothed.difference(smoothed_net_returns.columns)
    if missing_weights:
        raise ValueError(f"fund_weights is missing columns: {sorted(missing_weights)}")
    if missing_budgets:
        raise ValueError(f"raw_budgets is missing columns: {sorted(missing_budgets)}")
    if missing_smoothed:
        raise ValueError(
            f"smoothed_net_returns is missing columns: {sorted(missing_smoothed)}"
        )
    if transaction_cost_rate != 0.001:
        raise ValueError("Stage 7.11 transaction_cost_rate must equal 0.001")

    clean_returns = _validate_backtest_returns(returns)
    weights = fund_weights.copy()
    weights["decision_date"] = pd.to_datetime(weights["decision_date"])
    weights["effective_date"] = pd.to_datetime(weights["effective_date"])
    weights["sleeve_weight"] = pd.to_numeric(weights["sleeve_weight"], errors="raise")
    if not set(profiles).issubset(set(weights["profile"].astype(str))):
        raise ValueError("fund_weights must contain the three Raw Adaptive profiles")

    sleeve_check = weights.groupby(["decision_date", "effective_date", "ticker"])[
        "sleeve_weight"
    ].agg(lambda values: float(values.max() - values.min()))
    if sleeve_check.gt(1e-12).any():
        raise ValueError("saved sleeve weights must be identical across profiles")
    sleeves = (
        weights.sort_values(["decision_date", "effective_date", "ticker"])
        .drop_duplicates(["decision_date", "effective_date", "ticker"])
        [
            [
                "decision_date",
                "effective_date",
                "ticker",
                "asset_class",
                "sleeve_weight",
            ]
        ]
    )
    if set(sleeves["ticker"]).difference(clean_returns.columns):
        raise ValueError("returns must contain every saved Stage 7 sleeve asset")

    budgets = raw_budgets.copy()
    budgets["decision_date"] = pd.to_datetime(budgets["date"])
    budgets["raw_crypto_budget"] = pd.to_numeric(
        budgets["raw_crypto_budget"],
        errors="coerce",
    )
    budget_lookup = budgets.pivot(
        index="decision_date",
        columns="profile",
        values="raw_crypto_budget",
    )
    smoothed = smoothed_net_returns.copy()
    smoothed["date"] = pd.to_datetime(smoothed["date"])

    target_dates = sleeves[["decision_date", "effective_date"]].drop_duplicates()
    missing_budget_dates = sorted(
        set(target_dates["decision_date"]).difference(budget_lookup.dropna().index)
    )
    if missing_budget_dates:
        raise ValueError("Raw Budgets are not available for every decision date")

    records = []
    assets = pd.Index(sleeves["ticker"].drop_duplicates())
    for profile in profiles:
        smoothed_fund = f"{profile}_adaptive"
        raw_fund = f"{profile}_raw_adaptive"
        official_dates = (
            smoothed.loc[smoothed["fund"].astype(str).eq(smoothed_fund), "date"]
            .sort_values()
            .tolist()
        )
        if not official_dates:
            raise ValueError(f"missing Smoothed Adaptive dates for {profile}")

        target_rows = []
        for (decision_date, effective_date), panel in sleeves.groupby(
            ["decision_date", "effective_date"],
            sort=True,
        ):
            raw_crypto_budget = float(budget_lookup.loc[decision_date, profile])
            raw_equity_budget = 1.0 - raw_crypto_budget
            scale = panel["asset_class"].astype(str).eq("Crypto").map(
                {True: raw_crypto_budget, False: raw_equity_budget}
            )
            target = pd.Series(
                panel["sleeve_weight"].to_numpy(dtype=float) * scale.to_numpy(dtype=float),
                index=panel["ticker"],
                name=effective_date,
            ).reindex(assets).fillna(0.0)
            if not np.isclose(float(target.sum()), 1.0, atol=1e-10):
                raise ValueError("Raw Adaptive target weights must sum to one")
            target_rows.append(
                {
                    "decision_date": decision_date,
                    "effective_date": effective_date,
                    "target_crypto_budget": raw_crypto_budget,
                    "target_equity_budget": raw_equity_budget,
                    "weights": target,
                }
            )

        target_info = pd.DataFrame(target_rows).set_index("effective_date").sort_index()
        official_index = pd.DatetimeIndex(official_dates)
        if not target_info.index.isin(official_index).all():
            raise ValueError("Raw and Smoothed funds must use identical rebalance dates")
        if official_index.min() != target_info.index.min():
            raise ValueError("Raw and Smoothed funds must use identical start dates")

        current_weights = None
        current_crypto_budget = np.nan
        current_equity_budget = np.nan
        for date in official_index:
            turnover = 0.0
            if date in target_info.index:
                row = target_info.loc[date]
                target = row["weights"].astype(float)
                turnover = (
                    0.0
                    if current_weights is None
                    else 0.5 * float((target - current_weights).abs().sum())
                )
                current_weights = target.copy()
                current_crypto_budget = float(row["target_crypto_budget"])
                current_equity_budget = float(row["target_equity_budget"])

            held_assets = current_weights.index[
                current_weights.gt(_WEIGHT_BOUND_TOLERANCE)
            ]
            day_returns = clean_returns.loc[date, held_assets]
            if day_returns.isna().any():
                missing_live = day_returns.index[day_returns.isna()].tolist()
                raise ValueError(f"missing live returns on {date.date()}: {missing_live}")
            gross_return = float((day_returns * current_weights.loc[held_assets]).sum())
            transaction_cost = turnover * float(transaction_cost_rate)
            records.append(
                {
                    "date": date,
                    "fund": raw_fund,
                    "profile": profile,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "net_return": gross_return - transaction_cost,
                    "target_crypto_budget": current_crypto_budget,
                    "target_equity_budget": current_equity_budget,
                }
            )
            current_weights = _drift_weights_for_stage7_allocation(
                current_weights,
                clean_returns.loc[date],
                date,
            )

    out = pd.DataFrame(records)
    if set(out["profile"]) != set(profiles):
        raise ValueError("exactly three Raw Adaptive profiles must be created")
    return out.sort_values(["fund", "date"]).reset_index(drop=True)


def build_stage7_smoothing_comparison(
    raw_results: pd.DataFrame,
    smoothed_net_returns: pd.DataFrame,
    adaptive_budget_history: pd.DataFrame,
    smoothed_turnover: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare Raw versus Smoothed Adaptive budget stability and net performance."""
    profiles = ["conservative", "balanced", "growth"]
    if transaction_cost_rate != 0.001:
        raise ValueError("Stage 7.11 transaction_cost_rate must equal 0.001")

    def _net_metrics(panel: pd.DataFrame) -> dict:
        series = panel.set_index("date")["net_return"].astype(float).sort_index()
        metrics = performance_metrics(series, periods_per_year=periods_per_year)
        tail_cutoff = series.quantile(0.05)
        worst_tail = series.loc[series <= tail_cutoff]
        rolling_10_day = (1.0 + series).rolling(10).apply(np.prod, raw=True) - 1.0
        return {
            "net_annualised_return": metrics["annualised_return"],
            "net_annualised_volatility": metrics["annualised_volatility"],
            "net_sharpe_ratio": metrics["sharpe_ratio"],
            "net_maximum_drawdown": metrics["maximum_drawdown"],
            "net_realised_95_cvar": float(worst_tail.mean()),
            "net_worst_10_day_return": float(rolling_10_day.min()),
            "net_ending_growth_of_1": metrics["final_growth_of_1"],
        }

    raw = raw_results.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    smoothed = smoothed_net_returns.copy()
    smoothed["date"] = pd.to_datetime(smoothed["date"])
    budgets = adaptive_budget_history.copy()
    budgets["date"] = pd.to_datetime(budgets["date"])
    turnover = smoothed_turnover.copy()
    turnover["effective_date"] = pd.to_datetime(turnover["effective_date"])
    turnover["decision_date"] = pd.to_datetime(turnover["decision_date"])
    turnover["turnover"] = pd.to_numeric(turnover["turnover"], errors="coerce")

    rows = []
    for profile in profiles:
        raw_fund = f"{profile}_raw_adaptive"
        smoothed_fund = f"{profile}_adaptive"
        raw_panel = raw.loc[raw["fund"].astype(str).eq(raw_fund)].copy()
        smoothed_panel = smoothed.loc[
            smoothed["fund"].astype(str).eq(smoothed_fund)
        ].copy()
        if tuple(raw_panel["date"]) != tuple(smoothed_panel["date"]):
            raise ValueError("Raw and Smoothed funds must use identical dates")

        fund_turnover = turnover.loc[turnover["fund"].astype(str).eq(smoothed_fund)]
        decision_dates = pd.DatetimeIndex(fund_turnover["decision_date"].sort_values())
        budget_panel = budgets.loc[
            budgets["profile"].astype(str).eq(profile)
            & budgets["date"].isin(decision_dates)
        ].sort_values("date")
        if len(budget_panel) != len(decision_dates):
            raise ValueError("Raw and Smoothed Budget dates must match")

        raw_budget_change = budget_panel["raw_crypto_budget"].astype(float).diff().abs()
        smoothed_budget_change = (
            budget_panel["applied_crypto_budget"].astype(float).diff().abs()
        )
        effective_dates = set(fund_turnover["effective_date"])
        raw_rebalance_turnover = raw_panel.loc[
            raw_panel["date"].isin(effective_dates),
            ["date", "turnover"],
        ].copy()
        raw_rebalance_turnover.loc[
            raw_rebalance_turnover["date"].eq(min(effective_dates)),
            "turnover",
        ] = np.nan

        raw_values = {
            "average_absolute_budget_change": float(raw_budget_change.mean()),
            "maximum_absolute_budget_change": float(raw_budget_change.max()),
            "average_turnover": float(raw_rebalance_turnover["turnover"].mean()),
            "total_turnover": float(raw_panel["turnover"].sum()),
            "total_transaction_cost": float(raw_panel["transaction_cost"].sum()),
            **_net_metrics(raw_panel),
        }
        smoothed_values = {
            "average_absolute_budget_change": float(smoothed_budget_change.mean()),
            "maximum_absolute_budget_change": float(smoothed_budget_change.max()),
            "average_turnover": float(fund_turnover["turnover"].mean()),
            "total_turnover": float(fund_turnover["turnover"].sum()),
            "total_transaction_cost": float(
                smoothed_panel["transaction_cost"].astype(float).sum()
            ),
            **_net_metrics(smoothed_panel),
        }
        row = {
            "profile": profile,
            "no_adjustment_triggers": int(
                budget_panel["no_adjustment_triggered"].fillna(False).astype(bool).sum()
            ),
            "monthly_limit_triggers": int(
                budget_panel["monthly_limit_triggered"].fillna(False).astype(bool).sum()
            ),
        }
        for metric in [
            "average_absolute_budget_change",
            "maximum_absolute_budget_change",
            "average_turnover",
            "total_turnover",
            "total_transaction_cost",
            "net_annualised_return",
            "net_annualised_volatility",
            "net_sharpe_ratio",
            "net_maximum_drawdown",
            "net_realised_95_cvar",
            "net_worst_10_day_return",
            "net_ending_growth_of_1",
        ]:
            row[f"raw_{metric}"] = raw_values[metric]
            row[f"smoothed_{metric}"] = smoothed_values[metric]
            row[f"difference_{metric}"] = smoothed_values[metric] - raw_values[metric]
        rows.append(row)

    return pd.DataFrame(rows)


def build_stage7_comparison_tables(
    performance_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Stage 7.9 Fixed-vs-Adaptive and Stage 7.10 personalisation tables."""
    required = {"fund", "profile", "fund_type", *STAGE_7_PERFORMANCE_METRIC_COLUMNS}
    missing = required.difference(performance_table.columns)
    if missing:
        raise ValueError(f"performance_table is missing columns: {sorted(missing)}")

    metrics = performance_table.copy()
    profiles = ["conservative", "balanced", "growth"]
    comparison_rows = []
    for profile in profiles:
        fixed = metrics.loc[
            metrics["profile"].astype(str).eq(profile)
            & metrics["fund_type"].astype(str).eq("fixed")
        ]
        adaptive = metrics.loc[
            metrics["profile"].astype(str).eq(profile)
            & metrics["fund_type"].astype(str).eq("adaptive")
        ]
        if len(fixed) != 1 or len(adaptive) != 1:
            raise ValueError(f"missing Fixed/Adaptive pair for {profile}")
        row = {"profile": profile}
        fixed_row = fixed.iloc[0]
        adaptive_row = adaptive.iloc[0]
        for metric in STAGE_7_PERFORMANCE_METRIC_COLUMNS:
            row[f"fixed_{metric}"] = fixed_row[metric]
            row[f"adaptive_{metric}"] = adaptive_row[metric]
            row[f"difference_{metric}"] = adaptive_row[metric] - fixed_row[metric]
        comparison_rows.append(row)

    personalisation = metrics.loc[
        metrics["fund_type"].astype(str).eq("adaptive"),
        [
            "fund",
            "profile",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "realised_95_cvar",
            "worst_10_day_return",
            "ending_growth_of_1",
            "average_target_crypto_budget",
            "average_pretrade_crypto_weight",
            "average_turnover",
        ],
    ].copy()
    if set(personalisation["profile"]) != set(profiles):
        raise ValueError("personalisation table requires the three Adaptive profiles")

    personalisation["crypto_budget_rank"] = personalisation[
        "average_target_crypto_budget"
    ].rank(method="dense", ascending=True)
    personalisation["volatility_rank"] = personalisation[
        "annualised_volatility"
    ].rank(method="dense", ascending=True)
    personalisation["drawdown_risk_rank"] = personalisation[
        "maximum_drawdown"
    ].abs().rank(method="dense", ascending=True)

    profile_order = {"conservative": 0, "balanced": 1, "growth": 2}
    fixed_vs_adaptive = pd.DataFrame(comparison_rows)
    personalisation = personalisation.sort_values(
        "profile",
        key=lambda column: column.map(profile_order),
    ).reset_index(drop=True)
    return fixed_vs_adaptive, personalisation
