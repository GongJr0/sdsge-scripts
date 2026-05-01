from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f, t
from sklearn.metrics import r2_score


@dataclass
class OLSResult:
    X: np.ndarray
    y_name: str
    x_names: list[str]
    intercept: bool
    beta: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    r2: float
    cov_beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    nobs: int
    df_resid: int

    def coefficient_table(self, *, drop_intercept: bool = True, round_to: int | None = None) -> pd.DataFrame:
        names = ["const", *self.x_names] if self.intercept else self.x_names
        df = pd.DataFrame(
            {
                "coef": self.beta,
                "std_error": self.se,
                "t_stat": self.t_stat,
                "p_value": self.p_value,
            },
            index=names,
        )
        df.insert(0, "y", self.y_name)
        df.insert(1, "r2", self.r2)
        if drop_intercept and self.intercept:
            df = df.drop(index="const")
        if round_to is not None:
            df = df.round(round_to)
        return df


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _safe_std(x: np.ndarray, *, ddof: int = 0) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.std(x, ddof=ddof))


def _zscore(x: np.ndarray, *, ddof: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, ddof=ddof, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def ols(y: np.ndarray, X: np.ndarray, *, intercept: bool = True, y_name: str = "y", x_names: Sequence[str] | None = None) -> OLSResult:
    y = np.asarray(y, dtype=float).reshape(-1)
    X = _as_2d(np.asarray(X, dtype=float))

    if x_names is None:
        x_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        x_names = list(x_names)

    X_reg = np.column_stack([np.ones(X.shape[0]), X]) if intercept else X
    beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
    fitted = X_reg @ beta
    resid = y - fitted
    r2 = r2_score(y, fitted)

    n, k = X_reg.shape
    df_resid = n - k
    sigma2 = np.sum(resid ** 2) / df_resid if df_resid > 0 else np.nan
    try:
        xtx_inv = np.linalg.inv(X_reg.T @ X_reg)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X_reg.T @ X_reg)
    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.diag(var_beta))
    t_stat = beta / se
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=df_resid)) if df_resid > 0 else np.full_like(beta, np.nan)

    return OLSResult(
        X=X_reg,
        y_name=y_name,
        x_names=x_names,
        intercept=intercept,
        beta=beta,
        fitted=fitted,
        resid=resid,
        r2=r2,
        cov_beta=var_beta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        nobs=n,
        df_resid=df_resid,
    )


def standardized_slope_from_stds(beta_slope: float, x: np.ndarray, y: np.ndarray) -> float:
    std_x = _safe_std(x)
    std_y = _safe_std(y)
    if std_y == 0:
        return np.nan
    return float(beta_slope) * (std_x / std_y)


def standardized_slope_from_model(
    *,
    raw_model: OLSResult,
    x: np.ndarray,
    y: np.ndarray,
    intercept: bool,
    z_score_standardization: bool = False,
) -> float:
    slope_idx = 1 if intercept else 0
    if not z_score_standardization:
        return standardized_slope_from_stds(raw_model.beta[slope_idx], x, y)

    z_model = ols(
        y=_zscore(np.asarray(y, dtype=float).reshape(-1)),
        X=_zscore(_as_2d(x)),
        intercept=intercept,
        y_name=raw_model.y_name,
        x_names=raw_model.x_names,
    )
    return float(z_model.beta[slope_idx])


@dataclass
class OrthogonalizationBundle:
    residuals: Dict[str, np.ndarray]
    models: Dict[str, OLSResult]

    def summary_table(self, *, round_to: int | None = None) -> pd.DataFrame:
        rows = []
        for target, model in self.models.items():
            formula = f"{target} ~ " + " + ".join(model.x_names)

            X_full = model.X
            y = model.fitted + model.resid  # original dependent variable

            for i, regressor in enumerate(model.x_names, start=1):
                X_reduced = np.delete(X_full, i, axis=1)

                remaining_x_names = [x for k, x in enumerate(model.x_names) if k != (i - 1)]

                if len(remaining_x_names) == 0:
                    r2_reduced = 0.0
                else:
                    reduced_model = ols(
                        y=y,
                        X=X_reduced[:, 1:] if X_reduced.shape[1] > 1 else np.empty((len(y), 0)),
                        intercept=True,
                        y_name=target,
                        x_names=remaining_x_names,
                    )
                    r2_reduced = reduced_model.r2

                marginal_r2 = model.r2 - r2_reduced

                rows.append(
                    {
                        "target": target,
                        "model": formula,
                        "regressor": regressor,
                        "coef": model.beta[i],
                        "std_error": model.se[i],
                        "t_stat": model.t_stat[i],
                        "p_value": model.p_value[i],
                        "r2": model.r2,
                        "marginal_r2": marginal_r2,
                    }
                )

        df = pd.DataFrame(rows)
        if round_to is not None:
            df = df.round(round_to)
        return df


@dataclass
class MeasurementRegressionBundle:
    raw: pd.DataFrame
    pivot_coef: pd.DataFrame
    pivot_standardized_coef: pd.DataFrame
    pivot_p_value: pd.DataFrame
    pivot_std_error: pd.DataFrame
    pivot_r2: pd.DataFrame


@dataclass
class LagBlockRegressionBundle:
    raw: pd.DataFrame
    coefficients: pd.DataFrame


@dataclass
class SinglePredictorRegressionBundle:
    raw: pd.DataFrame
    by_measurement: Dict[str, pd.DataFrame]


STATE_NAME_MAP_DEFAULT = {2: "r", 3: "x", 4: "Pi"}
MEASUREMENT_NAME_MAP_DEFAULT = {0: "OutGap", 1: "Infl", 2: "Rate"}
ORTHOGONALIZATION_MAP_DEFAULT = {"Pi": ["r", "x"], "x": ["r", "Pi"], "r": ["x", "Pi"]}


def nw_style_lag_order(sample_size: int) -> int:
    if sample_size <= 1:
        return 0
    lag_order = int(np.floor(4.0 * (sample_size / 100.0) ** (2.0 / 9.0)))
    return min(max(lag_order, 0), sample_size - 1)


def extract_state_dict(kf, state_name_map: Mapping[int, str] | None = None) -> Dict[str, np.ndarray]:
    state_name_map = state_name_map or STATE_NAME_MAP_DEFAULT
    return {name: np.asarray(kf.x_pred[:, idx], dtype=float).reshape(-1, 1) for idx, name in state_name_map.items()}



def extract_measurement_dict(kf, measurement_name_map: Mapping[int, str] | None = None) -> Dict[str, np.ndarray]:
    measurement_name_map = measurement_name_map or MEASUREMENT_NAME_MAP_DEFAULT
    return {name: np.asarray(kf.innov[:, idx], dtype=float).reshape(-1, 1) for idx, name in measurement_name_map.items()}


def _lagged_design(x: np.ndarray, *, lag_order: int, predictor_name: str) -> tuple[np.ndarray, list[str]]:
    x_vec = np.asarray(x, dtype=float).reshape(-1)
    if lag_order < 0:
        raise ValueError("lag_order must be non-negative")
    if lag_order >= len(x_vec):
        raise ValueError("lag_order must be smaller than the sample size")

    columns = [x_vec[lag_order - lag : len(x_vec) - lag] for lag in range(lag_order + 1)]
    names = [f"{predictor_name}_L{lag}" for lag in range(lag_order + 1)]
    return np.column_stack(columns), names


def _block_wald_test(model: OLSResult, coefficient_indices: Sequence[int]) -> dict[str, float]:
    idx = np.asarray(list(coefficient_indices), dtype=int)
    q = int(len(idx))
    if q == 0 or model.df_resid <= 0:
        return {
            "block_wald_stat": np.nan,
            "block_f_stat": np.nan,
            "p_value": np.nan,
            "df_num": q,
            "df_denom": model.df_resid,
        }

    beta_block = model.beta[idx]
    cov_block = model.cov_beta[np.ix_(idx, idx)]
    try:
        cov_inv = np.linalg.inv(cov_block)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_block)

    wald_stat = float(beta_block @ cov_inv @ beta_block)
    f_stat = float(wald_stat / q)
    p_value = float(f.sf(f_stat, q, model.df_resid))
    return {
        "block_wald_stat": wald_stat,
        "block_f_stat": f_stat,
        "p_value": p_value,
        "df_num": q,
        "df_denom": model.df_resid,
    }



def orthogonalize_predictors(
    data: Mapping[str, np.ndarray],
    orthogonalization_map: Mapping[str, Sequence[str]],
    *,
    intercept: bool = True,
) -> OrthogonalizationBundle:
    residuals: Dict[str, np.ndarray] = {}
    models: Dict[str, OLSResult] = {}

    for target, controls in orthogonalization_map.items():
        model = ols(
            y=np.asarray(data[target], dtype=float),
            X=np.column_stack([np.asarray(data[c], dtype=float).reshape(-1) for c in controls]),
            intercept=intercept,
            y_name=target,
            x_names=list(controls),
        )
        residuals[target] = model.resid.reshape(-1, 1)
        models[target] = model

    return OrthogonalizationBundle(residuals=residuals, models=models)


def run_measurement_lag_block_regressions(
    measurements: Mapping[str, np.ndarray],
    predictors: Mapping[str, np.ndarray],
    *,
    lag_order: int | None = None,
    intercept: bool = True,
) -> LagBlockRegressionBundle:
    block_rows = []
    coefficient_rows = []
    sample_size = len(np.asarray(next(iter(measurements.values())), dtype=float).reshape(-1))
    selected_lag_order = nw_style_lag_order(sample_size) if lag_order is None else int(lag_order)

    for measurement_name, y in measurements.items():
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        y_aligned = y_vec[selected_lag_order:]
        for predictor_name, x in predictors.items():
            X_lagged, x_names = _lagged_design(x, lag_order=selected_lag_order, predictor_name=predictor_name)
            model = ols(
                y=y_aligned,
                X=X_lagged,
                intercept=intercept,
                y_name=measurement_name,
                x_names=x_names,
            )
            first_slope_idx = 1 if intercept else 0
            block_indices = list(range(first_slope_idx, first_slope_idx + len(x_names)))
            block_test = _block_wald_test(model, block_indices)

            block_rows.append(
                {
                    "measurement": measurement_name,
                    "predictor": predictor_name,
                    "intercept": intercept,
                    "lag_order": selected_lag_order,
                    "sample_size": model.nobs,
                    "n_block_coefficients": len(x_names),
                    "coef": model.beta[first_slope_idx],
                    "standardized_coef": standardized_slope_from_stds(
                        model.beta[first_slope_idx],
                        X_lagged[:, 0],
                        y_aligned,
                    ),
                    "std_error": model.se[first_slope_idx],
                    "t_stat": model.t_stat[first_slope_idx],
                    "coef_p_value": model.p_value[first_slope_idx],
                    "r2": model.r2,
                    **block_test,
                }
            )

            for lag, term_name in enumerate(x_names):
                slope_idx = first_slope_idx + lag
                coefficient_rows.append(
                    {
                        "measurement": measurement_name,
                        "predictor": predictor_name,
                        "lag": lag,
                        "term": term_name,
                        "intercept": intercept,
                        "lag_order": selected_lag_order,
                        "sample_size": model.nobs,
                        "coef": model.beta[slope_idx],
                        "standardized_coef": standardized_slope_from_stds(
                            model.beta[slope_idx],
                            X_lagged[:, lag],
                            y_aligned,
                        ),
                        "std_error": model.se[slope_idx],
                        "t_stat": model.t_stat[slope_idx],
                        "p_value": model.p_value[slope_idx],
                        "r2": model.r2,
                        "block_wald_stat": block_test["block_wald_stat"],
                        "block_f_stat": block_test["block_f_stat"],
                        "block_p_value": block_test["p_value"],
                        "df_num": block_test["df_num"],
                        "df_denom": block_test["df_denom"],
                    }
                )

    return LagBlockRegressionBundle(
        raw=pd.DataFrame(block_rows),
        coefficients=pd.DataFrame(coefficient_rows),
    )



def run_measurement_regressions(
    measurements: Mapping[str, np.ndarray],
    predictors: Mapping[str, np.ndarray],
    *,
    intercept: bool = True,
    z_score_standardization: bool = False,
) -> MeasurementRegressionBundle:
    rows = []
    predictor_names = list(predictors.keys())
    measurement_names = list(measurements.keys())

    for measurement_name, y in measurements.items():
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        for predictor_name, x in predictors.items():
            x_arr = _as_2d(np.asarray(x, dtype=float))
            model = ols(
                y=y_vec,
                X=x_arr,
                intercept=intercept,
                y_name=measurement_name,
                x_names=[predictor_name],
            )
            slope_idx = 1 if intercept else 0
            rows.append(
                {
                    "measurement": measurement_name,
                    "predictor": predictor_name,
                    "intercept": intercept,
                    "coef": model.beta[slope_idx],
                    "standardized_coef": standardized_slope_from_model(
                        raw_model=model,
                        x=x_arr,
                        y=y_vec,
                        intercept=intercept,
                        z_score_standardization=z_score_standardization,
                    ),
                    "std_error": model.se[slope_idx],
                    "t_stat": model.t_stat[slope_idx],
                    "p_value": model.p_value[slope_idx],
                    "r2": model.r2,
                }
            )

    raw = pd.DataFrame(rows)
    pivot_kwargs = dict(index="predictor", columns="measurement")
    return MeasurementRegressionBundle(
        raw=raw,
        pivot_coef=raw.pivot(**pivot_kwargs, values="coef").reindex(index=predictor_names, columns=measurement_names),
        pivot_standardized_coef=raw.pivot(**pivot_kwargs, values="standardized_coef").reindex(index=predictor_names, columns=measurement_names),
        pivot_p_value=raw.pivot(**pivot_kwargs, values="p_value").reindex(index=predictor_names, columns=measurement_names),
        pivot_std_error=raw.pivot(**pivot_kwargs, values="std_error").reindex(index=predictor_names, columns=measurement_names),
        pivot_r2=raw.pivot(**pivot_kwargs, values="r2").reindex(index=predictor_names, columns=measurement_names),
    )



def run_single_predictor_reports(
    measurements: Mapping[str, np.ndarray],
    predictors: Mapping[str, np.ndarray],
    *,
    intercept: bool = True,
    z_score_standardization: bool = False,
) -> SinglePredictorRegressionBundle:
    rows = []
    by_measurement: Dict[str, pd.DataFrame] = {}

    for measurement_name, y in measurements.items():
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        measurement_rows = []
        for predictor_name, x in predictors.items():
            x_arr = _as_2d(np.asarray(x, dtype=float))
            model = ols(
                y=y_vec,
                X=x_arr,
                intercept=intercept,
                y_name=measurement_name,
                x_names=[predictor_name],
            )
            slope_idx = 1 if intercept else 0
            row = {
                "measurement": measurement_name,
                "predictor": predictor_name,
                "intercept": intercept,
                "coef": model.beta[slope_idx],
                "standardized_coef": standardized_slope_from_model(
                    raw_model=model,
                    x=x_arr,
                    y=y_vec,
                    intercept=intercept,
                    z_score_standardization=z_score_standardization,
                ),
                "r2": model.r2,
                "std_error": model.se[slope_idx],
                "t_stat": model.t_stat[slope_idx],
                "p_value": model.p_value[slope_idx],
            }
            rows.append(row)
            measurement_rows.append(row)
        by_measurement[measurement_name] = pd.DataFrame(measurement_rows).set_index("predictor")

    return SinglePredictorRegressionBundle(raw=pd.DataFrame(rows), by_measurement=by_measurement)



def run_full_regression_diagnostics(
    kf,
    *,
    state_name_map: Mapping[int, str] | None = None,
    measurement_name_map: Mapping[int, str] | None = None,
    orthogonalization_map: Mapping[str, Sequence[str]] | None = None,
    intercept: bool = True,
    z_score_standardization: bool = False,
) -> dict[str, object]:
    state_name_map = state_name_map or STATE_NAME_MAP_DEFAULT
    measurement_name_map = measurement_name_map or MEASUREMENT_NAME_MAP_DEFAULT
    orthogonalization_map = orthogonalization_map or ORTHOGONALIZATION_MAP_DEFAULT

    raw_states = extract_state_dict(kf, state_name_map)
    measurements = extract_measurement_dict(kf, measurement_name_map)
    orth_bundle = orthogonalize_predictors(raw_states, orthogonalization_map, intercept=intercept)

    raw_measurement_regs = run_measurement_regressions(
        measurements,
        raw_states,
        intercept=intercept,
        z_score_standardization=z_score_standardization,
    )
    orth_measurement_regs = run_measurement_regressions(
        measurements,
        orth_bundle.residuals,
        intercept=intercept,
        z_score_standardization=z_score_standardization,
    )
    raw_measurement_regs_no_intercept = run_measurement_regressions(
        measurements,
        raw_states,
        intercept=False,
        z_score_standardization=z_score_standardization,
    )
    orth_measurement_regs_no_intercept = run_measurement_regressions(
        measurements,
        orth_bundle.residuals,
        intercept=False,
        z_score_standardization=z_score_standardization,
    )
    raw_lag_block_regs = run_measurement_lag_block_regressions(
        measurements,
        raw_states,
        intercept=intercept,
    )
    orth_lag_block_regs = run_measurement_lag_block_regressions(
        measurements,
        orth_bundle.residuals,
        intercept=intercept,
    )
    raw_lag_block_regs_no_intercept = run_measurement_lag_block_regressions(
        measurements,
        raw_states,
        intercept=False,
    )
    orth_lag_block_regs_no_intercept = run_measurement_lag_block_regressions(
        measurements,
        orth_bundle.residuals,
        intercept=False,
    )

    raw_single_predictor = run_single_predictor_reports(
        measurements,
        raw_states,
        intercept=intercept,
        z_score_standardization=z_score_standardization,
    )
    orth_single_predictor = run_single_predictor_reports(
        measurements,
        orth_bundle.residuals,
        intercept=intercept,
        z_score_standardization=z_score_standardization,
    )
    raw_single_predictor_no_intercept = run_single_predictor_reports(
        measurements,
        raw_states,
        intercept=False,
        z_score_standardization=z_score_standardization,
    )
    orth_single_predictor_no_intercept = run_single_predictor_reports(
        measurements,
        orth_bundle.residuals,
        intercept=False,
        z_score_standardization=z_score_standardization,
    )

    return {
        "states": raw_states,
        "measurements": measurements,
        "orthogonalization": orth_bundle,
        "measurement_regressions_raw": raw_measurement_regs,
        "measurement_regressions_orthogonalized": orth_measurement_regs,
        "measurement_regressions_raw_no_intercept": raw_measurement_regs_no_intercept,
        "measurement_regressions_orthogonalized_no_intercept": orth_measurement_regs_no_intercept,
        "measurement_lag_block_regressions_raw": raw_lag_block_regs,
        "measurement_lag_block_regressions_orthogonalized": orth_lag_block_regs,
        "measurement_lag_block_regressions_raw_no_intercept": raw_lag_block_regs_no_intercept,
        "measurement_lag_block_regressions_orthogonalized_no_intercept": orth_lag_block_regs_no_intercept,
        "single_predictor_reports_raw": raw_single_predictor,
        "single_predictor_reports_orthogonalized": orth_single_predictor,
        "single_predictor_reports_raw_no_intercept": raw_single_predictor_no_intercept,
        "single_predictor_reports_orthogonalized_no_intercept": orth_single_predictor_no_intercept,
        "z_score_standardization": z_score_standardization,
    }



def format_regression_outputs(diagnostics: Mapping[str, object], *, round_to: int = 3) -> dict[str, object]:
    orth_bundle: OrthogonalizationBundle = diagnostics["orthogonalization"]
    raw_regs: MeasurementRegressionBundle = diagnostics["measurement_regressions_raw"]
    orth_regs: MeasurementRegressionBundle = diagnostics["measurement_regressions_orthogonalized"]
    raw_regs_no_intercept: MeasurementRegressionBundle = diagnostics["measurement_regressions_raw_no_intercept"]
    orth_regs_no_intercept: MeasurementRegressionBundle = diagnostics["measurement_regressions_orthogonalized_no_intercept"]
    raw_lag_regs: LagBlockRegressionBundle = diagnostics["measurement_lag_block_regressions_raw"]
    orth_lag_regs: LagBlockRegressionBundle = diagnostics["measurement_lag_block_regressions_orthogonalized"]
    raw_lag_regs_no_intercept: LagBlockRegressionBundle = diagnostics["measurement_lag_block_regressions_raw_no_intercept"]
    orth_lag_regs_no_intercept: LagBlockRegressionBundle = diagnostics["measurement_lag_block_regressions_orthogonalized_no_intercept"]
    raw_single: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_raw"]
    orth_single: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_orthogonalized"]
    raw_single_no_intercept: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_raw_no_intercept"]
    orth_single_no_intercept: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_orthogonalized_no_intercept"]

    return {
        "z_score_standardization": diagnostics.get("z_score_standardization", False),
        "orthogonalization_summary": orth_bundle.summary_table(round_to=round_to),
        "measurement_coef_raw": raw_regs.pivot_coef.round(round_to),
        "measurement_standardized_coef_raw": raw_regs.pivot_standardized_coef.round(round_to),
        "measurement_p_value_raw": raw_regs.pivot_p_value.round(round_to),
        "measurement_std_error_raw": raw_regs.pivot_std_error.round(round_to),
        "measurement_coef_orthogonalized": orth_regs.pivot_coef.round(round_to),
        "measurement_standardized_coef_orthogonalized": orth_regs.pivot_standardized_coef.round(round_to),
        "measurement_p_value_orthogonalized": orth_regs.pivot_p_value.round(round_to),
        "measurement_std_error_orthogonalized": orth_regs.pivot_std_error.round(round_to),
        "measurement_coef_raw_no_intercept": raw_regs_no_intercept.pivot_coef.round(round_to),
        "measurement_standardized_coef_raw_no_intercept": raw_regs_no_intercept.pivot_standardized_coef.round(round_to),
        "measurement_p_value_raw_no_intercept": raw_regs_no_intercept.pivot_p_value.round(round_to),
        "measurement_std_error_raw_no_intercept": raw_regs_no_intercept.pivot_std_error.round(round_to),
        "measurement_coef_orthogonalized_no_intercept": orth_regs_no_intercept.pivot_coef.round(round_to),
        "measurement_standardized_coef_orthogonalized_no_intercept": orth_regs_no_intercept.pivot_standardized_coef.round(round_to),
        "measurement_p_value_orthogonalized_no_intercept": orth_regs_no_intercept.pivot_p_value.round(round_to),
        "measurement_std_error_orthogonalized_no_intercept": orth_regs_no_intercept.pivot_std_error.round(round_to),
        "single_predictor_raw": {k: v.round(round_to) for k, v in raw_single.by_measurement.items()},
        "single_predictor_orthogonalized": {k: v.round(round_to) for k, v in orth_single.by_measurement.items()},
        "single_predictor_raw_no_intercept": {k: v.round(round_to) for k, v in raw_single_no_intercept.by_measurement.items()},
        "single_predictor_orthogonalized_no_intercept": {k: v.round(round_to) for k, v in orth_single_no_intercept.by_measurement.items()},
        "measurement_regressions_raw_long": raw_regs.raw.round(round_to),
        "measurement_regressions_orthogonalized_long": orth_regs.raw.round(round_to),
        "measurement_regressions_raw_no_intercept_long": raw_regs_no_intercept.raw.round(round_to),
        "measurement_regressions_orthogonalized_no_intercept_long": orth_regs_no_intercept.raw.round(round_to),
        "measurement_lag_block_regressions_raw_long": raw_lag_regs.raw.round(round_to),
        "measurement_lag_block_regressions_orthogonalized_long": orth_lag_regs.raw.round(round_to),
        "measurement_lag_block_regressions_raw_no_intercept_long": raw_lag_regs_no_intercept.raw.round(round_to),
        "measurement_lag_block_regressions_orthogonalized_no_intercept_long": orth_lag_regs_no_intercept.raw.round(round_to),
        "measurement_lag_block_coefficients_raw_long": raw_lag_regs.coefficients.round(round_to),
        "measurement_lag_block_coefficients_orthogonalized_long": orth_lag_regs.coefficients.round(round_to),
        "measurement_lag_block_coefficients_raw_no_intercept_long": raw_lag_regs_no_intercept.coefficients.round(round_to),
        "measurement_lag_block_coefficients_orthogonalized_no_intercept_long": orth_lag_regs_no_intercept.coefficients.round(round_to),
    }
