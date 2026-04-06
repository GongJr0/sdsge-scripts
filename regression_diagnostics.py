from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import r2_score


@dataclass
class OLSResult:
    y_name: str
    x_names: list[str]
    beta: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    r2: float
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray

    def coefficient_table(self, *, drop_intercept: bool = True, round_to: int | None = None) -> pd.DataFrame:
        names = ["const", *self.x_names]
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
        if drop_intercept:
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
    sigma2 = np.sum(resid ** 2) / (n - k)
    var_beta = sigma2 * np.linalg.inv(X_reg.T @ X_reg)
    se = np.sqrt(np.diag(var_beta))
    t_stat = beta / se
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - k))

    return OLSResult(
        y_name=y_name,
        x_names=x_names,
        beta=beta,
        fitted=fitted,
        resid=resid,
        r2=r2,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
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
    if not z_score_standardization:
        return standardized_slope_from_stds(raw_model.beta[1], x, y)

    z_model = ols(
        y=_zscore(np.asarray(y, dtype=float).reshape(-1)),
        X=_zscore(_as_2d(x)),
        intercept=intercept,
        y_name=raw_model.y_name,
        x_names=raw_model.x_names,
    )
    return float(z_model.beta[1])


@dataclass
class OrthogonalizationBundle:
    residuals: Dict[str, np.ndarray]
    models: Dict[str, OLSResult]

    def summary_table(self, *, round_to: int | None = None) -> pd.DataFrame:
        rows = []
        for target, model in self.models.items():
            formula = f"{target} ~ " + " + ".join(model.x_names)
            for i, regressor in enumerate(model.x_names, start=1):
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
class SinglePredictorRegressionBundle:
    raw: pd.DataFrame
    by_measurement: Dict[str, pd.DataFrame]


STATE_NAME_MAP_DEFAULT = {2: "r", 3: "x", 4: "Pi"}
MEASUREMENT_NAME_MAP_DEFAULT = {0: "OutGap", 1: "Infl", 2: "Rate"}
ORTHOGONALIZATION_MAP_DEFAULT = {"Pi": ["r", "x"], "x": ["r", "Pi"], "r": ["x", "Pi"]}


def extract_state_dict(kf, state_name_map: Mapping[int, str] | None = None) -> Dict[str, np.ndarray]:
    state_name_map = state_name_map or STATE_NAME_MAP_DEFAULT
    return {name: np.asarray(kf.x_pred[:, idx], dtype=float).reshape(-1, 1) for idx, name in state_name_map.items()}



def extract_measurement_dict(kf, measurement_name_map: Mapping[int, str] | None = None) -> Dict[str, np.ndarray]:
    measurement_name_map = measurement_name_map or MEASUREMENT_NAME_MAP_DEFAULT
    return {name: np.asarray(kf.innov[:, idx], dtype=float).reshape(-1, 1) for idx, name in measurement_name_map.items()}



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
            rows.append(
                {
                    "measurement": measurement_name,
                    "predictor": predictor_name,
                    "coef": model.beta[1],
                    "standardized_coef": standardized_slope_from_model(
                        raw_model=model,
                        x=x_arr,
                        y=y_vec,
                        intercept=intercept,
                        z_score_standardization=z_score_standardization,
                    ),
                    "std_error": model.se[1],
                    "t_stat": model.t_stat[1],
                    "p_value": model.p_value[1],
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
            row = {
                "measurement": measurement_name,
                "predictor": predictor_name,
                "coef": model.beta[1],
                "standardized_coef": standardized_slope_from_model(
                    raw_model=model,
                    x=x_arr,
                    y=y_vec,
                    intercept=intercept,
                    z_score_standardization=z_score_standardization,
                ),
                "r2": model.r2,
                "std_error": model.se[1],
                "t_stat": model.t_stat[1],
                "p_value": model.p_value[1],
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

    return {
        "states": raw_states,
        "measurements": measurements,
        "orthogonalization": orth_bundle,
        "measurement_regressions_raw": raw_measurement_regs,
        "measurement_regressions_orthogonalized": orth_measurement_regs,
        "single_predictor_reports_raw": raw_single_predictor,
        "single_predictor_reports_orthogonalized": orth_single_predictor,
        "z_score_standardization": z_score_standardization,
    }



def format_regression_outputs(diagnostics: Mapping[str, object], *, round_to: int = 3) -> dict[str, object]:
    orth_bundle: OrthogonalizationBundle = diagnostics["orthogonalization"]
    raw_regs: MeasurementRegressionBundle = diagnostics["measurement_regressions_raw"]
    orth_regs: MeasurementRegressionBundle = diagnostics["measurement_regressions_orthogonalized"]
    raw_single: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_raw"]
    orth_single: SinglePredictorRegressionBundle = diagnostics["single_predictor_reports_orthogonalized"]

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
        "single_predictor_raw": {k: v.round(round_to) for k, v in raw_single.by_measurement.items()},
        "single_predictor_orthogonalized": {k: v.round(round_to) for k, v in orth_single.by_measurement.items()},
        "measurement_regressions_raw_long": raw_regs.raw.round(round_to),
        "measurement_regressions_orthogonalized_long": orth_regs.raw.round(round_to),
    }
