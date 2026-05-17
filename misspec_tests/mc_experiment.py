from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import chi2, norm
from statsmodels.stats.diagnostic import acorr_ljungbox

from SymbolicDSGE import Shock

from mc_storage import AugmentationTable, MCRecordBatch, MCTableStore, ReferenceTable
from regression_diagnostics import run_full_regression_diagnostics


MEASUREMENT_NAMES = ("OutGap", "Infl", "Rate")
STATE_NAMES = ("Pi", "x", "r")
ALPHA_DEFAULT = 0.05
LJUNG_BOX_LAGS = 20
STEADY_STATE_DEFAULT = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "MODELS" / "misspec_test"
AUGMENTED_CONFIG_BY_EQUATION = {
    "OutGap": MODEL_CONFIG_DIR / "augmented_reference_outgap.yaml",
    "Infl": MODEL_CONFIG_DIR / "augmented_reference_infl.yaml",
    "Rate": MODEL_CONFIG_DIR / "augmented_reference_rate.yaml",
}


def augmented_config_path(equation_name: str) -> str:
    try:
        return str(AUGMENTED_CONFIG_BY_EQUATION[equation_name])
    except KeyError as exc:
        valid = ", ".join(AUGMENTED_CONFIG_BY_EQUATION)
        raise ValueError(f"Unknown augmented equation {equation_name!r}. Valid choices: {valid}.") from exc


@dataclass
class ReferenceRepresentative:
    sim_dgp: dict[str, np.ndarray]
    obs: np.ndarray
    kf: Any
    std_innov: np.ndarray
    err_scale: np.ndarray


@dataclass
class AugmentationRepresentative:
    res_mle: Any
    sol_mle: Any
    kf_aug: Any
    std_innov_aug: np.ndarray
    sim_aug: dict[str, np.ndarray]


def make_seed_counter(*, start: int = 0) -> Iterator[int]:
    return count(start)


def next_seed(seed_counter: Iterator[int]) -> int:
    return int(next(seed_counter))


def make_filter_kwargs(err_scale: np.ndarray, *, known_r: bool) -> dict[str, np.ndarray]:
    n_obs = int(len(err_scale))
    if known_r:
        return {"R": np.diag(err_scale)}
    return {"R": np.zeros((n_obs, n_obs))}


def standardize_innovations(kf: Any) -> np.ndarray:
    innov = np.asarray(kf.innov, dtype=float)
    std_innov = np.empty_like(innov)
    for t in range(innov.shape[0]):
        s_t = 0.5 * (kf.S[t] + kf.S[t].T)
        chol = cholesky(s_t, lower=True)
        std_innov[t] = solve_triangular(chol, innov[t], lower=True)
    return std_innov


def wilson_interval(successes: int, trials: int, *, alpha: float = ALPHA_DEFAULT) -> tuple[float, float]:
    if trials <= 0:
        return (np.nan, np.nan)

    z = norm.ppf(1.0 - alpha / 2.0)
    phat = successes / trials
    denom = 1.0 + (z**2 / trials)
    center = (phat + z**2 / (2.0 * trials)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) / trials) + (z**2 / (4.0 * trials**2))) / denom
    return (center - radius, center + radius)


def monte_carlo_standard_error(values: pd.Series) -> float:
    values = values.dropna()
    n = len(values)
    if n <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(n))


def add_mc_se_columns(
    summary: pd.DataFrame,
    records: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    if not value_cols:
        return summary

    if group_cols:
        se_part = (
            records.groupby(group_cols, sort=False, dropna=False)[value_cols]
            .agg(monte_carlo_standard_error)
            .reset_index()
            .rename(columns={col: f"mc_se_{col}" for col in value_cols})
        )
        return summary.merge(se_part, on=group_cols, how="left")

    se_values = {
        f"mc_se_{col}": monte_carlo_standard_error(records[col])
        for col in value_cols
    }
    return summary.assign(**se_values)


def rejection_rate_standard_error(successes: int, trials: int) -> float:
    if trials <= 0:
        return np.nan
    phat = successes / trials
    return float(np.sqrt(phat * (1.0 - phat) / trials))


def add_rejection_summary(df: pd.DataFrame, group_cols: list[str], *, alpha: float = ALPHA_DEFAULT) -> pd.DataFrame:
    out = df.copy()
    out["reject"] = out["p_value"] < alpha

    numeric_cols = [
        col
        for col in out.select_dtypes(include=[np.number]).columns
        if col not in {*group_cols, "reject", "replication", "success"}
    ]

    grouped = out.groupby(group_cols, sort=False, dropna=False)
    mean_part = grouped[numeric_cols].mean().reset_index()
    mean_part = add_mc_se_columns(mean_part, out, group_cols, numeric_cols)
    reject_part = grouped["reject"].agg(["sum", "count"]).reset_index()
    reject_part["reject_rate"] = reject_part["sum"] / reject_part["count"]
    reject_part["reject_rate_mc_se"] = reject_part.apply(
        lambda row: rejection_rate_standard_error(int(row["sum"]), int(row["count"])),
        axis=1,
    )

    ci_bounds = reject_part.apply(
        lambda row: wilson_interval(int(row["sum"]), int(row["count"]), alpha=alpha),
        axis=1,
        result_type="expand",
    )
    ci_bounds.columns = ["reject_ci_low", "reject_ci_high"]
    reject_part = pd.concat([reject_part, ci_bounds], axis=1)
    reject_part = reject_part.rename(columns={"count": "n_replications", "sum": "n_rejections"})

    cols = group_cols + [
        "n_replications",
        "n_rejections",
        "reject_rate",
        "reject_rate_mc_se",
        "reject_ci_low",
        "reject_ci_high",
    ]
    return mean_part.merge(reject_part[cols], on=group_cols, how="left")


def aggregate_scalar_frame(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    value_cols: list[str] | None = None,
    alpha: float = ALPHA_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()
    if value_cols is None:
        value_cols = [
            col
            for col in out.select_dtypes(include=[np.number]).columns
            if col not in set(group_cols) and col not in {"replication", "success"}
        ]

    if group_cols:
        grouped = out.groupby(group_cols, sort=False, dropna=False)
        agg = grouped[value_cols].mean().reset_index()
    else:
        agg = pd.DataFrame([{col: float(out[col].mean()) for col in value_cols}])
    agg = add_mc_se_columns(agg, out, group_cols, value_cols)

    if "p_value" in out.columns:
        if group_cols:
            reject_part = grouped["p_value"].apply(lambda s: int(np.sum(s < alpha))).reset_index(name="n_rejections")
            reject_part["n_replications"] = grouped.size().to_numpy()
            reject_part["reject_rate"] = reject_part["n_rejections"] / reject_part["n_replications"]
            reject_part["reject_rate_mc_se"] = reject_part.apply(
                lambda row: rejection_rate_standard_error(int(row["n_rejections"]), int(row["n_replications"])),
                axis=1,
            )
            ci_bounds = reject_part.apply(
                lambda row: wilson_interval(int(row["n_rejections"]), int(row["n_replications"]), alpha=alpha),
                axis=1,
                result_type="expand",
            )
            ci_bounds.columns = ["reject_ci_low", "reject_ci_high"]
            reject_part = pd.concat([reject_part, ci_bounds], axis=1)
            agg = agg.merge(
                reject_part[
                    group_cols
                    + [
                        "n_replications",
                        "n_rejections",
                        "reject_rate",
                        "reject_rate_mc_se",
                        "reject_ci_low",
                        "reject_ci_high",
                    ]
                ],
                on=group_cols,
                how="left",
            )
        else:
            n_rejections = int(np.sum(out["p_value"] < alpha))
            n_replications = int(len(out))
            ci_low, ci_high = wilson_interval(n_rejections, n_replications, alpha=alpha)
            agg["n_replications"] = n_replications
            agg["n_rejections"] = n_rejections
            agg["reject_rate"] = n_rejections / n_replications if n_replications else np.nan
            agg["reject_rate_mc_se"] = rejection_rate_standard_error(n_rejections, n_replications)
            agg["reject_ci_low"] = ci_low
            agg["reject_ci_high"] = ci_high

    return agg


def records_by_predictor(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if records.empty or "predictor" not in records.columns:
        return {}
    return {
        str(predictor): group.reset_index(drop=True)
        for predictor, group in records.groupby("predictor", sort=False, dropna=False)
    }


INNOVATION_DECOMPOSITION_VALUE_COLUMNS = [
    "beta_measurement_error",
    "beta_state_prediction_error",
    "beta_total_innovation",
    "beta_component_sum",
    "beta_component_gap",
    "abs_beta_component_gap",
    "reconstruction_max_abs_error",
]


def _aligned_true_states(
    sol_dgp: Any,
    sol_kf: Any,
    sim_dgp: Mapping[str, np.ndarray],
    sample_size: int,
) -> np.ndarray:
    true_state_path = np.asarray(sim_dgp["_X"], dtype=float)
    if true_state_path.shape[0] < sample_size + 1:
        raise ValueError(
            f"DGP state path has {true_state_path.shape[0]} rows, but {sample_size + 1} are required."
        )

    try:
        dgp_indices = [sol_dgp.compiled.idx[name] for name in sol_kf.compiled.var_names]
    except KeyError as exc:
        raise KeyError(
            f"Cannot align DGP states to filter state {exc.args[0]!r}; the state is absent from the DGP."
        ) from exc

    return true_state_path[1 : sample_size + 1, :][:, dgp_indices]


def _centered_slope(y: np.ndarray, x: np.ndarray) -> float:
    y_vec = np.asarray(y, dtype=float).reshape(-1)
    x_vec = np.asarray(x, dtype=float).reshape(-1)
    if y_vec.shape[0] != x_vec.shape[0]:
        raise ValueError(
            f"Slope inputs must have equal length, got {y_vec.shape[0]} and {x_vec.shape[0]}."
        )

    y_centered = y_vec - np.mean(y_vec)
    x_centered = x_vec - np.mean(x_vec)
    denom = float(x_centered @ x_centered)
    if denom == 0.0:
        return np.nan
    return float((x_centered @ y_centered) / denom)


def innovation_decomposition_record_batch(
    sol_kf: Any,
    sol_dgp: Any,
    sim_dgp: Mapping[str, np.ndarray],
    obs: np.ndarray,
    kf: Any,
    predictors: Mapping[str, np.ndarray],
    *,
    replication: int | None = None,
    structure: str,
) -> MCRecordBatch:
    obs_arr = np.asarray(obs, dtype=float)
    innov = np.asarray(kf.innov, dtype=float)
    x_pred = np.asarray(kf.x_pred, dtype=float)
    sample_size = int(innov.shape[0])

    if obs_arr.shape != innov.shape:
        raise ValueError(f"Observed data shape {obs_arr.shape} does not match innovation shape {innov.shape}.")
    if x_pred.shape[0] != sample_size:
        raise ValueError(
            f"Predicted state length {x_pred.shape[0]} does not match innovation length {sample_size}."
        )

    true_states = _aligned_true_states(sol_dgp, sol_kf, sim_dgp, sample_size)
    H, delta = sol_kf._build_C_d_from_obs(list(MEASUREMENT_NAMES))
    measurement_error_component = obs_arr - (true_states @ H.T + delta)
    state_prediction_error_component = (true_states - x_pred) @ H.T
    reconstructed_innovation = measurement_error_component + state_prediction_error_component
    reconstruction_max_abs_error = float(np.max(np.abs(innov - reconstructed_innovation)))

    rows = []
    for measurement_idx, measurement_name in enumerate(MEASUREMENT_NAMES):
        for predictor_name in STATE_NAMES:
            if predictor_name not in predictors:
                continue

            predictor = predictors[predictor_name]
            beta_measurement = _centered_slope(measurement_error_component[:, measurement_idx], predictor)
            beta_state_prediction = _centered_slope(state_prediction_error_component[:, measurement_idx], predictor)
            beta_total = _centered_slope(innov[:, measurement_idx], predictor)
            beta_component_sum = beta_measurement + beta_state_prediction
            beta_component_gap = beta_total - beta_component_sum
            rows.append(
                {
                    "structure": structure,
                    "measurement": measurement_name,
                    "predictor": predictor_name,
                    "beta_measurement_error": beta_measurement,
                    "beta_state_prediction_error": beta_state_prediction,
                    "beta_total_innovation": beta_total,
                    "beta_component_sum": beta_component_sum,
                    "beta_component_gap": beta_component_gap,
                    "abs_beta_component_gap": abs(beta_component_gap),
                    "reconstruction_max_abs_error": reconstruction_max_abs_error,
                }
            )

    columns = [
        "structure",
        "measurement",
        "predictor",
        "beta_measurement_error",
        "beta_state_prediction_error",
        "beta_total_innovation",
        "beta_component_sum",
        "beta_component_gap",
        "abs_beta_component_gap",
        "reconstruction_max_abs_error",
    ]
    if replication is not None:
        rows = [{"replication": replication, **row} for row in rows]
        columns = ["replication", *columns]

    return MCRecordBatch.from_records(rows, columns=columns)


def innovation_decomposition_records(
    sol_kf: Any,
    sol_dgp: Any,
    sim_dgp: Mapping[str, np.ndarray],
    obs: np.ndarray,
    kf: Any,
    predictors: Mapping[str, np.ndarray],
    *,
    replication: int,
    structure: str,
) -> pd.DataFrame:
    out = innovation_decomposition_record_batch(
        sol_kf,
        sol_dgp,
        sim_dgp,
        obs,
        kf,
        predictors,
        replication=replication,
        structure=structure,
    ).to_frame()
    return out


def compare_regression_setups(
    baseline_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    *,
    baseline_name: str,
    candidate_name: str,
    group_cols: list[str],
    alpha: float = ALPHA_DEFAULT,
) -> pd.DataFrame:
    if baseline_records.empty or candidate_records.empty:
        return pd.DataFrame()

    baseline = add_rejection_summary(baseline_records, group_cols, alpha=alpha)
    candidate = add_rejection_summary(candidate_records, group_cols, alpha=alpha)
    keep_cols = group_cols + [
        "n_replications",
        "n_rejections",
        "reject_rate",
        "reject_rate_mc_se",
        "reject_ci_low",
        "reject_ci_high",
    ]
    baseline = baseline[keep_cols].rename(
        columns={col: f"{baseline_name}_{col}" for col in keep_cols if col not in group_cols}
    )
    candidate = candidate[keep_cols].rename(
        columns={col: f"{candidate_name}_{col}" for col in keep_cols if col not in group_cols}
    )
    comparison = baseline.merge(candidate, on=group_cols, how="outer")
    comparison["reject_rate_difference"] = (
        comparison[f"{candidate_name}_reject_rate"] - comparison[f"{baseline_name}_reject_rate"]
    )

    paired = baseline_records[group_cols + ["replication", "p_value"]].merge(
        candidate_records[group_cols + ["replication", "p_value"]],
        on=group_cols + ["replication"],
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    if paired.empty:
        return comparison

    paired["baseline_reject"] = paired["p_value_baseline"] < alpha
    paired["candidate_reject"] = paired["p_value_candidate"] < alpha
    paired["candidate_only_reject"] = paired["candidate_reject"] & ~paired["baseline_reject"]
    paired["baseline_only_reject"] = paired["baseline_reject"] & ~paired["candidate_reject"]

    paired_summary = (
        paired.groupby(group_cols, sort=False, dropna=False)[
            ["candidate_only_reject", "baseline_only_reject"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "candidate_only_reject": f"{candidate_name}_only_reject_rate",
                "baseline_only_reject": f"{baseline_name}_only_reject_rate",
            }
        )
    )
    return comparison.merge(paired_summary, on=group_cols, how="left")


def summarize_moment_record(std_innov: np.ndarray) -> dict[str, float]:
    std_innov = np.asarray(std_innov, dtype=float)
    mu = std_innov.mean(axis=0)
    cov = np.cov(std_innov, rowvar=False)
    corr = np.corrcoef(std_innov, rowvar=False)

    out: dict[str, float] = {"sample_size": float(std_innov.shape[0])}
    for idx, measurement in enumerate(MEASUREMENT_NAMES):
        out[f"mean_{measurement}"] = float(mu[idx])

    for row_name_idx, row_name in enumerate(MEASUREMENT_NAMES):
        for col_name_idx, col_name in enumerate(MEASUREMENT_NAMES):
            out[f"cov_{row_name}_{col_name}"] = float(cov[row_name_idx, col_name_idx])
            out[f"corr_{row_name}_{col_name}"] = float(corr[row_name_idx, col_name_idx])

    off_diag = cov - np.diag(np.diag(cov))
    out["max_abs_mean"] = float(np.max(np.abs(mu)))
    out["max_abs_diag_cov_minus_one"] = float(np.max(np.abs(np.diag(cov) - 1.0)))
    out["max_abs_offdiag_cov"] = float(np.max(np.abs(off_diag)))
    return out


def average_vector(moment_records: pd.DataFrame, prefix: str) -> pd.Series:
    return pd.Series(
        {
            measurement: float(moment_records[f"{prefix}_{measurement}"].mean())
            for measurement in MEASUREMENT_NAMES
        }
    )


def average_vector_mc_se(moment_records: pd.DataFrame, prefix: str) -> pd.Series:
    return pd.Series(
        {
            measurement: monte_carlo_standard_error(moment_records[f"{prefix}_{measurement}"])
            for measurement in MEASUREMENT_NAMES
        }
    )


def average_matrix(moment_records: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = {
        row_name: [
            float(moment_records[f"{prefix}_{row_name}_{col_name}"].mean())
            for col_name in MEASUREMENT_NAMES
        ]
        for row_name in MEASUREMENT_NAMES
    }
    return pd.DataFrame(data, index=MEASUREMENT_NAMES).T


def average_matrix_mc_se(moment_records: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = {
        row_name: [
            monte_carlo_standard_error(moment_records[f"{prefix}_{row_name}_{col_name}"])
            for col_name in MEASUREMENT_NAMES
        ]
        for row_name in MEASUREMENT_NAMES
    }
    return pd.DataFrame(data, index=MEASUREMENT_NAMES).T


def summarize_average_moments(moment_records: pd.DataFrame) -> pd.Series:
    mean_vector = average_vector(moment_records, "mean").to_numpy(dtype=float)
    covariance = average_matrix(moment_records, "cov").to_numpy(dtype=float)
    off_diag_covariance = covariance - np.diag(np.diag(covariance))

    return pd.Series(
        {
            "max_abs_mean": float(np.max(np.abs(mean_vector))),
            "max_abs_diag_cov_minus_one": float(np.max(np.abs(np.diag(covariance) - 1.0))),
            "max_abs_offdiag_cov": float(np.max(np.abs(off_diag_covariance))),
        }
    )


def summarize_average_moment_mc_se(moment_records: pd.DataFrame) -> pd.Series:
    mean_vector = average_vector(moment_records, "mean")
    covariance = average_matrix(moment_records, "cov")
    off_diag_covariance_array = covariance.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(off_diag_covariance_array, 0.0)
    off_diag_covariance = pd.DataFrame(
        off_diag_covariance_array,
        index=covariance.index,
        columns=covariance.columns,
    )

    max_mean_measurement = mean_vector.abs().idxmax()
    diag_abs = (pd.Series(np.diag(covariance), index=MEASUREMENT_NAMES) - 1.0).abs()
    max_diag_measurement = diag_abs.idxmax()
    offdiag_abs_array = off_diag_covariance.abs().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(offdiag_abs_array, np.nan)
    offdiag_abs = pd.DataFrame(
        offdiag_abs_array,
        index=off_diag_covariance.index,
        columns=off_diag_covariance.columns,
    )
    max_offdiag_row, max_offdiag_col = offdiag_abs.stack().idxmax()

    return pd.Series(
        {
            "max_abs_mean": monte_carlo_standard_error(moment_records[f"mean_{max_mean_measurement}"]),
            "max_abs_diag_cov_minus_one": monte_carlo_standard_error(
                moment_records[f"cov_{max_diag_measurement}_{max_diag_measurement}"]
            ),
            "max_abs_offdiag_cov": monte_carlo_standard_error(
                moment_records[f"cov_{max_offdiag_row}_{max_offdiag_col}"]
            ),
        }
    )


def _vech(matrix: np.ndarray) -> np.ndarray:
    tril_idx = np.tril_indices_from(matrix)
    return np.asarray(matrix[tril_idx], dtype=float)


def bartlett_hac_bandwidth(sample_size: int) -> int:
    if sample_size <= 1:
        return 0
    bandwidth = int(np.floor(4.0 * (sample_size / 100.0) ** (2.0 / 9.0)))
    return min(max(bandwidth, 0), sample_size - 1)


def newey_west_long_run_covariance(moment_process: np.ndarray, *, bandwidth: int) -> np.ndarray:
    moment_process = np.asarray(moment_process, dtype=float)
    sample_size = moment_process.shape[0]
    if sample_size <= 0:
        raise ValueError("moment_process must contain at least one observation")

    moment_process = moment_process - moment_process.mean(axis=0, keepdims=True)
    omega_hat = (moment_process.T @ moment_process) / sample_size
    max_lag = min(int(bandwidth), sample_size - 1)
    for lag in range(1, max_lag + 1):
        gamma_l = (moment_process[lag:].T @ moment_process[:-lag]) / sample_size
        weight = 1.0 - (lag / (max_lag + 1.0))
        # Standard vector Newey-West HAC uses the symmetrized positive-lag autocovariances.
        omega_hat = omega_hat + weight * (gamma_l + gamma_l.T)

    return 0.5 * (omega_hat + omega_hat.T)


def moment_specification_test_records(
    std_innov: np.ndarray,
    *,
    replication: int | None = None,
) -> MCRecordBatch:
    std_innov = np.asarray(std_innov, dtype=float)
    sample_size, k = std_innov.shape
    cov_df = k * (k + 1) / 2

    mean_vector = std_innov.mean(axis=0)
    bandwidth = bartlett_hac_bandwidth(sample_size)

    mean_distance = float(np.linalg.norm(mean_vector))
    mean_omega_hat = newey_west_long_run_covariance(std_innov, bandwidth=bandwidth)
    try:
        mean_omega_inv = np.linalg.inv(mean_omega_hat)
    except np.linalg.LinAlgError:
        mean_omega_inv = np.linalg.pinv(mean_omega_hat)
    mean_stat = float(sample_size * mean_vector @ mean_omega_inv @ mean_vector)
    mean_p_value = float(chi2.sf(mean_stat, df=k))

    identity = np.eye(k)
    second_moment = (std_innov.T @ std_innov) / sample_size
    moment_process = np.array(
        [_vech(np.outer(u_t, u_t) - identity) for u_t in std_innov],
        dtype=float,
    )
    gbar = moment_process.mean(axis=0)
    omega_hat = newey_west_long_run_covariance(moment_process, bandwidth=bandwidth)
    try:
        omega_inv = np.linalg.inv(omega_hat)
    except np.linalg.LinAlgError:
        omega_inv = np.linalg.pinv(omega_hat)
    cov_stat = float(sample_size * gbar @ omega_inv @ gbar)
    cov_p_value = float(chi2.sf(cov_stat, df=cov_df))
    cov_distance = float(np.linalg.norm(second_moment - identity, ord="fro"))

    rows = [
        {
                "test": "mean_zero_hac",
                "df": k,
                "sample_size": sample_size,
                "bandwidth": bandwidth,
                "distance": mean_distance,
                "stat": mean_stat,
                "p_value": mean_p_value,
        },
        {
                "test": "cov_identity",
                "df": cov_df,
                "sample_size": sample_size,
                "bandwidth": bandwidth,
                "distance": cov_distance,
                "stat": cov_stat,
                "p_value": cov_p_value,
        },
    ]
    columns = [
            "test",
            "df",
            "sample_size",
            "bandwidth",
            "distance",
            "stat",
            "p_value",
    ]
    if replication is not None:
        rows = [{"replication": replication, **row} for row in rows]
        columns = ["replication", *columns]

    return MCRecordBatch.from_records(rows, columns=columns)


def moment_specification_test_frame(std_innov: np.ndarray, *, replication: int) -> pd.DataFrame:
    return moment_specification_test_records(std_innov, replication=replication).to_frame()


def summarize_moment_specification_tests(
    test_records: pd.DataFrame,
    *,
    alpha: float = ALPHA_DEFAULT,
) -> pd.DataFrame:
    summary = aggregate_scalar_frame(
        test_records.dropna(subset=["p_value"]),
        ["test"],
        value_cols=["distance", "stat", "p_value"],
        alpha=alpha,
    )
    dfs = (
        test_records.groupby("test", sort=False, dropna=False)[["df", "sample_size", "bandwidth"]]
        .first()
        .reset_index()
    )
    return summary.merge(dfs, on="test", how="left")


def summarize_moment_specification_comparison(
    reference_tests: pd.DataFrame,
    augmented_tests: pd.DataFrame,
    *,
    alpha: float = ALPHA_DEFAULT,
) -> pd.DataFrame:
    paired = reference_tests.merge(
        augmented_tests,
        on=["replication", "test"],
        how="inner",
        suffixes=("_ref", "_aug"),
    )
    if paired.empty:
        return pd.DataFrame()

    paired["distance_improvement"] = paired["distance_ref"] - paired["distance_aug"]
    paired["stat_improvement"] = paired["stat_ref"] - paired["stat_aug"]
    paired["aug_closer"] = paired["distance_improvement"] > 0.0

    rows = []
    for test, group in paired.groupby("test", sort=False, dropna=False):
        n = int(len(group))
        n_closer = int(group["aug_closer"].sum())
        ci_low, ci_high = wilson_interval(n_closer, n, alpha=alpha)
        rows.append(
            {
                "test": test,
                "n_replications": n,
                "distance_ref": float(group["distance_ref"].mean()),
                "mc_se_distance_ref": monte_carlo_standard_error(group["distance_ref"]),
                "distance_aug": float(group["distance_aug"].mean()),
                "mc_se_distance_aug": monte_carlo_standard_error(group["distance_aug"]),
                "distance_improvement": float(group["distance_improvement"].mean()),
                "mc_se_distance_improvement": monte_carlo_standard_error(group["distance_improvement"]),
                "stat_ref": float(group["stat_ref"].mean()),
                "mc_se_stat_ref": monte_carlo_standard_error(group["stat_ref"]),
                "stat_aug": float(group["stat_aug"].mean()),
                "mc_se_stat_aug": monte_carlo_standard_error(group["stat_aug"]),
                "stat_improvement": float(group["stat_improvement"].mean()),
                "mc_se_stat_improvement": monte_carlo_standard_error(group["stat_improvement"]),
                "aug_closer_rate": n_closer / n if n else np.nan,
                "aug_closer_rate_mc_se": rejection_rate_standard_error(n_closer, n),
                "aug_closer_ci_low": ci_low,
                "aug_closer_ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def lb_test_records(std_innov: np.ndarray, *, alpha: float = ALPHA_DEFAULT, lags: int = LJUNG_BOX_LAGS) -> MCRecordBatch:
    rows = []
    for idx, measurement in enumerate(MEASUREMENT_NAMES):
        test = acorr_ljungbox(std_innov[:, idx], lags=lags)
        rows.append(
            {
                "measurement": measurement,
                "lb_stat": float(test["lb_stat"].iloc[0]),
                "p_value": float(test["lb_pvalue"].iloc[0]),
            }
        )
    return MCRecordBatch.from_records(rows, columns=["measurement", "lb_stat", "p_value"])


def lb_test_frame(std_innov: np.ndarray, *, alpha: float = ALPHA_DEFAULT, lags: int = LJUNG_BOX_LAGS) -> pd.DataFrame:
    return lb_test_records(std_innov, alpha=alpha, lags=lags).to_frame()


def simulate_obs(
    sol_dgp: Any,
    *,
    T: int,
    err_scale: np.ndarray,
    seed_counter: Iterator[int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    gz_seed = next_seed(seed_counter)
    r_seed = next_seed(seed_counter)

    sim_dgp = sol_dgp.sim(
        T=T,
        shocks={
            "g,z": Shock(T, "norm", multivar=True, seed=gz_seed).shock_generator(),
            "r": Shock(T, "norm", multivar=False, seed=r_seed).shock_generator(),
        },
        observables=True,
    )

    obs = np.column_stack([sim_dgp["OutGap"], sim_dgp["Infl"], sim_dgp["Rate"]])[1:, :]

    if np.any(np.asarray(err_scale, dtype=float) != 0.0):
        rng = np.random.default_rng(next_seed(seed_counter))
        obs = obs + rng.normal(scale=np.sqrt(err_scale), size=obs.shape)

    return sim_dgp, obs


def summarize_reference_experiment(
    sol: Any,
    sol_dgp: Any,
    *,
    T: int,
    err_var: np.ndarray,
    meas_err_scale: float,
    mc_samples: int,
    known_r: bool,
    alpha: float = ALPHA_DEFAULT,
    summary_only: bool = False,
    include_by_predictor: bool = True,
) -> dict[str, Any]:
    err_scale = np.asarray(err_var, dtype=float) * float(meas_err_scale)
    filter_kwargs = make_filter_kwargs(err_scale, known_r=known_r)

    records = MCTableStore(mc_samples)
    representative: ReferenceRepresentative | None = None
    seed_counter = make_seed_counter()

    for replication in range(mc_samples):
        sim_dgp, obs = simulate_obs(sol_dgp, T=T, err_scale=err_scale, seed_counter=seed_counter)
        kf = sol.kalman(
            y=obs,
            filter_mode="linear",
            estimate_R_diag=False,
            **filter_kwargs,
        )
        std_innov = standardize_innovations(kf)

        records.append(ReferenceTable.LB, replication, lb_test_records(std_innov, alpha=alpha))

        records.append(
            ReferenceTable.MOMENT,
            replication,
            {"replication": replication, **summarize_moment_record(std_innov)},
        )
        records.append(
            ReferenceTable.MOMENT_SPEC,
            replication,
            moment_specification_test_records(std_innov, replication=replication),
        )

        reg_diag = run_full_regression_diagnostics(kf, include_single_predictor_reports=False)

        records.append(
            ReferenceTable.ORTHOGONALIZATION,
            replication,
            reg_diag["orthogonalization"].summary_records(),
        )

        records.append(
            ReferenceTable.RAW_REGRESSION,
            replication,
            reg_diag["measurement_regressions_raw"].records,
        )
        records.append(
            ReferenceTable.RAW_JOINT_REGRESSION,
            replication,
            reg_diag["measurement_joint_regressions_raw"].records,
        )
        records.append(
            ReferenceTable.RAW_JOINT_RELATIVE_WALD,
            replication,
            reg_diag["measurement_joint_regressions_raw"].relative_wald_records,
        )

        records.append(
            ReferenceTable.ORTH_REGRESSION,
            replication,
            reg_diag["measurement_regressions_orthogonalized"].records,
        )

        records.append(
            ReferenceTable.RAW_LAG_BLOCK,
            replication,
            reg_diag["measurement_lag_block_regressions_raw"].records,
        )

        records.append(
            ReferenceTable.ORTH_LAG_BLOCK,
            replication,
            reg_diag["measurement_lag_block_regressions_orthogonalized"].records,
        )

        records.append(
            ReferenceTable.RAW_LAG_COEFFICIENT,
            replication,
            reg_diag["measurement_lag_block_regressions_raw"].coefficient_records,
        )

        records.append(
            ReferenceTable.ORTH_LAG_COEFFICIENT,
            replication,
            reg_diag["measurement_lag_block_regressions_orthogonalized"].coefficient_records,
        )

        records.append(
            ReferenceTable.RAW_DECOMPOSITION,
            replication,
            innovation_decomposition_record_batch(
                sol,
                sol_dgp,
                sim_dgp,
                obs,
                kf,
                reg_diag["states"],
                replication=replication,
                structure="raw",
            ),
        )
        records.append(
            ReferenceTable.ORTH_DECOMPOSITION,
            replication,
            innovation_decomposition_record_batch(
                sol,
                sol_dgp,
                sim_dgp,
                obs,
                kf,
                reg_diag["orthogonalization"].residuals,
                replication=replication,
                structure="orthogonalized",
            ),
        )

        if representative is None:
            representative = ReferenceRepresentative(
                sim_dgp=sim_dgp,
                obs=obs,
                kf=kf,
                std_innov=std_innov,
                err_scale=err_scale.copy(),
            )

    if summary_only:
        out: dict[str, Any] = {
            "representative": representative,
            "filter_kwargs": filter_kwargs,
            "err_scale": err_scale,
        }

        lb_records = records.frame(ReferenceTable.LB)
        out["lb_summary"] = aggregate_scalar_frame(lb_records, ["measurement"], alpha=alpha)
        del lb_records

        moment_records = records.frame(ReferenceTable.MOMENT)
        out["moment_summary"] = summarize_average_moments(moment_records)
        out["moment_summary_mc_se"] = summarize_average_moment_mc_se(moment_records)
        out["moment_mean_vector"] = average_vector(moment_records, "mean")
        out["moment_mean_vector_mc_se"] = average_vector_mc_se(moment_records, "mean")
        out["moment_covariance"] = average_matrix(moment_records, "cov")
        out["moment_covariance_mc_se"] = average_matrix_mc_se(moment_records, "cov")
        out["moment_correlation"] = average_matrix(moment_records, "corr")
        out["moment_correlation_mc_se"] = average_matrix_mc_se(moment_records, "corr")
        del moment_records

        moment_spec_tests = records.frame(ReferenceTable.MOMENT_SPEC)
        out["moment_specification_test_summary"] = summarize_moment_specification_tests(
            moment_spec_tests,
            alpha=alpha,
        )
        del moment_spec_tests

        orth_records = records.frame(ReferenceTable.ORTHOGONALIZATION)
        out["orthogonalization_summary"] = add_rejection_summary(
            orth_records,
            ["target", "regressor"],
            alpha=alpha,
        )
        del orth_records

        raw_reg_records = records.frame(ReferenceTable.RAW_REGRESSION)
        out["measurement_regressions_raw_summary"] = add_rejection_summary(
            raw_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        )
        raw_joint_reg_records = records.frame(ReferenceTable.RAW_JOINT_REGRESSION)
        out["measurement_joint_regressions_raw_summary"] = add_rejection_summary(
            raw_joint_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        )
        del raw_joint_reg_records
        raw_joint_wald_records = records.frame(ReferenceTable.RAW_JOINT_RELATIVE_WALD)
        out["measurement_joint_relative_wald_raw_summary"] = add_rejection_summary(
            raw_joint_wald_records,
            ["measurement", "predictor_i", "predictor_j"],
            alpha=alpha,
        )
        del raw_joint_wald_records
        raw_lag_block_records = records.frame(ReferenceTable.RAW_LAG_BLOCK)
        out["measurement_lag_block_regressions_raw_summary"] = add_rejection_summary(
            raw_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        )
        out["measurement_regression_setup_comparison_raw"] = compare_regression_setups(
            raw_reg_records,
            raw_lag_block_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        )
        del raw_reg_records, raw_lag_block_records

        orth_reg_records = records.frame(ReferenceTable.ORTH_REGRESSION)
        out["measurement_regressions_orthogonalized_summary"] = add_rejection_summary(
            orth_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        )
        orth_lag_block_records = records.frame(ReferenceTable.ORTH_LAG_BLOCK)
        out["measurement_lag_block_regressions_orthogonalized_summary"] = add_rejection_summary(
            orth_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        )
        out["measurement_regression_setup_comparison_orthogonalized"] = compare_regression_setups(
            orth_reg_records,
            orth_lag_block_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        )
        del orth_reg_records, orth_lag_block_records

        raw_decomposition_records = records.frame(ReferenceTable.RAW_DECOMPOSITION)
        out["innovation_decomposition_raw_summary"] = aggregate_scalar_frame(
            raw_decomposition_records,
            ["measurement", "predictor"],
            value_cols=INNOVATION_DECOMPOSITION_VALUE_COLUMNS,
            alpha=alpha,
        )
        del raw_decomposition_records

        orth_decomposition_records = records.frame(ReferenceTable.ORTH_DECOMPOSITION)
        out["innovation_decomposition_orthogonalized_summary"] = aggregate_scalar_frame(
            orth_decomposition_records,
            ["measurement", "predictor"],
            value_cols=INNOVATION_DECOMPOSITION_VALUE_COLUMNS,
            alpha=alpha,
        )
        del orth_decomposition_records

        return out

    lb_records = records.frame(ReferenceTable.LB)
    moment_records = records.frame(ReferenceTable.MOMENT)
    moment_spec_tests = records.frame(ReferenceTable.MOMENT_SPEC)
    orth_records = records.frame(ReferenceTable.ORTHOGONALIZATION)
    raw_reg_records = records.frame(ReferenceTable.RAW_REGRESSION)
    raw_joint_reg_records = records.frame(ReferenceTable.RAW_JOINT_REGRESSION)
    raw_joint_wald_records = records.frame(ReferenceTable.RAW_JOINT_RELATIVE_WALD)
    orth_reg_records = records.frame(ReferenceTable.ORTH_REGRESSION)
    raw_lag_block_records = records.frame(ReferenceTable.RAW_LAG_BLOCK)
    orth_lag_block_records = records.frame(ReferenceTable.ORTH_LAG_BLOCK)
    raw_lag_block_coefficient_records = records.frame(ReferenceTable.RAW_LAG_COEFFICIENT)
    orth_lag_block_coefficient_records = records.frame(ReferenceTable.ORTH_LAG_COEFFICIENT)
    raw_decomposition_records = records.frame(ReferenceTable.RAW_DECOMPOSITION)
    orth_decomposition_records = records.frame(ReferenceTable.ORTH_DECOMPOSITION)

    out: dict[str, Any] = {
        "representative": representative,
        "filter_kwargs": filter_kwargs,
        "err_scale": err_scale,
        "lb_summary": aggregate_scalar_frame(lb_records, ["measurement"], alpha=alpha),
        "moment_summary": summarize_average_moments(moment_records),
        "moment_summary_mc_se": summarize_average_moment_mc_se(moment_records),
        "moment_mean_vector": average_vector(moment_records, "mean"),
        "moment_mean_vector_mc_se": average_vector_mc_se(moment_records, "mean"),
        "moment_covariance": average_matrix(moment_records, "cov"),
        "moment_covariance_mc_se": average_matrix_mc_se(moment_records, "cov"),
        "moment_correlation": average_matrix(moment_records, "corr"),
        "moment_correlation_mc_se": average_matrix_mc_se(moment_records, "corr"),
        "moment_specification_test_summary": summarize_moment_specification_tests(
            moment_spec_tests,
            alpha=alpha,
        ),
        "orthogonalization_summary": add_rejection_summary(
            orth_records,
            ["target", "regressor"],
            alpha=alpha,
        ),
        "measurement_regressions_raw_summary": add_rejection_summary(
            raw_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_joint_regressions_raw_summary": add_rejection_summary(
            raw_joint_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_joint_relative_wald_raw_summary": add_rejection_summary(
            raw_joint_wald_records,
            ["measurement", "predictor_i", "predictor_j"],
            alpha=alpha,
        ),
        "measurement_regressions_orthogonalized_summary": add_rejection_summary(
            orth_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_raw_summary": add_rejection_summary(
            raw_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_orthogonalized_summary": add_rejection_summary(
            orth_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "innovation_decomposition_raw_summary": aggregate_scalar_frame(
            raw_decomposition_records,
            ["measurement", "predictor"],
            value_cols=INNOVATION_DECOMPOSITION_VALUE_COLUMNS,
            alpha=alpha,
        ),
        "innovation_decomposition_orthogonalized_summary": aggregate_scalar_frame(
            orth_decomposition_records,
            ["measurement", "predictor"],
            value_cols=INNOVATION_DECOMPOSITION_VALUE_COLUMNS,
            alpha=alpha,
        ),
        "measurement_regression_setup_comparison_raw": compare_regression_setups(
            raw_reg_records,
            raw_lag_block_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_regression_setup_comparison_orthogonalized": compare_regression_setups(
            orth_reg_records,
            orth_lag_block_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        ),
    }

    if not summary_only:
        out.update(
            {
                "lb_records": lb_records,
                "moment_records": moment_records,
                "moment_specification_tests": moment_spec_tests,
                "orthogonalization_records": orth_records,
                "measurement_regressions_raw_records": raw_reg_records,
                "measurement_joint_regressions_raw_records": raw_joint_reg_records,
                "measurement_joint_relative_wald_raw_records": raw_joint_wald_records,
                "measurement_regressions_orthogonalized_records": orth_reg_records,
                "measurement_lag_block_regressions_raw_records": raw_lag_block_records,
                "measurement_lag_block_regressions_orthogonalized_records": orth_lag_block_records,
                "measurement_lag_block_coefficients_raw_records": raw_lag_block_coefficient_records,
                "measurement_lag_block_coefficients_orthogonalized_records": orth_lag_block_coefficient_records,
                "innovation_decomposition_raw_records": raw_decomposition_records,
                "innovation_decomposition_orthogonalized_records": orth_decomposition_records,
            }
        )
        if include_by_predictor:
            out.update(
                {
                    "measurement_regressions_raw_by_predictor": records_by_predictor(raw_reg_records),
                    "measurement_joint_regressions_raw_by_predictor": records_by_predictor(raw_joint_reg_records),
                    "measurement_regressions_orthogonalized_by_predictor": records_by_predictor(orth_reg_records),
                    "measurement_lag_block_regressions_raw_by_predictor": records_by_predictor(raw_lag_block_records),
                    "measurement_lag_block_regressions_orthogonalized_by_predictor": records_by_predictor(orth_lag_block_records),
                    "measurement_lag_block_coefficients_raw_by_predictor": records_by_predictor(raw_lag_block_coefficient_records),
                    "measurement_lag_block_coefficients_orthogonalized_by_predictor": records_by_predictor(orth_lag_block_coefficient_records),
                    "innovation_decomposition_raw_by_predictor": records_by_predictor(raw_decomposition_records),
                    "innovation_decomposition_orthogonalized_by_predictor": records_by_predictor(orth_decomposition_records),
                }
            )

    return out


def summarize_mle_augmentation_experiment(
    sol: Any,
    solver_aug: Any,
    comp_aug: Any,
    sol_dgp: Any,
    reference_summary: dict[str, Any],
    *,
    T: int,
    candidate_param: str,
    mc_samples: int,
    alpha: float = ALPHA_DEFAULT,
    summary_only: bool = False,
) -> dict[str, Any]:
    filter_kwargs = reference_summary["filter_kwargs"]
    representative_reference: ReferenceRepresentative = reference_summary["representative"]
    err_scale = np.asarray(reference_summary["err_scale"], dtype=float)
    reference_seed_counter = make_seed_counter()
    augmentation_seed_counter = make_seed_counter(start=1_000_000)

    lr_rows: list[dict[str, Any]] = []
    records = MCTableStore(mc_samples)
    representative: AugmentationRepresentative | None = None

    for replication in range(mc_samples):
        _, obs = simulate_obs(sol_dgp, T=T, err_scale=err_scale, seed_counter=reference_seed_counter)
        ref_kf = sol.kalman(
            y=obs,
            filter_mode="linear",
            estimate_R_diag=False,
            **filter_kwargs,
        )

        try:
            res_mle, sol_mle = solver_aug.estimate_and_solve(
                compiled=comp_aug,
                method="mle",
                y=obs,
                estimated_params=[candidate_param],
                steady_state=STEADY_STATE_DEFAULT,
                **filter_kwargs,
            )
        except Exception as exc:
            lr_rows.append(
                {
                    "replication": replication,
                    "estimated_coef": np.nan,
                    "loglik_ref": float(ref_kf.loglik),
                    "loglik_aug": np.nan,
                    "lr": np.nan,
                    "p_value": np.nan,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        try:
            kf_aug = sol_mle.kalman(
                y=obs,
                filter_mode="linear",
                estimate_R_diag=False,
                **filter_kwargs,
            )
            std_innov_aug = standardize_innovations(kf_aug)
            lr = 2.0 * (float(kf_aug.loglik) - float(ref_kf.loglik))
            p_value = 1.0 - chi2.cdf(lr, df=1)
            lr_rows.append(
                {
                    "replication": replication,
                    "estimated_coef": float(sol_mle.config.calibration.parameters[candidate_param]),
                    "loglik_ref": float(ref_kf.loglik),
                    "loglik_aug": float(kf_aug.loglik),
                    "lr": float(lr),
                    "p_value": float(p_value),
                    "success": True,
                }
            )
            records.append(AugmentationTable.LB, replication, lb_test_records(std_innov_aug, alpha=alpha))
            records.append(
                AugmentationTable.MOMENT,
                replication,
                {"replication": replication, **summarize_moment_record(std_innov_aug)},
            )
            records.append(
                AugmentationTable.MOMENT_SPEC,
                replication,
                moment_specification_test_records(std_innov_aug, replication=replication),
            )
        except Exception as exc:
            lr_rows.append(
                {
                    "replication": replication,
                    "estimated_coef": float(sol_mle.config.calibration.parameters[candidate_param]),
                    "loglik_ref": float(ref_kf.loglik),
                    "loglik_aug": np.nan,
                    "lr": np.nan,
                    "p_value": np.nan,
                    "success": True,
                    "filter_success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        if representative is None:
            sim_aug = sol_mle.sim(
                T=T,
                shocks={
                    "g,z": Shock(T, "norm", multivar=True, seed=next_seed(augmentation_seed_counter)).shock_generator(),
                    "r": Shock(T, "norm", multivar=False, seed=next_seed(augmentation_seed_counter)).shock_generator(),
                },
                observables=True,
            )
            representative = AugmentationRepresentative(
                res_mle=res_mle,
                sol_mle=sol_mle,
                kf_aug=kf_aug,
                std_innov_aug=std_innov_aug,
                sim_aug=sim_aug,
            )

    lr_records = pd.DataFrame(lr_rows)
    lb_records = records.frame(AugmentationTable.LB)
    if lb_records.empty:
        lb_records = pd.DataFrame(columns=["measurement", "lb_stat", "p_value", "replication"])
    moment_records = records.frame(AugmentationTable.MOMENT)
    moment_spec_tests = records.frame(AugmentationTable.MOMENT_SPEC)
    reference_moment_spec_tests = reference_summary.get("moment_specification_tests", pd.DataFrame())

    out: dict[str, Any] = {
        "representative": representative,
        "lr_summary": aggregate_scalar_frame(lr_records.dropna(subset=["p_value"]), [], alpha=alpha) if not lr_records.dropna(subset=["p_value"]).empty else pd.DataFrame(),
        "lb_summary": aggregate_scalar_frame(lb_records, ["measurement"], alpha=alpha) if not lb_records.empty else pd.DataFrame(),
        "moment_summary": summarize_average_moments(moment_records) if not moment_records.empty else pd.Series(dtype=float),
        "moment_summary_mc_se": summarize_average_moment_mc_se(moment_records) if not moment_records.empty else pd.Series(dtype=float),
        "moment_mean_vector": average_vector(moment_records, "mean") if not moment_records.empty else pd.Series(dtype=float),
        "moment_mean_vector_mc_se": average_vector_mc_se(moment_records, "mean") if not moment_records.empty else pd.Series(dtype=float),
        "moment_covariance": average_matrix(moment_records, "cov") if not moment_records.empty else pd.DataFrame(),
        "moment_covariance_mc_se": average_matrix_mc_se(moment_records, "cov") if not moment_records.empty else pd.DataFrame(),
        "moment_correlation": average_matrix(moment_records, "corr") if not moment_records.empty else pd.DataFrame(),
        "moment_correlation_mc_se": average_matrix_mc_se(moment_records, "corr") if not moment_records.empty else pd.DataFrame(),
        "moment_specification_test_summary": summarize_moment_specification_tests(
            moment_spec_tests,
            alpha=alpha,
        ) if not moment_spec_tests.empty else pd.DataFrame(),
        "moment_specification_comparison": summarize_moment_specification_comparison(
            reference_moment_spec_tests,
            moment_spec_tests,
            alpha=alpha,
        ) if not moment_spec_tests.empty and not reference_moment_spec_tests.empty else pd.DataFrame(),
    }

    if not summary_only:
        out.update(
            {
                "lr_records": lr_records,
                "lb_records": lb_records,
                "moment_records": moment_records,
                "moment_specification_tests": moment_spec_tests,
            }
        )

    return out
