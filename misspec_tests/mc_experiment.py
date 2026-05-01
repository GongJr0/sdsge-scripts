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
    if np.any(np.asarray(err_scale, dtype=float) != 0.0):
        return {"R": np.zeros((n_obs, n_obs))}
    return {}


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


def size_adjusted_power_comparison(
    size_records_by_setup: Mapping[str, pd.DataFrame],
    power_records_by_setup: Mapping[str, pd.DataFrame],
    *,
    group_cols: list[str],
    alpha: float = ALPHA_DEFAULT,
    p_value_col: str = "p_value",
) -> pd.DataFrame:
    """Calibrate each setup's p-value cutoff on size records, then apply it to power records."""
    rows = []
    for setup_name, size_records in size_records_by_setup.items():
        power_records = power_records_by_setup.get(setup_name)
        if power_records is None or size_records.empty or power_records.empty:
            continue

        thresholds = (
            size_records.dropna(subset=[p_value_col])
            .groupby(group_cols, sort=False, dropna=False)[p_value_col]
            .quantile(alpha)
            .reset_index(name="size_adjusted_p_value_cutoff")
        )
        if thresholds.empty:
            continue

        size_eval = size_records.merge(thresholds, on=group_cols, how="inner")
        power_eval = power_records.merge(thresholds, on=group_cols, how="inner")
        size_eval["size_reject"] = size_eval[p_value_col] <= size_eval["size_adjusted_p_value_cutoff"]
        power_eval["power_reject"] = power_eval[p_value_col] <= power_eval["size_adjusted_p_value_cutoff"]

        for key, power_group in power_eval.groupby(group_cols, sort=False, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            selector = np.ones(len(size_eval), dtype=bool)
            for col, value in zip(group_cols, key_tuple):
                selector &= size_eval[col].eq(value).to_numpy()
            size_group = size_eval.loc[selector]
            n_power = int(len(power_group))
            n_power_reject = int(power_group["power_reject"].sum())
            ci_low, ci_high = wilson_interval(n_power_reject, n_power, alpha=alpha)
            row = {
                col: value for col, value in zip(group_cols, key_tuple)
            }
            row.update(
                {
                    "setup": setup_name,
                    "size_adjusted_p_value_cutoff": float(power_group["size_adjusted_p_value_cutoff"].iloc[0]),
                    "empirical_size": float(size_group["size_reject"].mean()),
                    "n_size_replications": int(len(size_group)),
                    "size_adjusted_power": n_power_reject / n_power if n_power else np.nan,
                    "size_adjusted_power_mc_se": rejection_rate_standard_error(n_power_reject, n_power),
                    "size_adjusted_power_ci_low": ci_low,
                    "size_adjusted_power_ci_high": ci_high,
                    "n_power_replications": n_power,
                    "n_power_rejections": n_power_reject,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


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
    off_diag_covariance = covariance.copy()
    np.fill_diagonal(off_diag_covariance.values, 0.0)

    max_mean_measurement = mean_vector.abs().idxmax()
    diag_abs = (pd.Series(np.diag(covariance), index=MEASUREMENT_NAMES) - 1.0).abs()
    max_diag_measurement = diag_abs.idxmax()
    offdiag_abs = off_diag_covariance.abs()
    np.fill_diagonal(offdiag_abs.values, np.nan)
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

    omega_hat = (moment_process.T @ moment_process) / sample_size
    max_lag = min(int(bandwidth), sample_size - 1)
    for lag in range(1, max_lag + 1):
        gamma_l = (moment_process[lag:].T @ moment_process[:-lag]) / sample_size
        weight = 1.0 - (lag / (max_lag + 1.0))
        # Standard vector Newey-West HAC uses the symmetrized positive-lag autocovariances.
        omega_hat = omega_hat + weight * (gamma_l + gamma_l.T)

    return 0.5 * (omega_hat + omega_hat.T)


def moment_specification_test_frame(std_innov: np.ndarray, *, replication: int) -> pd.DataFrame:
    std_innov = np.asarray(std_innov, dtype=float)
    sample_size, k = std_innov.shape
    cov_df = k * (k + 1) / 2

    mean_vector = std_innov.mean(axis=0)
    covariance = np.cov(std_innov, rowvar=False)

    mean_distance = float(np.linalg.norm(mean_vector))
    try:
        covariance_inv = np.linalg.inv(covariance)
        mean_stat = float(sample_size * mean_vector @ covariance_inv @ mean_vector)
    except np.linalg.LinAlgError:
        mean_stat = float(sample_size * mean_vector @ np.linalg.pinv(covariance) @ mean_vector)
    mean_p_value = float(chi2.sf(mean_stat, df=k))

    identity = np.eye(k)
    second_moment = (std_innov.T @ std_innov) / sample_size
    moment_process = np.array(
        [_vech(np.outer(u_t, u_t) - identity) for u_t in std_innov],
        dtype=float,
    )
    gbar = moment_process.mean(axis=0)
    bandwidth = bartlett_hac_bandwidth(sample_size)
    omega_hat = newey_west_long_run_covariance(moment_process, bandwidth=bandwidth)
    try:
        omega_inv = np.linalg.inv(omega_hat)
    except np.linalg.LinAlgError:
        omega_inv = np.linalg.pinv(omega_hat)
    cov_stat = float(sample_size * gbar @ omega_inv @ gbar)
    cov_p_value = float(chi2.sf(cov_stat, df=cov_df))
    cov_distance = float(np.linalg.norm(second_moment - identity, ord="fro"))

    return pd.DataFrame(
        [
            {
                "replication": replication,
                "test": "mean_zero_chi2",
                "df": k,
                "sample_size": sample_size,
                "bandwidth": np.nan,
                "distance": mean_distance,
                "stat": mean_stat,
                "p_value": mean_p_value,
            },
            {
                "replication": replication,
                "test": "cov_identity",
                "df": cov_df,
                "sample_size": sample_size,
                "bandwidth": bandwidth,
                "distance": cov_distance,
                "stat": cov_stat,
                "p_value": cov_p_value,
            },
        ]
    )


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


def lb_test_frame(std_innov: np.ndarray, *, alpha: float = ALPHA_DEFAULT, lags: int = LJUNG_BOX_LAGS) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


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
) -> dict[str, Any]:
    err_scale = np.asarray(err_var, dtype=float) * float(meas_err_scale)
    filter_kwargs = make_filter_kwargs(err_scale, known_r=known_r)

    lb_rows: list[pd.DataFrame] = []
    moment_rows: list[dict[str, float]] = []
    moment_spec_rows: list[pd.DataFrame] = []
    orth_rows: list[pd.DataFrame] = []
    raw_reg_rows: list[pd.DataFrame] = []
    orth_reg_rows: list[pd.DataFrame] = []
    raw_reg_no_intercept_rows: list[pd.DataFrame] = []
    orth_reg_no_intercept_rows: list[pd.DataFrame] = []
    raw_lag_block_rows: list[pd.DataFrame] = []
    orth_lag_block_rows: list[pd.DataFrame] = []
    raw_lag_block_no_intercept_rows: list[pd.DataFrame] = []
    orth_lag_block_no_intercept_rows: list[pd.DataFrame] = []
    raw_lag_block_coefficient_rows: list[pd.DataFrame] = []
    orth_lag_block_coefficient_rows: list[pd.DataFrame] = []
    raw_lag_block_no_intercept_coefficient_rows: list[pd.DataFrame] = []
    orth_lag_block_no_intercept_coefficient_rows: list[pd.DataFrame] = []
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

        lb_df = lb_test_frame(std_innov, alpha=alpha)
        lb_df["replication"] = replication
        lb_rows.append(lb_df)

        moment_rows.append({"replication": replication, **summarize_moment_record(std_innov)})
        moment_spec_rows.append(moment_specification_test_frame(std_innov, replication=replication))

        reg_diag = run_full_regression_diagnostics(kf)

        orth_df = reg_diag["orthogonalization"].summary_table(round_to=None)
        orth_df["replication"] = replication
        orth_rows.append(orth_df)

        raw_df = reg_diag["measurement_regressions_raw"].raw.copy()
        raw_df["replication"] = replication
        raw_reg_rows.append(raw_df)

        orth_reg_df = reg_diag["measurement_regressions_orthogonalized"].raw.copy()
        orth_reg_df["replication"] = replication
        orth_reg_rows.append(orth_reg_df)

        raw_no_intercept_df = reg_diag["measurement_regressions_raw_no_intercept"].raw.copy()
        raw_no_intercept_df["replication"] = replication
        raw_reg_no_intercept_rows.append(raw_no_intercept_df)

        orth_no_intercept_df = reg_diag["measurement_regressions_orthogonalized_no_intercept"].raw.copy()
        orth_no_intercept_df["replication"] = replication
        orth_reg_no_intercept_rows.append(orth_no_intercept_df)

        raw_lag_block_df = reg_diag["measurement_lag_block_regressions_raw"].raw.copy()
        raw_lag_block_df["replication"] = replication
        raw_lag_block_rows.append(raw_lag_block_df)

        orth_lag_block_df = reg_diag["measurement_lag_block_regressions_orthogonalized"].raw.copy()
        orth_lag_block_df["replication"] = replication
        orth_lag_block_rows.append(orth_lag_block_df)

        raw_lag_block_no_intercept_df = reg_diag["measurement_lag_block_regressions_raw_no_intercept"].raw.copy()
        raw_lag_block_no_intercept_df["replication"] = replication
        raw_lag_block_no_intercept_rows.append(raw_lag_block_no_intercept_df)

        orth_lag_block_no_intercept_df = reg_diag["measurement_lag_block_regressions_orthogonalized_no_intercept"].raw.copy()
        orth_lag_block_no_intercept_df["replication"] = replication
        orth_lag_block_no_intercept_rows.append(orth_lag_block_no_intercept_df)

        raw_lag_block_coefs_df = reg_diag["measurement_lag_block_regressions_raw"].coefficients.copy()
        raw_lag_block_coefs_df["replication"] = replication
        raw_lag_block_coefficient_rows.append(raw_lag_block_coefs_df)

        orth_lag_block_coefs_df = reg_diag["measurement_lag_block_regressions_orthogonalized"].coefficients.copy()
        orth_lag_block_coefs_df["replication"] = replication
        orth_lag_block_coefficient_rows.append(orth_lag_block_coefs_df)

        raw_lag_block_no_intercept_coefs_df = reg_diag["measurement_lag_block_regressions_raw_no_intercept"].coefficients.copy()
        raw_lag_block_no_intercept_coefs_df["replication"] = replication
        raw_lag_block_no_intercept_coefficient_rows.append(raw_lag_block_no_intercept_coefs_df)

        orth_lag_block_no_intercept_coefs_df = reg_diag["measurement_lag_block_regressions_orthogonalized_no_intercept"].coefficients.copy()
        orth_lag_block_no_intercept_coefs_df["replication"] = replication
        orth_lag_block_no_intercept_coefficient_rows.append(orth_lag_block_no_intercept_coefs_df)

        if representative is None:
            representative = ReferenceRepresentative(
                sim_dgp=sim_dgp,
                obs=obs,
                kf=kf,
                std_innov=std_innov,
                err_scale=err_scale.copy(),
            )

    lb_records = pd.concat(lb_rows, ignore_index=True)
    moment_records = pd.DataFrame(moment_rows)
    moment_spec_tests = pd.concat(moment_spec_rows, ignore_index=True)
    orth_records = pd.concat(orth_rows, ignore_index=True)
    raw_reg_records = pd.concat(raw_reg_rows, ignore_index=True)
    orth_reg_records = pd.concat(orth_reg_rows, ignore_index=True)
    raw_reg_no_intercept_records = pd.concat(raw_reg_no_intercept_rows, ignore_index=True)
    orth_reg_no_intercept_records = pd.concat(orth_reg_no_intercept_rows, ignore_index=True)
    raw_lag_block_records = pd.concat(raw_lag_block_rows, ignore_index=True)
    orth_lag_block_records = pd.concat(orth_lag_block_rows, ignore_index=True)
    raw_lag_block_no_intercept_records = pd.concat(raw_lag_block_no_intercept_rows, ignore_index=True)
    orth_lag_block_no_intercept_records = pd.concat(orth_lag_block_no_intercept_rows, ignore_index=True)
    raw_lag_block_coefficient_records = pd.concat(raw_lag_block_coefficient_rows, ignore_index=True)
    orth_lag_block_coefficient_records = pd.concat(orth_lag_block_coefficient_rows, ignore_index=True)
    raw_lag_block_no_intercept_coefficient_records = pd.concat(raw_lag_block_no_intercept_coefficient_rows, ignore_index=True)
    orth_lag_block_no_intercept_coefficient_records = pd.concat(orth_lag_block_no_intercept_coefficient_rows, ignore_index=True)

    return {
        "representative": representative,
        "filter_kwargs": filter_kwargs,
        "err_scale": err_scale,
        "lb_records": lb_records,
        "lb_summary": aggregate_scalar_frame(lb_records, ["measurement"], alpha=alpha),
        "moment_records": moment_records,
        "moment_summary": summarize_average_moments(moment_records),
        "moment_summary_mc_se": summarize_average_moment_mc_se(moment_records),
        "moment_mean_vector": average_vector(moment_records, "mean"),
        "moment_mean_vector_mc_se": average_vector_mc_se(moment_records, "mean"),
        "moment_covariance": average_matrix(moment_records, "cov"),
        "moment_covariance_mc_se": average_matrix_mc_se(moment_records, "cov"),
        "moment_correlation": average_matrix(moment_records, "corr"),
        "moment_correlation_mc_se": average_matrix_mc_se(moment_records, "corr"),
        "moment_specification_tests": moment_spec_tests,
        "moment_specification_test_summary": summarize_moment_specification_tests(
            moment_spec_tests,
            alpha=alpha,
        ),
        "orthogonalization_records": orth_records,
        "orthogonalization_summary": add_rejection_summary(
            orth_records,
            ["target", "regressor"],
            alpha=alpha,
        ),
        "measurement_regressions_raw_records": raw_reg_records,
        "measurement_regressions_raw_by_predictor": records_by_predictor(raw_reg_records),
        "measurement_regressions_raw_summary": add_rejection_summary(
            raw_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_regressions_orthogonalized_records": orth_reg_records,
        "measurement_regressions_orthogonalized_by_predictor": records_by_predictor(orth_reg_records),
        "measurement_regressions_orthogonalized_summary": add_rejection_summary(
            orth_reg_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_regressions_raw_no_intercept_records": raw_reg_no_intercept_records,
        "measurement_regressions_raw_no_intercept_by_predictor": records_by_predictor(raw_reg_no_intercept_records),
        "measurement_regressions_raw_no_intercept_summary": add_rejection_summary(
            raw_reg_no_intercept_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_regressions_orthogonalized_no_intercept_records": orth_reg_no_intercept_records,
        "measurement_regressions_orthogonalized_no_intercept_by_predictor": records_by_predictor(orth_reg_no_intercept_records),
        "measurement_regressions_orthogonalized_no_intercept_summary": add_rejection_summary(
            orth_reg_no_intercept_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_raw_records": raw_lag_block_records,
        "measurement_lag_block_regressions_raw_by_predictor": records_by_predictor(raw_lag_block_records),
        "measurement_lag_block_regressions_raw_summary": add_rejection_summary(
            raw_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_orthogonalized_records": orth_lag_block_records,
        "measurement_lag_block_regressions_orthogonalized_by_predictor": records_by_predictor(orth_lag_block_records),
        "measurement_lag_block_regressions_orthogonalized_summary": add_rejection_summary(
            orth_lag_block_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_raw_no_intercept_records": raw_lag_block_no_intercept_records,
        "measurement_lag_block_regressions_raw_no_intercept_by_predictor": records_by_predictor(raw_lag_block_no_intercept_records),
        "measurement_lag_block_regressions_raw_no_intercept_summary": add_rejection_summary(
            raw_lag_block_no_intercept_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_regressions_orthogonalized_no_intercept_records": orth_lag_block_no_intercept_records,
        "measurement_lag_block_regressions_orthogonalized_no_intercept_by_predictor": records_by_predictor(orth_lag_block_no_intercept_records),
        "measurement_lag_block_regressions_orthogonalized_no_intercept_summary": add_rejection_summary(
            orth_lag_block_no_intercept_records,
            ["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_lag_block_coefficients_raw_records": raw_lag_block_coefficient_records,
        "measurement_lag_block_coefficients_raw_by_predictor": records_by_predictor(raw_lag_block_coefficient_records),
        "measurement_lag_block_coefficients_orthogonalized_records": orth_lag_block_coefficient_records,
        "measurement_lag_block_coefficients_orthogonalized_by_predictor": records_by_predictor(orth_lag_block_coefficient_records),
        "measurement_lag_block_coefficients_raw_no_intercept_records": raw_lag_block_no_intercept_coefficient_records,
        "measurement_lag_block_coefficients_raw_no_intercept_by_predictor": records_by_predictor(raw_lag_block_no_intercept_coefficient_records),
        "measurement_lag_block_coefficients_orthogonalized_no_intercept_records": orth_lag_block_no_intercept_coefficient_records,
        "measurement_lag_block_coefficients_orthogonalized_no_intercept_by_predictor": records_by_predictor(orth_lag_block_no_intercept_coefficient_records),
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
        "measurement_regression_setup_comparison_raw_no_intercept": compare_regression_setups(
            raw_reg_no_intercept_records,
            raw_lag_block_no_intercept_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        ),
        "measurement_regression_setup_comparison_orthogonalized_no_intercept": compare_regression_setups(
            orth_reg_no_intercept_records,
            orth_lag_block_no_intercept_records,
            baseline_name="contemporaneous",
            candidate_name="lag_block",
            group_cols=["measurement", "predictor"],
            alpha=alpha,
        ),
    }


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
) -> dict[str, Any]:
    filter_kwargs = reference_summary["filter_kwargs"]
    representative_reference: ReferenceRepresentative = reference_summary["representative"]
    err_scale = np.asarray(reference_summary["err_scale"], dtype=float)
    reference_seed_counter = make_seed_counter()
    augmentation_seed_counter = make_seed_counter(start=1_000_000)

    lr_rows: list[dict[str, Any]] = []
    lb_rows: list[pd.DataFrame] = []
    moment_rows: list[dict[str, float]] = []
    moment_spec_rows: list[pd.DataFrame] = []
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
            lb_df = lb_test_frame(std_innov_aug, alpha=alpha)
            lb_df["replication"] = replication
            lb_rows.append(lb_df)
            moment_rows.append({"replication": replication, **summarize_moment_record(std_innov_aug)})
            moment_spec_rows.append(moment_specification_test_frame(std_innov_aug, replication=replication))
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
    lb_records = pd.concat(lb_rows, ignore_index=True) if lb_rows else pd.DataFrame(columns=["measurement", "lb_stat", "p_value", "replication"])
    moment_records = pd.DataFrame(moment_rows)
    moment_spec_tests = (
        pd.concat(moment_spec_rows, ignore_index=True)
        if moment_spec_rows
        else pd.DataFrame()
    )
    reference_moment_spec_tests = reference_summary.get("moment_specification_tests", pd.DataFrame())

    return {
        "representative": representative,
        "lr_records": lr_records,
        "lr_summary": aggregate_scalar_frame(lr_records.dropna(subset=["p_value"]), [], alpha=alpha) if not lr_records.dropna(subset=["p_value"]).empty else pd.DataFrame(),
        "lb_records": lb_records,
        "lb_summary": aggregate_scalar_frame(lb_records, ["measurement"], alpha=alpha) if not lb_records.empty else pd.DataFrame(),
        "moment_records": moment_records,
        "moment_summary": summarize_average_moments(moment_records) if not moment_records.empty else pd.Series(dtype=float),
        "moment_summary_mc_se": summarize_average_moment_mc_se(moment_records) if not moment_records.empty else pd.Series(dtype=float),
        "moment_mean_vector": average_vector(moment_records, "mean") if not moment_records.empty else pd.Series(dtype=float),
        "moment_mean_vector_mc_se": average_vector_mc_se(moment_records, "mean") if not moment_records.empty else pd.Series(dtype=float),
        "moment_covariance": average_matrix(moment_records, "cov") if not moment_records.empty else pd.DataFrame(),
        "moment_covariance_mc_se": average_matrix_mc_se(moment_records, "cov") if not moment_records.empty else pd.DataFrame(),
        "moment_correlation": average_matrix(moment_records, "corr") if not moment_records.empty else pd.DataFrame(),
        "moment_correlation_mc_se": average_matrix_mc_se(moment_records, "corr") if not moment_records.empty else pd.DataFrame(),
        "moment_specification_tests": moment_spec_tests,
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
