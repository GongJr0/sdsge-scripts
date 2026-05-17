from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Number
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ReferenceTable(Enum):
    LB = "lb_records"
    MOMENT = "moment_records"
    MOMENT_SPEC = "moment_specification_tests"
    ORTHOGONALIZATION = "orthogonalization_records"
    RAW_REGRESSION = "measurement_regressions_raw_records"
    ORTH_REGRESSION = "measurement_regressions_orthogonalized_records"
    RAW_LAG_BLOCK = "measurement_lag_block_regressions_raw_records"
    ORTH_LAG_BLOCK = "measurement_lag_block_regressions_orthogonalized_records"
    RAW_LAG_COEFFICIENT = "measurement_lag_block_coefficients_raw_records"
    ORTH_LAG_COEFFICIENT = "measurement_lag_block_coefficients_orthogonalized_records"
    RAW_DECOMPOSITION = "innovation_decomposition_raw_records"
    ORTH_DECOMPOSITION = "innovation_decomposition_orthogonalized_records"


class AugmentationTable(Enum):
    LR = "lr_records"
    LB = "lb_records"
    MOMENT = "moment_records"
    MOMENT_SPEC = "moment_specification_tests"


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (Number, np.number, bool, np.bool_)) and not isinstance(value, str)


@dataclass(frozen=True)
class MCRecordBatch:
    columns: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
    ) -> "MCRecordBatch":
        if columns is None:
            if not records:
                columns = ()
            else:
                columns = tuple(records[0].keys())
        return cls(columns=tuple(columns), records=tuple(records))

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        columns: Sequence[str] | None = None,
    ) -> "MCRecordBatch":
        return cls.from_records([record], columns=columns)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(record) for record in self.records], columns=list(self.columns))


@dataclass
class MCFrameStore:
    n_replications: int
    columns: list[str] | None = None
    numeric_columns: list[str] = field(default_factory=list)
    static_columns: list[str] = field(default_factory=list)
    dtypes: dict[str, Any] = field(default_factory=dict)
    static_frame: pd.DataFrame | None = None
    values: np.ndarray | None = None
    filled: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.filled = np.zeros((self.n_replications,), dtype=bool)

    def append(self, replication: int, frame: pd.DataFrame | Mapping[str, Any] | MCRecordBatch) -> None:
        if not 0 <= replication < self.n_replications:
            raise IndexError(f"replication {replication} is outside [0, {self.n_replications})")

        if isinstance(frame, MCRecordBatch):
            self._append_batch(replication, frame)
            return

        if isinstance(frame, Mapping):
            self._append_batch(replication, MCRecordBatch.from_record(frame))
            return

        if isinstance(frame, dict):
            frame = pd.DataFrame([frame])
        else:
            frame = frame.copy()
        if "replication" not in frame.columns:
            frame["replication"] = replication
        frame = frame.reset_index(drop=True)

        if self.columns is None:
            self._initialize(frame)
        else:
            missing = set(self.columns).difference(frame.columns)
            extra = set(frame.columns).difference(self.columns)
            if missing or extra:
                raise ValueError(
                    f"Frame columns changed. Missing={sorted(missing)}, extra={sorted(extra)}"
                )
            frame = frame[self.columns]
            if frame.shape[0] != self.values.shape[1]:
                raise ValueError(
                    f"Frame row count changed from {self.values.shape[1]} to {frame.shape[0]}"
                )
            self._check_static_columns(frame)

        self.values[replication, :, :] = frame[self.numeric_columns].to_numpy(dtype=float)
        self.filled[replication] = True

    def _append_batch(self, replication: int, batch: MCRecordBatch) -> None:
        if len(batch.records) == 0:
            raise ValueError("Cannot append an empty MCRecordBatch.")

        if "replication" in batch.columns:
            columns = list(batch.columns)
        else:
            columns = [*batch.columns, "replication"]

        if self.columns is None:
            self._initialize_from_batch(batch, columns)
        else:
            missing = set(self.columns).difference(columns)
            extra = set(columns).difference(self.columns)
            if missing or extra:
                raise ValueError(
                    f"Frame columns changed. Missing={sorted(missing)}, extra={sorted(extra)}"
                )
            columns = self.columns
            if len(batch.records) != self.values.shape[1]:
                raise ValueError(
                    f"Frame row count changed from {self.values.shape[1]} to {len(batch.records)}"
                )
            self._check_static_columns_batch(batch, columns, replication)

        self.values[replication, :, :] = self._batch_numeric_values(batch, columns, replication)
        self.filled[replication] = True

    def _initialize_from_batch(self, batch: MCRecordBatch, columns: list[str]) -> None:
        self.columns = columns
        self.dtypes = {}
        self.numeric_columns = []
        self.static_columns = []
        for col in self.columns:
            values = [
                record.get(col, None) if col != "replication" else 0
                for record in batch.records
            ]
            first_value = values[0]
            if _is_numeric_value(first_value):
                self.numeric_columns.append(col)
                if any(isinstance(value, (float, np.floating)) for value in values):
                    self.dtypes[col] = np.dtype(np.float64)
                elif any(isinstance(value, (bool, np.bool_)) for value in values):
                    self.dtypes[col] = np.dtype(np.bool_)
                else:
                    self.dtypes[col] = np.dtype(np.int64)
            else:
                self.static_columns.append(col)
                self.dtypes[col] = object

        self.static_frame = pd.DataFrame(
            {
                col: [
                    record.get(col, None) if col != "replication" else 0
                    for record in batch.records
                ]
                for col in self.static_columns
            },
            columns=self.static_columns,
        )
        self.values = np.full(
            (self.n_replications, len(batch.records), len(self.numeric_columns)),
            np.nan,
            dtype=np.float64,
        )

    def _batch_numeric_values(
        self,
        batch: MCRecordBatch,
        columns: list[str],
        replication: int,
    ) -> np.ndarray:
        values = np.empty((len(batch.records), len(self.numeric_columns)), dtype=np.float64)
        for col_idx, col in enumerate(self.numeric_columns):
            if col == "replication" and col not in batch.columns:
                values[:, col_idx] = float(replication)
            else:
                values[:, col_idx] = [record[col] for record in batch.records]
        return values

    def _check_static_columns_batch(
        self,
        batch: MCRecordBatch,
        columns: list[str],
        replication: int,
    ) -> None:
        if not self.static_columns:
            return
        assert self.static_frame is not None
        current = pd.DataFrame(
            {
                col: [
                    record.get(col, None) if col != "replication" else replication
                    for record in batch.records
                ]
                for col in self.static_columns
            },
            columns=self.static_columns,
        )
        if not current.equals(self.static_frame):
            raise ValueError("Non-numeric frame labels changed across replications.")

    def _initialize(self, frame: pd.DataFrame) -> None:
        self.columns = list(frame.columns)
        self.dtypes = {col: frame[col].dtype for col in self.columns}
        self.numeric_columns = [
            col for col in self.columns if pd.api.types.is_numeric_dtype(frame[col])
        ]
        self.static_columns = [col for col in self.columns if col not in self.numeric_columns]
        self.static_frame = frame[self.static_columns].reset_index(drop=True)
        self.values = np.full(
            (self.n_replications, frame.shape[0], len(self.numeric_columns)),
            np.nan,
            dtype=np.float64,
        )

    def _check_static_columns(self, frame: pd.DataFrame) -> None:
        if not self.static_columns:
            return
        assert self.static_frame is not None
        current = frame[self.static_columns].reset_index(drop=True)
        if not current.equals(self.static_frame):
            raise ValueError("Non-numeric frame labels changed across replications.")

    def to_frame(self) -> pd.DataFrame:
        if self.columns is None or self.values is None:
            return pd.DataFrame()

        reps = np.flatnonzero(self.filled)
        if reps.size == 0:
            return pd.DataFrame(columns=self.columns)

        row_count = self.values.shape[1]
        out: dict[str, Any] = {}
        static_frame = self.static_frame
        flat_values = self.values[reps].reshape(reps.size * row_count, len(self.numeric_columns))

        for col in self.columns:
            if col in self.numeric_columns:
                idx = self.numeric_columns.index(col)
                out[col] = flat_values[:, idx]
            else:
                assert static_frame is not None
                out[col] = np.tile(static_frame[col].to_numpy(), reps.size)

        df = pd.DataFrame(out, columns=self.columns)
        for col, dtype in self.dtypes.items():
            try:
                df[col] = df[col].astype(dtype, copy=False)
            except (TypeError, ValueError):
                pass
        return df


@dataclass
class MCTableStore:
    n_replications: int
    tables: dict[Enum, MCFrameStore] = field(default_factory=dict)

    def append(
        self,
        table: Enum,
        replication: int,
        frame: pd.DataFrame | Mapping[str, Any] | MCRecordBatch,
    ) -> None:
        if table not in self.tables:
            self.tables[table] = MCFrameStore(self.n_replications)
        self.tables[table].append(replication, frame)

    def frame(self, table: Enum) -> pd.DataFrame:
        store = self.tables.get(table)
        if store is None:
            return pd.DataFrame()
        return store.to_frame()
