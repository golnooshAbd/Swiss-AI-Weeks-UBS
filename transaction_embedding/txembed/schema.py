from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Column names and deterministic feature conventions.

    ``day_of_week`` is expected to be 0..6. Day of month and month are expected
    to be 1-based. Change the offsets if the upstream cleaner uses a different
    convention; the selected values are persisted with preprocessing artifacts.
    """

    client_id: str = "client_id"
    timestamp: str = "timestamp"
    # Embed the normalized semantic text produced by dataset_cleaning.
    description: str = "clean_description"

    categorical: tuple[str, ...] = (
        "candidate_family",
        "mcc",
        "type",
        "currency",
        "direction",
    )
    categorical_dims: tuple[int, ...] = (8, 8, 6, 4, 2)

    log_continuous: tuple[str, ...] = (
        "amount",
        "days_to_cutoff",
        "days_since_prev_transaction",
        "days_since_prev_same_family",
        "count_same_family_before",
        "median_interval_same_family",
        "std_interval_same_family",
        "median_amount_same_family",
    )
    plain_continuous: tuple[str, ...] = ("amount_vs_family_median",)

    # These values depend on prior transactions. A missing bit is retained after
    # the numeric value is filled with zero.
    history: tuple[str, ...] = (
        "days_since_prev_transaction",
        "days_since_prev_same_family",
        "count_same_family_before",
        "median_interval_same_family",
        "std_interval_same_family",
        "median_amount_same_family",
        "amount_vs_family_median",
    )

    calendar: tuple[str, ...] = ("day_of_week", "day_of_month", "month")
    calendar_periods: tuple[int, ...] = (7, 31, 12)
    calendar_offsets: tuple[int, ...] = (0, 1, 1)

    @property
    def continuous(self) -> tuple[str, ...]:
        return self.log_continuous + self.plain_continuous

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.client_id,
            self.timestamp,
            self.description,
            *self.categorical,
            *self.continuous,
            *self.calendar,
        )

    @property
    def dense_feature_names(self) -> tuple[str, ...]:
        cyclic = tuple(f"{name}_{part}" for name in self.calendar for part in ("sin", "cos"))
        missing = tuple(f"{name}_missing" for name in self.history)
        return (*self.continuous, *cyclic, *missing)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        tuple_fields = {
            "categorical",
            "categorical_dims",
            "log_continuous",
            "plain_continuous",
            "history",
            "calendar",
            "calendar_periods",
            "calendar_offsets",
        }
        return cls(**{key: tuple(value) if key in tuple_fields else value for key, value in data.items()})
