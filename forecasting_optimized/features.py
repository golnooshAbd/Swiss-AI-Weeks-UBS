from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FAMILIES: tuple[str, ...] = (
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
)
LABELS: tuple[str, ...] = (*FAMILIES, "none")
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
WINDOWS: tuple[int, ...] = (30, 60, 90, 180, 365)
STREAM_FEATURES: tuple[str, ...] = (
    "count",
    "days_since_last",
    "history_span",
    "gap_last",
    "gap_median",
    "gap_std",
    "gap_cv",
    "cycles_since_last",
    "projected_days_to_next",
    "projected_in_horizon",
    "amount_mean",
    "amount_cv",
    "description_unique",
    "description_mode_share",
)


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return np.nan
    return numerator / denominator


def _median_absolute_deviation(values: np.ndarray) -> float:
    if not len(values):
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _amount_clusters(frame: pd.DataFrame, relative_tolerance: float = 0.12) -> list[pd.DataFrame]:
    """Split a family into approximate merchant streams using stable amount bands."""
    if frame.empty:
        return []
    ordered = frame.sort_values("amount", kind="stable")
    clusters: list[list[int]] = []
    centers: list[float] = []
    for index, amount in zip(ordered.index, ordered["amount"]):
        amount = float(amount)
        if not clusters:
            clusters.append([index])
            centers.append(amount)
            continue
        relative_difference = abs(amount - centers[-1]) / max(abs(centers[-1]), 1.0)
        if relative_difference <= relative_tolerance:
            clusters[-1].append(index)
            centers[-1] = float(frame.loc[clusters[-1], "amount"].median())
        else:
            clusters.append([index])
            centers.append(amount)
    return [frame.loc[indices] for indices in clusters]


def _transaction_stats(
    frame: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF
) -> dict[str, float]:
    """Summarize one client's transactions for one candidate family."""
    if frame.empty:
        return {}

    ordered = frame.sort_values("timestamp", kind="stable")
    timestamps = ordered["timestamp"]
    days_before_cutoff = (cutoff - timestamps).dt.total_seconds().to_numpy() / 86_400
    gaps = timestamps.diff().dt.total_seconds().dropna().to_numpy() / 86_400
    amounts = ordered["amount"].to_numpy(dtype=np.float64)

    gap_median = float(np.median(gaps)) if len(gaps) else np.nan
    gap_mean = float(np.mean(gaps)) if len(gaps) else np.nan
    gap_std = float(np.std(gaps)) if len(gaps) else np.nan
    days_since_last = float(days_before_cutoff[-1])
    cycles_since_last = _safe_divide(days_since_last, gap_median)
    if np.isfinite(cycles_since_last):
        cycles_to_advance = max(1.0, float(np.ceil(cycles_since_last)))
        projected_days = cycles_to_advance * gap_median - days_since_last
    else:
        projected_days = np.nan

    descriptions = ordered["description"].astype(str)
    description_counts = descriptions.value_counts()
    last_description = descriptions.iloc[-1]
    amount_mean = float(np.mean(amounts))
    amount_std = float(np.std(amounts))

    stats: dict[str, float] = {
        "count": float(len(ordered)),
        "days_since_last": days_since_last,
        "days_since_first": float(days_before_cutoff[0]),
        "history_span": float(days_before_cutoff[0] - days_before_cutoff[-1]),
        "gap_count": float(len(gaps)),
        "gap_last": float(gaps[-1]) if len(gaps) else np.nan,
        "gap_previous": float(gaps[-2]) if len(gaps) >= 2 else np.nan,
        "gap_mean": gap_mean,
        "gap_median": gap_median,
        "gap_std": gap_std,
        "gap_min": float(np.min(gaps)) if len(gaps) else np.nan,
        "gap_max": float(np.max(gaps)) if len(gaps) else np.nan,
        "gap_q25": float(np.quantile(gaps, 0.25)) if len(gaps) else np.nan,
        "gap_q75": float(np.quantile(gaps, 0.75)) if len(gaps) else np.nan,
        "gap_mad": _median_absolute_deviation(gaps),
        "gap_cv": _safe_divide(gap_std, gap_mean),
        "gap_last_vs_median": _safe_divide(
            float(gaps[-1]) - gap_median, gap_median
        )
        if len(gaps)
        else np.nan,
        "recent_gap_median": float(np.median(gaps[-3:])) if len(gaps) else np.nan,
        "cycles_since_last": cycles_since_last,
        "projected_days_to_next": projected_days,
        "projected_in_horizon": float(
            np.isfinite(projected_days) and 0 <= projected_days <= 90
        ),
        "amount_last": float(amounts[-1]),
        "amount_mean": amount_mean,
        "amount_median": float(np.median(amounts)),
        "amount_std": amount_std,
        "amount_cv": _safe_divide(amount_std, amount_mean),
        "amount_last_vs_mean": _safe_divide(amounts[-1] - amount_mean, amount_mean),
        "description_unique": float(descriptions.nunique()),
        "description_mode_share": float(description_counts.iloc[0] / len(ordered)),
        "last_description_share": float(
            description_counts.get(last_description, 0) / len(ordered)
        ),
        "day_of_month_std": float(timestamps.dt.day.to_numpy().std()),
        "weekday_unique": float(timestamps.dt.dayofweek.nunique()),
        "day_of_month_mode_share": float(
            pd.Series(timestamps.dt.day).value_counts().iloc[0] / max(len(ordered), 1)
        ),
        "amount_slope_normalized": float(
            np.polyfit(np.arange(len(amounts)), amounts, 1)[0] / max(abs(amount_mean), 1e-6)
        ) if len(amounts) >= 3 else 0.0,
        "gap_autocorr": float(
            np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
        ) if len(gaps) >= 3 else 0.0,
    }
    for window in WINDOWS:
        stats[f"count_{window}d"] = float((days_before_cutoff <= window).sum())

    stream_stats = [
        _transaction_stats_without_streams(cluster, cutoff)
        for cluster in _amount_clusters(ordered)
        if len(cluster) >= 2
    ]
    stream_stats.sort(
        key=lambda values: (
            -values["count"],
            values["days_since_last"],
            values.get("gap_cv", np.inf),
        )
    )
    stats["stream_count"] = float(len(stream_stats))
    for rank, stream in enumerate(stream_stats[:3]):
        for name in STREAM_FEATURES:
            stats[f"stream_{rank}__{name}"] = stream.get(name, np.nan)

    return stats


def _transaction_stats_without_streams(
    frame: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF
) -> dict[str, float]:
    """Compute core statistics without recursively creating amount streams."""
    ordered = frame.sort_values("timestamp", kind="stable")
    timestamps = ordered["timestamp"]
    days_before_cutoff = (cutoff - timestamps).dt.total_seconds().to_numpy() / 86_400
    gaps = timestamps.diff().dt.total_seconds().dropna().to_numpy() / 86_400
    amounts = ordered["amount"].to_numpy(dtype=np.float64)
    gap_median = float(np.median(gaps)) if len(gaps) else np.nan
    gap_mean = float(np.mean(gaps)) if len(gaps) else np.nan
    gap_std = float(np.std(gaps)) if len(gaps) else np.nan
    days_since_last = float(days_before_cutoff[-1])
    cycles_since_last = _safe_divide(days_since_last, gap_median)
    if np.isfinite(cycles_since_last):
        projected_days = max(1.0, float(np.ceil(cycles_since_last))) * gap_median - days_since_last
    else:
        projected_days = np.nan
    descriptions = ordered["description"].astype(str)
    amount_mean = float(np.mean(amounts))
    amount_std = float(np.std(amounts))
    return {
        "count": float(len(ordered)),
        "days_since_last": days_since_last,
        "history_span": float(days_before_cutoff[0] - days_before_cutoff[-1]),
        "gap_last": float(gaps[-1]) if len(gaps) else np.nan,
        "gap_median": gap_median,
        "gap_std": gap_std,
        "gap_cv": _safe_divide(gap_std, gap_mean),
        "cycles_since_last": cycles_since_last,
        "projected_days_to_next": projected_days,
        "projected_in_horizon": float(
            np.isfinite(projected_days) and 0 <= projected_days <= 90
        ),
        "amount_mean": amount_mean,
        "amount_cv": _safe_divide(amount_std, amount_mean),
        "description_unique": float(descriptions.nunique()),
        "description_mode_share": float(descriptions.value_counts().iloc[0] / len(ordered)),
    }


def _global_stats(frame: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> dict[str, float]:
    timestamps = frame["timestamp"]
    days = (cutoff - timestamps).dt.total_seconds().to_numpy() / 86_400
    amounts = frame["amount"].to_numpy(dtype=np.float64)
    candidate = frame["candidate_family"].isin(FAMILIES)
    result = {
        "global__transaction_count": float(len(frame)),
        "global__days_since_last": float(days.min()),
        "global__history_span": float(days.max() - days.min()),
        "global__amount_mean": float(amounts.mean()),
        "global__amount_std": float(amounts.std()),
        "global__candidate_count": float(candidate.sum()),
        "global__candidate_share": float(candidate.mean()),
        "global__candidate_family_count": float(
            frame.loc[candidate, "candidate_family"].nunique()
        ),
    }
    for window in WINDOWS:
        result[f"global__count_{window}d"] = float((days <= window).sum())
    return result


def build_feature_tables(
    frame: pd.DataFrame,
    expected_clients: Iterable[str] | None = None,
    cutoff: pd.Timestamp = CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return client-wide and client/family feature tables.

    The wide table is useful for direct multiclass models. The pair table lets one
    binary model learn recurrence patterns shared by all seven merchant families.
    """
    required = {"client_id", "timestamp", "amount", "description", "candidate_family"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    import joblib
    from pathlib import Path
    artifact_dir = Path(__file__).parent / "artifacts"
    svd_pipeline = None
    svd_dict = {}
    if (artifact_dir / "text_svd_pipeline.joblib").exists():
        svd_pipeline = joblib.load(artifact_dir / "text_svd_pipeline.joblib")

    data = frame.copy()
    data["client_id"] = data["client_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    
    if svd_pipeline is not None:
        data["clean_description"] = data.get("clean_description", data["description"]).fillna("").astype(str)
        print("Extracting SVD features in batch...", flush=True)
        family_texts_df = data.groupby(["client_id", "candidate_family"])["clean_description"].apply(lambda texts: " ".join(texts)).reset_index()
        svd_features = svd_pipeline.transform(family_texts_df["clean_description"].values)
        for i, row in family_texts_df.iterrows():
            svd_dict[(row["client_id"], row["candidate_family"])] = svd_features[i]
    if (data["timestamp"] >= cutoff).any():
        raise ValueError("Feature construction received a transaction at or after the cutoff")

    available_clients = set(data["client_id"])
    client_ids = (
        sorted(available_clients)
        if expected_clients is None
        else [str(value) for value in expected_clients]
    )
    missing_clients = set(client_ids) - available_clients
    if missing_clients:
        raise ValueError(
            f"No transactions for {len(missing_clients)} expected clients, "
            f"e.g. {sorted(missing_clients)[:5]}"
        )

    by_client = {client_id: group for client_id, group in data.groupby("client_id", sort=False)}
    wide_rows: list[dict[str, float | str]] = []
    pair_rows: list[dict[str, float | str]] = []

    for client_id in client_ids:
        client = by_client[client_id]
        global_features = _global_stats(client, cutoff)
        family_stats: dict[str, dict[str, float]] = {}
        wide: dict[str, float | str] = {"client_id": client_id, **global_features}

        for family in FAMILIES:
            stats = _transaction_stats(
                client[client["candidate_family"] == family], cutoff
            )
            if (client_id, family) in svd_dict:
                for idx, val in enumerate(svd_dict[(client_id, family)]):
                    stats[f"text_svd_{idx}"] = float(val)
            
            family_stats[family] = stats
            wide.update({f"{family}__{name}": value for name, value in stats.items()})

        counts = {family: family_stats[family].get("count", 0.0) for family in FAMILIES}
        recencies = {
            family: family_stats[family].get("days_since_last", np.inf)
            for family in FAMILIES
        }
        proj_days = {
            family: family_stats[family].get("projected_days_to_next", np.nan)
            for family in FAMILIES
        }
        valid_projs = {
            f: p for f, p in proj_days.items() 
            if np.isfinite(p) and 0 <= p <= 90
        }
        min_proj = min(valid_projs.values()) if valid_projs else np.nan
        cycles = {
            family: family_stats[family].get("cycles_since_last", np.nan)
            for family in FAMILIES
        }
        lapsed = {
            family: float(np.isfinite(cycles[family]) and cycles[family] > 1.8)
            for family in FAMILIES
        }
        active_candidates = [
            f for f in FAMILIES
            if counts[f] >= 2 and np.isfinite(cycles[f]) and cycles[f] <= 1.8 and f in valid_projs
        ]
        n_active = float(len(active_candidates))
        global_features["active_candidate_count"] = n_active
        wide["active_candidate_count"] = n_active

        for family in FAMILIES:
            stats = family_stats[family]
            p_val = proj_days[family]
            amed = stats.get("amount_median", 0.0)
            
            if np.isfinite(p_val) and 0 <= p_val <= 90:
                rank = float(1 + sum(v < p_val for v in valid_projs.values()))
                gap = float(p_val - min_proj)
            else:
                rank = 8.0
                gap = 999.0
                
            is_act = float(family in active_candidates)
            family_super_signals = {
                "proj_rank": rank,
                "is_proj_rank_1": float(rank == 1.0),
                "is_proj_rank_2": float(rank == 2.0),
                "proj_gap_to_min": gap,
                "is_lapsed": lapsed[family],
                "is_active_candidate": is_act,
                "is_only_active_candidate": float(n_active == 1.0 and is_act),
                "amount_below_16": float(amed > 0 and amed < 16.0),
                "amount_16_to_28": float(16.0 <= amed < 28.0),
                "amount_above_28": float(amed >= 28.0),
            }
            wide.update({f"{family}__{k}": v for k, v in family_super_signals.items()})

            pair: dict[str, float | str] = {
                "client_id": client_id,
                "family": family,
                **global_features,
                **stats,
                **family_super_signals,
                "count_rank": float(
                    1 + sum(value > counts[family] for value in counts.values())
                ),
                "recency_rank": float(
                    1 + sum(value < recencies[family] for value in recencies.values())
                ),
            }
            for candidate in FAMILIES:
                pair[f"family_is_{candidate}"] = float(family == candidate)
            pair_rows.append(pair)

        wide_rows.append(wide)

    wide_frame = pd.DataFrame(wide_rows).set_index("client_id").sort_index(axis=1)
    pair_frame = pd.DataFrame(pair_rows).set_index(["client_id", "family"]).sort_index(axis=1)
    return wide_frame, pair_frame


def build_stream_table(
    frame: pd.DataFrame,
    expected_clients: Iterable[str] | None = None,
    cutoff: pd.Timestamp = CUTOFF,
) -> pd.DataFrame:
    """Create one row per approximate recurring merchant stream.

    Streams are amount-stable clusters inside a client/family pair. This avoids
    corrupting interval statistics when a client has several merchants assigned to
    the same broad family.
    """
    data = frame.copy()
    data["client_id"] = data["client_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    available_clients = set(data["client_id"])
    client_ids = (
        sorted(available_clients)
        if expected_clients is None
        else [str(value) for value in expected_clients]
    )
    missing_clients = set(client_ids) - available_clients
    if missing_clients:
        raise ValueError(f"No transactions for expected clients: {sorted(missing_clients)[:5]}")

    by_client = {client_id: group for client_id, group in data.groupby("client_id", sort=False)}
    rows: list[dict[str, float | str]] = []
    for client_id in client_ids:
        client = by_client[client_id]
        global_features = _global_stats(client, cutoff)
        client_streams: list[dict[str, float | str]] = []
        candidates = client[client["candidate_family"].isin(FAMILIES)]
        for family in FAMILIES:
            family_rows = candidates[candidates["candidate_family"] == family]
            for stream_number, stream in enumerate(_amount_clusters(family_rows)):
                stats = _transaction_stats_without_streams(stream, cutoff)
                timestamps = stream.sort_values("timestamp")["timestamp"]
                gap_median = stats.get("gap_median", np.nan)
                cadence_distance = (
                    min(abs(gap_median - cadence) for cadence in (7, 14, 30, 60, 90, 365))
                    if np.isfinite(gap_median)
                    else np.nan
                )
                row: dict[str, float | str] = {
                    "client_id": client_id,
                    "family": family,
                    "stream_number": float(stream_number),
                    **global_features,
                    **stats,
                    "cadence_distance": cadence_distance,
                    "last_day_of_month": float(timestamps.iloc[-1].day),
                    "last_month": float(timestamps.iloc[-1].month),
                }
                for candidate in FAMILIES:
                    row[f"family_is_{candidate}"] = float(family == candidate)
                client_streams.append(row)

        if not client_streams:
            # The supplied data currently always has candidates, but retain a
            # deterministic placeholder for robustness and complete client coverage.
            client_streams.append(
                {"client_id": client_id, "family": FAMILIES[0], **global_features}
            )
        counts = [float(row.get("count", 0.0)) for row in client_streams]
        recencies = [float(row.get("days_since_last", np.inf)) for row in client_streams]
        for index, row in enumerate(client_streams):
            row["stream_count_rank"] = float(1 + sum(value > counts[index] for value in counts))
            row["stream_recency_rank"] = float(
                1 + sum(value < recencies[index] for value in recencies)
            )
            row["client_stream_count"] = float(len(client_streams))
            rows.append(row)

    return pd.DataFrame(rows).set_index(["client_id", "family"]).sort_index(axis=1)
