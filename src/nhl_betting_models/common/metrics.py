"""Forecast and betting performance metrics."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import xlogy

from .market_matching import require_comparable
from .odds import american_to_implied_probability
from .schemas import (
    BetResult,
    CalibrationBin,
    Offer,
    PerformanceRecord,
)

if TYPE_CHECKING:
    import polars as pl


def _arrays(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if (
        p.ndim != 1
        or y.ndim != 1
        or len(p) == 0
        or len(p) != len(y)
    ):
        raise ValueError(
            "Probabilities and outcomes must be "
            "non-empty equal-length vectors",
        )

    if (
        np.any(~np.isfinite(p))
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(
            "Probabilities must be finite and within [0, 1]",
        )

    if np.any((y != 0) & (y != 1)):
        raise ValueError(
            "Outcomes must be binary 0/1 values",
        )

    return p, y


def brier_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """Calculate mean binary Brier score."""

    p, y = _arrays(probabilities, outcomes)
    return float(np.mean((p - y) ** 2))


def binary_log_loss(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """Calculate numerically stable binary log loss."""

    p, y = _arrays(probabilities, outcomes)

    epsilon = np.finfo(float).eps
    p = np.clip(
        p,
        epsilon,
        1.0 - epsilon,
    )

    return float(
        -np.mean(
            xlogy(y, p)
            + xlogy(1.0 - y, 1.0 - p),
        )
    )


def calibration_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Build equal-width reliability bins over [0, 1]."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    p, y = _arrays(probabilities, outcomes)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bucket = np.minimum(
        np.searchsorted(
            edges,
            p,
            side="right",
        )
        - 1,
        n_bins - 1,
    )

    result: list[CalibrationBin] = []

    for index in range(n_bins):
        mask = bucket == index
        count = int(mask.sum())

        result.append(
            CalibrationBin(
                bin_lower=float(edges[index]),
                bin_upper=float(edges[index + 1]),
                count=count,
                mean_prediction=(
                    float(p[mask].mean())
                    if count
                    else None
                ),
                observed_rate=(
                    float(y[mask].mean())
                    if count
                    else None
                ),
            )
        )

    return result


def roi(
    total_profit: float,
    total_stake: float,
) -> float:
    """Return net profit divided by total amount staked."""

    if total_stake <= 0:
        raise ValueError(
            "Total stake must be positive",
        )

    return total_profit / total_stake


def yield_rate(
    total_profit: float,
    total_stake: float,
) -> float:
    """Project definition of yield: net profit / total stake."""

    return roi(total_profit, total_stake)


def win_rate(
    results: Sequence[BetResult],
) -> float:
    """Return wins divided by decisions, excluding pushes/voids/pending."""

    settled = [
        result
        for result in results
        if result
        in {
            BetResult.WIN,
            BetResult.LOSS,
        }
    ]

    if not settled:
        raise ValueError(
            "At least one win/loss result is required",
        )

    wins = sum(
        result == BetResult.WIN
        for result in settled
    )

    return wins / len(settled)


def closing_line_value(
    taken: Offer,
    closing: Offer,
) -> float:
    """Calculate probability-space CLV for exact comparable offers.

    CLV is closing raw implied probability minus the raw implied probability
    of the price taken. Positive values therefore mean the bettor took a
    better price for the identical side/line than the later closing snapshot.
    """

    require_comparable(taken, closing)

    if (
        closing.retrieval_timestamp
        <= taken.retrieval_timestamp
    ):
        raise ValueError(
            "Closing snapshot must be later than the taken snapshot",
        )

    closing_probability = (
        american_to_implied_probability(
            closing.american_odds,
        )
    )
    taken_probability = (
        american_to_implied_probability(
            taken.american_odds,
        )
    )

    return closing_probability - taken_probability


def grouped_performance_summary(
    records: Sequence[PerformanceRecord],
    *,
    odds_bucket_width: int = 50,
    line_bucket_width: float = 1.0,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Aggregate performance by required reporting dimensions.

    Date range is represented by optional inclusive start and exclusive end
    filters plus the minimum and maximum settlement timestamps in each group.
    """

    import polars as pl

    if (
        odds_bucket_width <= 0
        or line_bucket_width <= 0
    ):
        raise ValueError(
            "Bucket widths must be positive",
        )

    rows = [
        record.model_dump(mode="python")
        for record in records
    ]

    if not rows:
        return pl.DataFrame()

    frame = pl.DataFrame(rows)

    if start is not None:
        frame = frame.filter(
            pl.col("settled_at") >= start,
        )

    if end is not None:
        frame = frame.filter(
            pl.col("settled_at") < end,
        )

    frame = frame.with_columns(
        (
            (
                pl.col("american_odds")
                / odds_bucket_width
            ).floor()
            * odds_bucket_width
        ).alias("odds_bucket"),
        (
            (
                pl.col("line")
                / line_bucket_width
            ).floor()
            * line_bucket_width
        ).alias("line_bucket"),
        pl.when(
            pl.col("result")
            == BetResult.WIN.value
        )
        .then(1)
        .otherwise(0)
        .alias("win"),
        pl.when(
            pl.col("result").is_in(
                [
                    BetResult.WIN.value,
                    BetResult.LOSS.value,
                ]
            )
        )
        .then(1)
        .otherwise(0)
        .alias("decision"),
    )

    group_columns = [
        "model_version",
        "sportsbook",
        "market_key",
        "market_side",
        "odds_bucket",
        "line_bucket",
    ]

    return (
        frame.group_by(group_columns)
        .agg(
            pl.len().alias("n_bets"),
            pl.col("stake")
            .sum()
            .alias("total_stake"),
            pl.col("profit")
            .sum()
            .alias("net_profit"),
            (
                pl.col("profit").sum()
                / pl.col("stake").sum()
            ).alias("roi"),
            (
                pl.col("profit").sum()
                / pl.col("stake").sum()
            ).alias("yield"),
            (
                pl.col("win").sum()
                / pl.col("decision").sum()
            ).alias("win_rate"),
            pl.col("settled_at")
            .min()
            .alias("date_start"),
            pl.col("settled_at")
            .max()
            .alias("date_end"),
        )
        .sort(group_columns)
    )