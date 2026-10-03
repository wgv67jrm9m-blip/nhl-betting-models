"""Synthetic tests for model and betting metrics."""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from nhl_betting_models.common.metrics import (
    binary_log_loss,
    brier_score,
    calibration_bins,
    closing_line_value,
    roi,
    win_rate,
)
from nhl_betting_models.common.odds import (
    american_to_decimal,
)
from nhl_betting_models.common.schemas import (
    BetResult,
    MarketSide,
    Offer,
)


def synthetic_offer(
    **overrides: Any,
) -> Offer:
    """SYNTHETIC TEST FIXTURE: no live sportsbook data."""

    values: dict[str, Any] = {
        "record_id": "synthetic-offer-1",
        "source_name": "synthetic-test-source",
        "sportsbook": "SyntheticBook",
        "canonical_event_id": "synthetic-game-1",
        "canonical_subject_id": "synthetic-player-1",
        "market_key": "player_sog",
        "side": MarketSide.OVER,
        "line": 3.5,
        "settlement_rule_id": "synthetic-sog-rule-v1",
        "american_odds": -110,
        "retrieval_timestamp": datetime(
            2026,
            1,
            1,
            17,
            tzinfo=UTC,
        ),
        "as_of_timestamp": datetime(
            2026,
            1,
            1,
            17,
            tzinfo=UTC,
        ),
    }

    values.update(overrides)
    values.setdefault(
        "decimal_odds",
        american_to_decimal(
            values["american_odds"],
        ),
    )

    return Offer(**values)


def test_forecast_metrics() -> None:
    assert brier_score(
        [0.8, 0.2],
        [1, 0],
    ) == pytest.approx(0.04)

    assert binary_log_loss(
        [0.8, 0.2],
        [1, 0],
    ) == pytest.approx(
        -math.log(0.8),
    )

    bins = calibration_bins(
        [0.2, 0.8],
        [0, 1],
        n_bins=2,
    )

    assert sum(
        item.count
        for item in bins
    ) == 2


def test_betting_metrics() -> None:
    assert roi(
        2.0,
        10.0,
    ) == pytest.approx(0.2)

    assert win_rate(
        [
            BetResult.WIN,
            BetResult.LOSS,
            BetResult.PUSH,
        ]
    ) == pytest.approx(0.5)


def test_clv_requires_comparable_closing_line() -> None:
    taken = synthetic_offer(
        record_id="taken",
        american_odds=-110,
    )

    closing = synthetic_offer(
        record_id="closing",
        american_odds=-130,
        retrieval_timestamp=(
            taken.retrieval_timestamp
            + timedelta(hours=1)
        ),
    )

    assert closing_line_value(
        taken,
        closing,
    ) > 0

    bad = closing.model_copy(
        update={"line": 4.5},
    )

    with pytest.raises(
        ValueError,
        match="not comparable",
    ):
        closing_line_value(
            taken,
            bad,
        )