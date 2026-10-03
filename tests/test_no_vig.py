"""Synthetic tests for no-vig calculations."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from nhl_betting_models.common.no_vig import (
    multiway_proportional,
    two_way_proportional,
)
from nhl_betting_models.common.odds import (
    american_to_decimal,
)
from nhl_betting_models.common.schemas import (
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


def test_two_way_no_vig() -> None:
    over = synthetic_offer(
        record_id="o",
        american_odds=-110,
    )
    under = synthetic_offer(
        record_id="u",
        side=MarketSide.UNDER,
        american_odds=-110,
    )

    result = two_way_proportional(
        [over, under],
    )

    assert result.no_vig_probabilities == pytest.approx(
        {
            "over": 0.5,
            "under": 0.5,
        }
    )
    assert result.overround == pytest.approx(
        2 * (110 / 210) - 1,
    )
    assert (
        result.raw_implied_probabilities["over"]
        == pytest.approx(110 / 210)
    )


def test_multiway_no_vig() -> None:
    common = {
        "market_key": "regulation_moneyline",
        "canonical_subject_id": "synthetic-game-market",
        "line": 0.0,
        "american_odds": 200,
    }

    home = synthetic_offer(
        record_id="h",
        side=MarketSide.HOME,
        **common,
    )
    away = synthetic_offer(
        record_id="a",
        side=MarketSide.AWAY,
        **common,
    )
    draw = synthetic_offer(
        record_id="d",
        side=MarketSide.DRAW,
        **common,
    )

    result = multiway_proportional(
        [home, away, draw],
        {"home", "away", "draw"},
    )

    assert result.no_vig_probabilities == pytest.approx(
        {
            "home": 1 / 3,
            "away": 1 / 3,
            "draw": 1 / 3,
        }
    )


def test_no_vig_rejects_incomplete_market() -> None:
    with pytest.raises(
        ValueError,
        match="Incomplete",
    ):
        two_way_proportional(
            [synthetic_offer()],
        )


def test_no_vig_rejects_stale_or_mismatched_market() -> None:
    over = synthetic_offer(
        record_id="o",
    )
    stale_under = synthetic_offer(
        record_id="u",
        side=MarketSide.UNDER,
        is_stale=True,
    )

    with pytest.raises(
        ValueError,
        match="stale",
    ):
        two_way_proportional(
            [over, stale_under],
        )

    mismatched_under = synthetic_offer(
        record_id="u2",
        side=MarketSide.UNDER,
        retrieval_timestamp=(
            over.retrieval_timestamp
            + timedelta(minutes=1)
        ),
    )

    with pytest.raises(
        ValueError,
        match="mismatched",
    ):
        two_way_proportional(
            [over, mismatched_under],
        )