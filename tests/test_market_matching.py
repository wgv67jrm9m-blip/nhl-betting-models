"""Synthetic tests for exact market matching."""

from datetime import UTC, datetime
from typing import Any

import pytest

from nhl_betting_models.common.market_matching import (
    compare_offers,
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


def test_market_matching_success() -> None:
    left = synthetic_offer(
        record_id="a",
        american_odds=-120,
    )
    right = synthetic_offer(
        record_id="b",
        american_odds=105,
    )

    assert compare_offers(
        left,
        right,
    ).comparable


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"line": 4.5},
            "line_mismatch",
        ),
        (
            {"side": MarketSide.UNDER},
            "side_mismatch",
        ),
        (
            {
                "canonical_subject_id":
                    "synthetic-player-2",
            },
            "subject_mismatch",
        ),
        (
            {
                "canonical_event_id":
                    "synthetic-game-2",
            },
            "event_mismatch",
        ),
    ],
)
def test_market_matching_failures(
    overrides: dict[str, Any],
    reason: str,
) -> None:
    left = synthetic_offer(
        record_id="a",
    )
    right = synthetic_offer(
        record_id="b",
        **overrides,
    )

    result = compare_offers(
        left,
        right,
    )

    assert not result.comparable
    assert reason in result.reasons