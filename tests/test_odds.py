"""Synthetic tests for shared odds conversion."""

import pytest

from nhl_betting_models.common.odds import (
    american_to_decimal,
    american_to_implied_probability,
    decimal_to_american,
    probability_to_fair_american,
    probability_to_fair_decimal,
)


def test_positive_and_negative_american_odds_conversion() -> None:
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)

    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200


@pytest.mark.parametrize(
    "odds",
    [0, -100, 99, -50],
)
def test_invalid_american_odds(
    odds: int,
) -> None:
    with pytest.raises(ValueError):
        american_to_decimal(odds)


def test_implied_probability_and_fair_odds_round_trip() -> None:
    probability = american_to_implied_probability(-150)

    assert probability == pytest.approx(0.6)
    assert probability_to_fair_decimal(
        probability,
    ) == pytest.approx(5 / 3)
    assert probability_to_fair_american(
        probability,
    ) == -150


@pytest.mark.parametrize(
    "probability",
    [0.0, 1.0, -0.1, 1.1],
)
def test_fair_odds_reject_closed_or_outside_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        probability_to_fair_decimal(
            probability,
        )