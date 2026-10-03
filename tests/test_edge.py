"""Synthetic tests for EV, fair odds, and eligibility."""

import pytest

from nhl_betting_models.common.config import (
    EligibilityThresholds,
)
from nhl_betting_models.common.edge import (
    evaluate_market,
    expected_value,
    expected_value_no_push,
)
from nhl_betting_models.common.odds import (
    probability_to_fair_american,
    probability_to_fair_decimal,
)


def test_expected_value_no_push() -> None:
    assert expected_value_no_push(
        0.55,
        2.0,
    ) == pytest.approx(0.10)


def test_expected_value_with_push() -> None:
    assert expected_value(
        0.50,
        0.40,
        0.10,
        2.0,
    ) == pytest.approx(0.10)


def test_fair_odds_calculations() -> None:
    assert probability_to_fair_decimal(
        0.4,
    ) == pytest.approx(2.5)

    assert probability_to_fair_american(
        0.4,
    ) == 150


def test_positive_ev_does_not_bypass_thresholds() -> None:
    evaluation = evaluate_market(
        model_probability=0.55,
        no_vig_market_probability=0.54,
        offered_american_odds=100,
        offered_decimal_odds=2.0,
        thresholds=EligibilityThresholds(
            min_ev=0.05,
            min_edge=0.02,
        ),
    )

    assert evaluation.expected_value > 0
    assert not evaluation.eligible
    assert (
        "edge_below_threshold"
        in evaluation.threshold_failures
    )