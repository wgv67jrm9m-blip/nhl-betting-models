"""Synthetic tests for NHL SOG sportsbook evaluation."""

from datetime import UTC, datetime
from typing import Any

import pytest

from nhl_betting_models.common.config import EligibilityThresholds
from nhl_betting_models.common.edge import expected_value
from nhl_betting_models.common.odds import american_to_decimal
from nhl_betting_models.common.schemas import MarketSide, Offer
from nhl_betting_models.nhl.schemas import (
    ConfidenceTier,
    SOGDistributionConfig,
    SOGDistributionFamily,
    SOGProjectionResult,
    SOGQualityFlag,
)
from nhl_betting_models.nhl.sog_distribution import build_distribution
from nhl_betting_models.nhl.sog_evaluation import (
    evaluate_sog_offer,
    evaluate_sog_offers,
)

MODEL_RUN = datetime(2026, 1, 1, 18, tzinfo=UTC)
ODDS_TIME = datetime(2026, 1, 1, 17, tzinfo=UTC)


def synthetic_offer(
    *,
    side: MarketSide,
    american_odds: int,
    **overrides: Any,
) -> Offer:
    """SYNTHETIC TEST FIXTURE: no live sportsbook data."""

    values: dict[str, Any] = {
        "record_id": f"synthetic-{side.value}",
        "source_name": "synthetic-test-source",
        "sportsbook": "SyntheticBook",
        "canonical_event_id": "synthetic-game-1",
        "canonical_subject_id": "synthetic-player-1",
        "market_key": "player_sog",
        "side": side,
        "line": 3.0,
        "settlement_rule_id": "synthetic-sog-rule-v1",
        "american_odds": american_odds,
        "retrieval_timestamp": ODDS_TIME,
        "as_of_timestamp": ODDS_TIME,
    }

    values.update(overrides)

    if "decimal_odds" not in values:
        values["decimal_odds"] = american_to_decimal(
            values["american_odds"]
        )

    return Offer(**values)


def synthetic_projection(
    *,
    confidence_tier: ConfidenceTier = ConfidenceTier.HIGH,
    quality_flags: tuple[SOGQualityFlag, ...] = (),
) -> SOGProjectionResult:
    """SYNTHETIC TEST FIXTURE: no real NHL player projection."""

    return SOGProjectionResult(
        canonical_event_id="synthetic-game-1",
        canonical_player_id="synthetic-player-1",
        model_run_timestamp=MODEL_RUN,
        base_expected_sog=3.0,
        adjusted_expected_sog=3.0,
        components=(),
        confidence_tier=confidence_tier,
        quality_flags=quality_flags,
    )


def synthetic_distribution():
    """SYNTHETIC TEST FIXTURE: no real NHL distribution forecast."""

    return build_distribution(
        3.0,
        SOGDistributionConfig(
            family=SOGDistributionFamily.POISSON,
        ),
    )


def permissive_thresholds() -> EligibilityThresholds:
    """SYNTHETIC TEST FIXTURE: not production eligibility thresholds."""

    return EligibilityThresholds(
        min_ev=-1.0,
        min_edge=-1.0,
        min_model_probability=0.0,
    )


def synthetic_complete_market() -> tuple[Offer, Offer]:
    """SYNTHETIC TEST FIXTURE: complete two-way SOG market."""

    return (
        synthetic_offer(
            side=MarketSide.OVER,
            american_odds=100,
        ),
        synthetic_offer(
            side=MarketSide.UNDER,
            american_odds=-120,
        ),
    )


def test_whole_number_evaluation_uses_push_aware_ev() -> None:
    over, under = synthetic_complete_market()

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
    )

    assert result is not None
    assert not flags

    probabilities = result.probabilities
    expected = expected_value(
        win_probability=probabilities.win_probability,
        loss_probability=probabilities.loss_probability,
        push_probability=probabilities.push_probability,
        decimal_odds=over.decimal_odds,
    )

    assert probabilities.push_probability > 0.0
    assert result.market_evaluation.expected_value == pytest.approx(
        expected
    )


def test_mismatched_game_offer_is_rejected() -> None:
    target = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        canonical_event_id="synthetic-game-2",
    )
    under = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    with pytest.raises(ValueError, match="event"):
        evaluate_sog_offer(
            offer=target,
            market_offers=(target, under),
            projection=synthetic_projection(),
            distribution=synthetic_distribution(),
            thresholds=permissive_thresholds(),
        )


def test_mismatched_player_offer_is_rejected() -> None:
    target = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        canonical_subject_id="synthetic-player-2",
    )
    under = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    with pytest.raises(ValueError, match="subject"):
        evaluate_sog_offer(
            offer=target,
            market_offers=(target, under),
            projection=synthetic_projection(),
            distribution=synthetic_distribution(),
            thresholds=permissive_thresholds(),
        )


def test_unsupported_market_is_rejected() -> None:
    target = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        market_key="synthetic_unsupported_market",
    )
    under = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    with pytest.raises(
        ValueError,
        match="supported player SOG market",
    ):
        evaluate_sog_offer(
            offer=target,
            market_offers=(target, under),
            projection=synthetic_projection(),
            distribution=synthetic_distribution(),
            thresholds=permissive_thresholds(),
        )


def test_wrong_side_is_rejected() -> None:
    target = synthetic_offer(
        side=MarketSide.HOME,
        american_odds=100,
    )

    with pytest.raises(
        ValueError,
        match="over or under",
    ):
        evaluate_sog_offer(
            offer=target,
            market_offers=(target,),
            projection=synthetic_projection(),
            distribution=synthetic_distribution(),
            thresholds=permissive_thresholds(),
        )


def test_mismatched_line_is_rejected_as_incomplete_market() -> None:
    over = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        line=3.0,
    )
    under = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
        line=4.0,
    )

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is None
    assert "incomplete_market" in flags


def test_same_side_offer_is_not_an_opposite_market_side() -> None:
    over_one = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        record_id="synthetic-over-1",
    )
    over_two = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=-120,
        record_id="synthetic-over-2",
    )

    result, flags = evaluate_sog_offer(
        offer=over_one,
        market_offers=(over_one, over_two),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is None
    assert "incomplete_market" in flags


def test_incomplete_market_cannot_be_automatically_eligible() -> None:
    over = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
    )

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over,),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is None
    assert "incomplete_market" in flags


def test_stale_market_cannot_be_automatically_eligible() -> None:
    over = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        is_stale=True,
    )
    under = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is None
    assert "stale_odds" in flags


def test_low_confidence_projection_cannot_be_automatically_eligible() -> None:
    over, under = synthetic_complete_market()
    projection = synthetic_projection(
        confidence_tier=ConfidenceTier.LOW,
        quality_flags=(
            SOGQualityFlag.UNCERTAIN_ROLE,
        ),
    )

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=projection,
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is not None
    assert "unconfirmed_role" in flags
    assert "low_confidence_projection" in flags
    assert not result.market_evaluation.eligible
    assert (
        "sog_quality_gate_failed"
        in result.market_evaluation.threshold_failures
    )


def test_no_projection_cannot_be_automatically_eligible() -> None:
    over, under = synthetic_complete_market()
    projection = synthetic_projection(
        confidence_tier=ConfidenceTier.NO_PROJECTION,
        quality_flags=(
            SOGQualityFlag.MISSING_PROJECTED_MINUTES,
        ),
    )

    result, flags = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=projection,
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert result is None
    assert "no_projection" in flags


def test_automatic_eligibility_is_disabled_by_default() -> None:
    over, under = synthetic_complete_market()

    result, _ = evaluate_sog_offer(
        offer=over,
        market_offers=(over, under),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
    )

    assert result is not None
    assert not result.market_evaluation.eligible
    assert (
        "automatic_eligibility_disabled"
        in result.market_evaluation.threshold_failures
    )


def test_alternate_lines_are_evaluated_separately() -> None:
    over_three = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=100,
        line=3.0,
        record_id="synthetic-over-3",
    )
    under_three = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-120,
        line=3.0,
        record_id="synthetic-under-3",
    )
    over_four = synthetic_offer(
        side=MarketSide.OVER,
        american_odds=110,
        line=4.0,
        market_key="player_sog_alt",
        record_id="synthetic-over-4",
    )
    under_four = synthetic_offer(
        side=MarketSide.UNDER,
        american_odds=-130,
        line=4.0,
        market_key="player_sog_alt",
        record_id="synthetic-under-4",
    )

    evaluations, blocked = evaluate_sog_offers(
        offers=(
            over_three,
            under_three,
            over_four,
            under_four,
        ),
        projection=synthetic_projection(),
        distribution=synthetic_distribution(),
        thresholds=permissive_thresholds(),
        automatic_eligibility=True,
    )

    assert not blocked
    assert len(evaluations) == 4
    assert {
        (evaluation.market_key, evaluation.line)
        for evaluation in evaluations
    } == {
        ("player_sog", 3.0),
        ("player_sog_alt", 4.0),
    }