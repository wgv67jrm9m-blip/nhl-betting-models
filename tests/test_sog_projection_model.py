"""Tests for expected NHL SOG projection calculation."""

from __future__ import annotations

from datetime import datetime

import pytest

from nhl_betting_models.nhl.schemas import (
    ConfidenceTier,
    PlayerPosition,
    SOGContextualAdjustment,
    SOGGameState,
    SOGProjectedMinutes,
    SOGProjectionInput,
    SOGQualityFlag,
    SOGShrunkRate,
)
from nhl_betting_models.nhl.sog_projection_model import project_sog

TIMESTAMP = datetime.fromisoformat("2026-01-12T12:00:00+00:00")


def _rate(
    game_state: SOGGameState,
    posterior_sog_per_60: float,
    *,
    quality_flags: tuple[SOGQualityFlag, ...] = (),
) -> SOGShrunkRate:
    """Build one small valid shrunk-rate fixture."""

    return SOGShrunkRate(
        canonical_player_id="canonical-player-a",
        position=PlayerPosition.FORWARD,
        game_state=game_state,
        observed_sog_per_60=posterior_sog_per_60,
        observed_exposure_minutes=100.0,
        prior_sog_per_60=posterior_sog_per_60,
        prior_equivalent_minutes=60.0,
        posterior_sog_per_60=posterior_sog_per_60,
        as_of_timestamp=TIMESTAMP,
        quality_flags=quality_flags,
    )


def _projection_input(
    *,
    five_on_five_minutes: float = 15.0,
    power_play_minutes: float = 3.0,
    short_handed_minutes: float = 1.0,
    rates: tuple[SOGShrunkRate, ...] | None = None,
    contextual_adjustments: tuple[SOGContextualAdjustment, ...] = (),
    input_quality_flags: tuple[SOGQualityFlag, ...] = (),
) -> SOGProjectionInput:
    """Build one valid canonical projection-input fixture."""

    return SOGProjectionInput(
        canonical_event_id="canonical-game-a",
        canonical_player_id="canonical-player-a",
        position=PlayerPosition.FORWARD,
        model_run_timestamp=TIMESTAMP,
        projected_minutes=SOGProjectedMinutes(
            canonical_player_id="canonical-player-a",
            five_on_five_minutes=five_on_five_minutes,
            power_play_minutes=power_play_minutes,
            short_handed_minutes=short_handed_minutes,
            as_of_timestamp=TIMESTAMP,
        ),
        shrunk_rates=(
            rates
            if rates is not None
            else (
                _rate(SOGGameState.FIVE_ON_FIVE, 8.0),
                _rate(SOGGameState.POWER_PLAY, 12.0),
                _rate(SOGGameState.SHORT_HANDED, 3.0),
            )
        ),
        contextual_adjustments=contextual_adjustments,
        input_quality_flags=input_quality_flags,
    )


def test_project_sog_sums_game_state_components() -> None:
    """Calculate expected SOG from three projected game-state components."""

    result = project_sog(_projection_input())

    assert result.base_expected_sog == pytest.approx(2.65)
    assert result.adjusted_expected_sog == pytest.approx(2.65)
    assert result.confidence_tier is ConfidenceTier.HIGH
    assert result.quality_flags == ()
    assert {
        component.game_state for component in result.components
    } == {
        SOGGameState.FIVE_ON_FIVE,
        SOGGameState.POWER_PLAY,
        SOGGameState.SHORT_HANDED,
    }


def test_project_sog_ignores_zero_minute_state_without_rate() -> None:
    """Allow a missing rate when no minutes are projected for that state."""

    result = project_sog(
        _projection_input(
            short_handed_minutes=0.0,
            rates=(
                _rate(SOGGameState.FIVE_ON_FIVE, 8.0),
                _rate(SOGGameState.POWER_PLAY, 12.0),
            ),
        )
    )

    assert result.base_expected_sog == pytest.approx(2.6)
    assert result.adjusted_expected_sog == pytest.approx(2.6)
    assert result.confidence_tier is ConfidenceTier.HIGH
    assert SOGQualityFlag.MISSING_PLAYER_RATE not in result.quality_flags
    assert len(result.components) == 2


def test_project_sog_returns_no_projection_for_missing_required_rate() -> None:
    """Do not emit an expectation when nonzero minutes lack a rate."""

    result = project_sog(
        _projection_input(
            rates=(
                _rate(SOGGameState.FIVE_ON_FIVE, 8.0),
                _rate(SOGGameState.POWER_PLAY, 12.0),
            ),
        )
    )

    assert result.base_expected_sog == 0.0
    assert result.adjusted_expected_sog == 0.0
    assert result.confidence_tier is ConfidenceTier.NO_PROJECTION
    assert SOGQualityFlag.MISSING_PLAYER_RATE in result.quality_flags


def test_project_sog_applies_contextual_multipliers() -> None:
    """Apply all bounded contextual multipliers after base aggregation."""

    adjustment = SOGContextualAdjustment(
        adjustment_key="opponent_shot_suppression",
        requested_multiplier=1.1,
        lower_bound=0.8,
        upper_bound=1.2,
        applied_multiplier=1.1,
        as_of_timestamp=TIMESTAMP,
    )

    result = project_sog(
        _projection_input(
            contextual_adjustments=(adjustment,),
        )
    )

    assert result.base_expected_sog == pytest.approx(2.65)
    assert result.adjusted_expected_sog == pytest.approx(2.915)


def test_project_sog_propagates_role_quality_to_confidence() -> None:
    """Downgrade confidence when projected player-role data is used."""

    result = project_sog(
        _projection_input(
            input_quality_flags=(
                SOGQualityFlag.PROJECTED_ROLE,
            ),
        )
    )

    assert result.confidence_tier is ConfidenceTier.MEDIUM
    assert SOGQualityFlag.PROJECTED_ROLE in result.quality_flags


def test_project_sog_rejects_duplicate_game_state_rates() -> None:
    """Fail rather than silently choosing between duplicate state rates."""

    projection_input = _projection_input(
        rates=(
            _rate(SOGGameState.FIVE_ON_FIVE, 8.0),
            _rate(SOGGameState.FIVE_ON_FIVE, 9.0),
        ),
    )

    with pytest.raises(
        ValueError,
        match="multiple shrunk rates found for the same game state",
    ):
        project_sog(projection_input)