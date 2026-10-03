"""Synthetic tests for transparent NHL SOG projection."""

from datetime import UTC, datetime, timedelta

import pytest

from nhl_betting_models.nhl.schemas import (
    ConfidenceTier,
    PlayerPosition,
    SOGGameState,
    SOGProjectedMinutes,
    SOGProjectionInput,
    SOGQualityFlag,
    SOGShrunkRate,
)
from nhl_betting_models.nhl.sog_projection import project_expected_sog

MODEL_RUN = datetime(2026, 1, 1, 18, tzinfo=UTC)
FEATURE_TIME = datetime(2026, 1, 1, 17, tzinfo=UTC)


def synthetic_rate(
    *,
    game_state: SOGGameState,
    posterior_sog_per_60: float,
    observed_exposure_minutes: float = 300.0,
    as_of_timestamp: datetime = FEATURE_TIME,
) -> SOGShrunkRate:
    """SYNTHETIC TEST FIXTURE: no real NHL player rate."""

    return SOGShrunkRate(
        canonical_player_id="synthetic-player-1",
        position=PlayerPosition.FORWARD,
        game_state=game_state,
        observed_sog_per_60=posterior_sog_per_60,
        observed_exposure_minutes=observed_exposure_minutes,
        prior_sog_per_60=posterior_sog_per_60,
        prior_equivalent_minutes=100.0,
        posterior_sog_per_60=posterior_sog_per_60,
        as_of_timestamp=as_of_timestamp,
    )


def synthetic_projection_input(
    *,
    five_on_five_minutes: float = 14.0,
    power_play_minutes: float = 3.0,
    short_handed_minutes: float = 0.0,
    projected_minutes_as_of: datetime = FEATURE_TIME,
    input_quality_flags: tuple[SOGQualityFlag, ...] = (),
) -> SOGProjectionInput:
    """SYNTHETIC TEST FIXTURE: no live role or TOI projection."""

    return SOGProjectionInput(
        canonical_event_id="synthetic-game-1",
        canonical_player_id="synthetic-player-1",
        position=PlayerPosition.FORWARD,
        model_run_timestamp=MODEL_RUN,
        projected_minutes=SOGProjectedMinutes(
            canonical_player_id="synthetic-player-1",
            five_on_five_minutes=five_on_five_minutes,
            power_play_minutes=power_play_minutes,
            short_handed_minutes=short_handed_minutes,
            as_of_timestamp=projected_minutes_as_of,
        ),
        shrunk_rates=(
            synthetic_rate(
                game_state=SOGGameState.FIVE_ON_FIVE,
                posterior_sog_per_60=9.0,
            ),
            synthetic_rate(
                game_state=SOGGameState.POWER_PLAY,
                posterior_sog_per_60=12.0,
            ),
        ),
        input_quality_flags=input_quality_flags,
    )


def test_expected_sog_uses_game_state_components() -> None:
    projection = project_expected_sog(
        synthetic_projection_input()
    )

    expected_five_on_five = 14.0 * 9.0 / 60.0
    expected_power_play = 3.0 * 12.0 / 60.0

    assert projection.base_expected_sog == pytest.approx(
        expected_five_on_five + expected_power_play
    )
    assert projection.adjusted_expected_sog == pytest.approx(
        projection.base_expected_sog
    )


def test_higher_five_on_five_minutes_increase_expected_sog() -> None:
    baseline = project_expected_sog(
        synthetic_projection_input(
            five_on_five_minutes=14.0,
        )
    )
    increased = project_expected_sog(
        synthetic_projection_input(
            five_on_five_minutes=17.0,
        )
    )

    assert increased.base_expected_sog > baseline.base_expected_sog
    assert (
        increased.base_expected_sog
        - baseline.base_expected_sog
    ) == pytest.approx(
        3.0 * 9.0 / 60.0
    )


def test_higher_power_play_minutes_increase_expected_sog() -> None:
    baseline = project_expected_sog(
        synthetic_projection_input(
            power_play_minutes=2.0,
        )
    )
    increased = project_expected_sog(
        synthetic_projection_input(
            power_play_minutes=5.0,
        )
    )

    assert increased.base_expected_sog > baseline.base_expected_sog
    assert (
        increased.base_expected_sog
        - baseline.base_expected_sog
    ) == pytest.approx(
        3.0 * 12.0 / 60.0
    )


def test_missing_projected_minutes_returns_no_projection() -> None:
    projection = project_expected_sog(
        synthetic_projection_input(
            five_on_five_minutes=0.0,
            power_play_minutes=0.0,
            short_handed_minutes=0.0,
        )
    )

    assert (
        projection.confidence_tier
        is ConfidenceTier.NO_PROJECTION
    )
    assert projection.base_expected_sog == 0.0
    assert projection.adjusted_expected_sog == 0.0
    assert (
        SOGQualityFlag.MISSING_PROJECTED_MINUTES
        in projection.quality_flags
    )


def test_future_dated_feature_returns_no_projection() -> None:
    projection = project_expected_sog(
        synthetic_projection_input(
            projected_minutes_as_of=(
                MODEL_RUN + timedelta(seconds=1)
            ),
        )
    )

    assert (
        projection.confidence_tier
        is ConfidenceTier.NO_PROJECTION
    )
    assert (
        SOGQualityFlag.POST_FORECAST_FEATURE
        in projection.quality_flags
    )


def test_uncertain_role_produces_low_confidence() -> None:
    projection = project_expected_sog(
        synthetic_projection_input(
            input_quality_flags=(
                SOGQualityFlag.UNCERTAIN_ROLE,
            ),
        )
    )

    assert projection.confidence_tier is ConfidenceTier.LOW