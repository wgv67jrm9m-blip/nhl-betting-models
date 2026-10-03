"""Synthetic tests for NHL SOG empirical-Bayes feature preparation."""

from datetime import UTC, datetime, timedelta

import pytest

from nhl_betting_models.nhl.schemas import (
    PlayerPosition,
    SOGGameState,
    SOGQualityFlag,
    SOGRateObservation,
    SOGRatePrior,
)
from nhl_betting_models.nhl.sog_features import shrink_sog_rate

MODEL_RUN = datetime(2026, 1, 1, 18, tzinfo=UTC)
FEATURE_TIME = datetime(2026, 1, 1, 17, tzinfo=UTC)


def synthetic_observation(
    *,
    sog_per_60: float,
    exposure_minutes: float,
    as_of_timestamp: datetime = FEATURE_TIME,
) -> SOGRateObservation:
    """SYNTHETIC TEST FIXTURE: no real NHL player data."""

    return SOGRateObservation(
        canonical_player_id="synthetic-player-1",
        position=PlayerPosition.FORWARD,
        game_state=SOGGameState.FIVE_ON_FIVE,
        sog_per_60=sog_per_60,
        exposure_minutes=exposure_minutes,
        as_of_timestamp=as_of_timestamp,
    )


def synthetic_prior() -> SOGRatePrior:
    """SYNTHETIC TEST FIXTURE: no empirically fitted NHL prior."""

    return SOGRatePrior(
        position=PlayerPosition.FORWARD,
        game_state=SOGGameState.FIVE_ON_FIVE,
        mean_sog_per_60=2.0,
        prior_equivalent_minutes=100.0,
        prior_version="synthetic-prior-v1",
    )


def test_low_exposure_rate_shrinks_more_toward_prior() -> None:
    prior = synthetic_prior()

    low_exposure = shrink_sog_rate(
        synthetic_observation(
            sog_per_60=8.0,
            exposure_minutes=10.0,
        ),
        prior,
        model_run_timestamp=MODEL_RUN,
    )
    high_exposure = shrink_sog_rate(
        synthetic_observation(
            sog_per_60=8.0,
            exposure_minutes=1000.0,
        ),
        prior,
        model_run_timestamp=MODEL_RUN,
    )

    assert low_exposure.posterior_sog_per_60 is not None
    assert high_exposure.posterior_sog_per_60 is not None

    prior_rate = prior.mean_sog_per_60

    assert abs(
        low_exposure.posterior_sog_per_60 - prior_rate
    ) < abs(
        high_exposure.posterior_sog_per_60 - prior_rate
    )
    assert low_exposure.prior_weight > high_exposure.prior_weight
    assert low_exposure.observed_weight < high_exposure.observed_weight


def test_empirical_bayes_weights_and_posterior_are_correct() -> None:
    result = shrink_sog_rate(
        synthetic_observation(
            sog_per_60=6.0,
            exposure_minutes=50.0,
        ),
        synthetic_prior(),
        model_run_timestamp=MODEL_RUN,
    )

    assert result.has_projection
    assert result.observed_weight == pytest.approx(1.0 / 3.0)
    assert result.prior_weight == pytest.approx(2.0 / 3.0)
    assert result.posterior_sog_per_60 == pytest.approx(
        (1.0 / 3.0) * 6.0 + (2.0 / 3.0) * 2.0
    )


def test_missing_observation_and_prior_returns_safe_no_projection() -> None:
    result = shrink_sog_rate(
        None,
        None,
        model_run_timestamp=MODEL_RUN,
    )

    assert not result.has_projection
    assert result.shrunk_rate is None
    assert result.observed_weight == 0.0
    assert result.prior_weight == 0.0
    assert result.posterior_sog_per_60 is None
    assert SOGQualityFlag.MISSING_PLAYER_RATE in result.quality_flags
    assert SOGQualityFlag.MISSING_POSITION_PRIOR in result.quality_flags


def test_missing_matching_prior_returns_safe_no_projection() -> None:
    result = shrink_sog_rate(
        synthetic_observation(
            sog_per_60=5.0,
            exposure_minutes=40.0,
        ),
        None,
        model_run_timestamp=MODEL_RUN,
    )

    assert not result.has_projection
    assert result.shrunk_rate is None
    assert SOGQualityFlag.MISSING_POSITION_PRIOR in result.quality_flags


def test_small_observed_sample_is_flagged() -> None:
    result = shrink_sog_rate(
        synthetic_observation(
            sog_per_60=5.0,
            exposure_minutes=20.0,
        ),
        synthetic_prior(),
        model_run_timestamp=MODEL_RUN,
        minimum_player_exposure_minutes=30.0,
    )

    assert result.has_projection
    assert (
        SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE
        in result.quality_flags
    )


def test_post_model_run_feature_timestamp_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="newer than model-run timestamp",
    ):
        shrink_sog_rate(
            synthetic_observation(
                sog_per_60=5.0,
                exposure_minutes=40.0,
                as_of_timestamp=MODEL_RUN + timedelta(seconds=1),
            ),
            synthetic_prior(),
            model_run_timestamp=MODEL_RUN,
        )