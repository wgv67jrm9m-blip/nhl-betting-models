"""Feature preparation and empirical-Bayes shrinkage for NHL player SOG models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from nhl_betting_models.common.validation import reject_post_forecast_features
from nhl_betting_models.nhl.schemas import (
    PlayerPosition,
    SOGGameState,
    SOGQualityFlag,
    SOGRateObservation,
    SOGRatePrior,
    SOGShrunkRate,
)


@dataclass(frozen=True, slots=True)
class SOGShrinkageResult:
    """Auditable empirical-Bayes shrinkage result for one game state."""

    shrunk_rate: SOGShrunkRate | None
    observed_weight: float
    prior_weight: float
    quality_flags: tuple[SOGQualityFlag, ...]

    @property
    def has_projection(self) -> bool:
        """Return whether a usable posterior SOG rate was produced."""

        return self.shrunk_rate is not None

    @property
    def observed_sog_per_60(self) -> float | None:
        """Return the observed SOG/60 when a posterior exists."""

        if self.shrunk_rate is None:
            return None
        return self.shrunk_rate.observed_sog_per_60

    @property
    def observed_exposure_minutes(self) -> float | None:
        """Return observed exposure minutes when a posterior exists."""

        if self.shrunk_rate is None:
            return None
        return self.shrunk_rate.observed_exposure_minutes

    @property
    def prior_sog_per_60(self) -> float | None:
        """Return the prior SOG/60 when a posterior exists."""

        if self.shrunk_rate is None:
            return None
        return self.shrunk_rate.prior_sog_per_60

    @property
    def prior_equivalent_minutes(self) -> float | None:
        """Return prior equivalent minutes when a posterior exists."""

        if self.shrunk_rate is None:
            return None
        return self.shrunk_rate.prior_equivalent_minutes

    @property
    def posterior_sog_per_60(self) -> float | None:
        """Return the posterior SOG/60 when available."""

        if self.shrunk_rate is None:
            return None
        return self.shrunk_rate.posterior_sog_per_60


def _deduplicate_flags(
    flags: Iterable[SOGQualityFlag],
) -> tuple[SOGQualityFlag, ...]:
    """Return quality flags in deterministic first-seen order."""

    return tuple(dict.fromkeys(flags))


def find_rate_prior(
    priors: Iterable[SOGRatePrior],
    *,
    position: PlayerPosition,
    game_state: SOGGameState,
) -> SOGRatePrior | None:
    """Find the unique configured prior for a position and game state."""

    matches = [
        prior
        for prior in priors
        if prior.position == position
        and prior.game_state == game_state
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            "multiple SOG priors found for the same position and game state"
        )

    return matches[0]


def shrink_sog_rate(
    observation: SOGRateObservation | None,
    prior: SOGRatePrior | None,
    *,
    model_run_timestamp: datetime,
    minimum_player_exposure_minutes: float = 0.0,
) -> SOGShrinkageResult:
    """Shrink one observed player SOG/60 rate toward its configured prior."""

    if not isfinite(minimum_player_exposure_minutes):
        raise ValueError(
            "minimum_player_exposure_minutes must be finite"
        )

    if minimum_player_exposure_minutes < 0.0:
        raise ValueError(
            "minimum_player_exposure_minutes must be non-negative"
        )

    flags: list[SOGQualityFlag] = []

    if observation is None:
        flags.append(SOGQualityFlag.MISSING_PLAYER_RATE)

    if prior is None:
        flags.append(SOGQualityFlag.MISSING_POSITION_PRIOR)

    if observation is None or prior is None:
        return SOGShrinkageResult(
            shrunk_rate=None,
            observed_weight=0.0,
            prior_weight=0.0,
            quality_flags=_deduplicate_flags(flags),
        )

    reject_post_forecast_features(
        [observation.as_of_timestamp],
        model_run_timestamp,
    )

    if observation.position != prior.position:
        raise ValueError(
            "observation and prior positions must match"
        )

    if observation.game_state != prior.game_state:
        raise ValueError(
            "observation and prior game states must match"
        )

    if not isfinite(observation.sog_per_60):
        raise ValueError(
            "observed SOG/60 must be finite"
        )

    if not isfinite(observation.exposure_minutes):
        raise ValueError(
            "observed exposure minutes must be finite"
        )

    if not isfinite(prior.mean_sog_per_60):
        raise ValueError(
            "prior SOG/60 must be finite"
        )

    if not isfinite(prior.prior_equivalent_minutes):
        raise ValueError(
            "prior equivalent minutes must be finite"
        )

    if observation.exposure_minutes < minimum_player_exposure_minutes:
        flags.append(
            SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE
        )

    total_equivalent_minutes = (
        observation.exposure_minutes
        + prior.prior_equivalent_minutes
    )

    if total_equivalent_minutes <= 0.0:
        return SOGShrinkageResult(
            shrunk_rate=None,
            observed_weight=0.0,
            prior_weight=0.0,
            quality_flags=_deduplicate_flags(
                (
                    *flags,
                    SOGQualityFlag.MISSING_PLAYER_RATE,
                    SOGQualityFlag.MISSING_POSITION_PRIOR,
                )
            ),
        )

    observed_weight = (
        observation.exposure_minutes
        / total_equivalent_minutes
    )
    prior_weight = (
        prior.prior_equivalent_minutes
        / total_equivalent_minutes
    )

    posterior_sog_per_60 = (
        observed_weight * observation.sog_per_60
        + prior_weight * prior.mean_sog_per_60
    )

    quality_flags = _deduplicate_flags(flags)

    shrunk_rate = SOGShrunkRate(
        canonical_player_id=observation.canonical_player_id,
        position=observation.position,
        game_state=observation.game_state,
        observed_sog_per_60=observation.sog_per_60,
        observed_exposure_minutes=observation.exposure_minutes,
        prior_sog_per_60=prior.mean_sog_per_60,
        prior_equivalent_minutes=prior.prior_equivalent_minutes,
        posterior_sog_per_60=posterior_sog_per_60,
        as_of_timestamp=observation.as_of_timestamp,
        quality_flags=quality_flags,
    )

    return SOGShrinkageResult(
        shrunk_rate=shrunk_rate,
        observed_weight=observed_weight,
        prior_weight=prior_weight,
        quality_flags=quality_flags,
    )


def shrink_player_game_state_rates(
    observations: Iterable[SOGRateObservation],
    priors: Iterable[SOGRatePrior],
    *,
    canonical_player_id: str,
    position: PlayerPosition,
    model_run_timestamp: datetime,
    minimum_player_exposure_minutes: float = 0.0,
) -> dict[SOGGameState, SOGShrinkageResult]:
    """Build timestamp-safe shrinkage results for all modeled game states."""

    observation_list = [
        observation
        for observation in observations
        if observation.canonical_player_id == canonical_player_id
    ]
    prior_list = list(priors)

    reject_post_forecast_features(
        [
            observation.as_of_timestamp
            for observation in observation_list
        ],
        model_run_timestamp,
    )

    observations_by_state: dict[
        SOGGameState,
        SOGRateObservation,
    ] = {}

    for observation in observation_list:
        if observation.position != position:
            raise ValueError(
                "player observation position does not match requested position"
            )

        if observation.game_state in observations_by_state:
            raise ValueError(
                "multiple SOG observations found for the same player and game state"
            )

        observations_by_state[observation.game_state] = observation

    results: dict[
        SOGGameState,
        SOGShrinkageResult,
    ] = {}

    for game_state in SOGGameState:
        state_observation = observations_by_state.get(game_state)
        prior = find_rate_prior(
            prior_list,
            position=position,
            game_state=game_state,
        )

        results[game_state] = shrink_sog_rate(
            state_observation,
            prior,
            model_run_timestamp=model_run_timestamp,
            minimum_player_exposure_minutes=minimum_player_exposure_minutes,
        )

    return results