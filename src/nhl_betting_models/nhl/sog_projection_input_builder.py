"""Build canonical SOG projection inputs from eligible slate candidates."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from nhl_betting_models.data.base import ConfirmationStatus
from nhl_betting_models.data.canonical_ids import (
    CanonicalEntityType,
    CanonicalIdMap,
    IdentityResolutionStatus,
)
from nhl_betting_models.nhl.schemas import (
    SOGGameState,
    SOGProjectedMinutes,
    SOGProjectionInput,
    SOGQualityFlag,
    SOGRateObservation,
    SOGRatePrior,
    SOGShrunkRate,
)
from nhl_betting_models.nhl.sog_features import (
    shrink_player_game_state_rates,
)
from nhl_betting_models.nhl.sog_projection_candidates import (
    SOGProjectionCandidate,
    SOGProjectionCandidateResult,
)


def _resolve_required_id(
    identity_map: CanonicalIdMap,
    *,
    entity_type: CanonicalEntityType,
    source_name: str,
    source_entity_id: str,
) -> str:
    """Resolve one source identity or raise a projection-input error."""

    resolution = identity_map.resolve(
        entity_type=entity_type,
        source_name=source_name,
        source_entity_id=source_entity_id,
    )

    if resolution.status is not IdentityResolutionStatus.RESOLVED:
        raise ValueError(
            "unable to resolve required canonical "
            f"{entity_type.value} ID: "
            f"{source_name}:{source_entity_id}"
        )

    assert resolution.canonical_id is not None
    return resolution.canonical_id


def _canonical_event_id(
    candidate: SOGProjectionCandidate,
    identity_map: CanonicalIdMap,
) -> str:
    """Resolve a candidate game through the role-projection source."""

    return _resolve_required_id(
        identity_map,
        entity_type=CanonicalEntityType.GAME,
        source_name=candidate.role_projection.provenance.source_name,
        source_entity_id=candidate.source_game_id,
    )


def _canonical_player_id(
    candidate: SOGProjectionCandidate,
    identity_map: CanonicalIdMap,
) -> str:
    """Resolve and reconcile role and statistic player identities."""

    role_player_id = _resolve_required_id(
        identity_map,
        entity_type=CanonicalEntityType.PLAYER,
        source_name=candidate.role_projection.provenance.source_name,
        source_entity_id=candidate.source_player_id,
    )

    stat_player_ids = {
        _resolve_required_id(
            identity_map,
            entity_type=CanonicalEntityType.PLAYER,
            source_name=player_stat.provenance.source_name,
            source_entity_id=player_stat.source_player_id,
        )
        for player_stat in candidate.player_game_stats
    }

    if stat_player_ids != {role_player_id}:
        raise ValueError(
            "role projection and player statistics resolve to "
            "different canonical player IDs"
        )

    return role_player_id


def _projected_minutes(
    candidate: SOGProjectionCandidate,
    canonical_player_id: str,
) -> SOGProjectedMinutes:
    """Build model-ready game-state minutes from one role projection."""

    role_projection = candidate.role_projection
    quality_flags: list[SOGQualityFlag] = []

    if role_projection.confirmation_status is ConfirmationStatus.PROJECTED:
        quality_flags.append(SOGQualityFlag.PROJECTED_ROLE)
    elif role_projection.confirmation_status in {
        ConfirmationStatus.UNCONFIRMED,
        ConfirmationStatus.UNKNOWN,
    }:
        quality_flags.append(SOGQualityFlag.UNCERTAIN_ROLE)

    return SOGProjectedMinutes(
        canonical_player_id=canonical_player_id,
        five_on_five_minutes=(
            role_projection.five_on_five_minutes or 0.0
        ),
        power_play_minutes=role_projection.power_play_minutes or 0.0,
        short_handed_minutes=(
            role_projection.short_handed_minutes or 0.0
        ),
        as_of_timestamp=role_projection.provenance.as_of_timestamp,
        quality_flags=tuple(quality_flags),
    )


def _rate_observations(
    candidate: SOGProjectionCandidate,
    canonical_player_id: str,
) -> tuple[SOGRateObservation, ...]:
    """Convert candidate game-state statistics into rate observations."""

    observations: list[SOGRateObservation] = []

    for player_stat in candidate.player_game_stats:
        if player_stat.exposure_minutes <= 0.0:
            sog_per_60 = 0.0
        else:
            sog_per_60 = (
                player_stat.shots_on_goal
                * 60.0
                / player_stat.exposure_minutes
            )

        observations.append(
            SOGRateObservation(
                canonical_player_id=canonical_player_id,
                position=candidate.position,
                game_state=player_stat.game_state,
                sog_per_60=sog_per_60,
                exposure_minutes=player_stat.exposure_minutes,
                as_of_timestamp=(
                    player_stat.provenance.as_of_timestamp
                ),
            )
        )

    return tuple(observations)


def _shrunk_rates(
    candidate: SOGProjectionCandidate,
    canonical_player_id: str,
    priors: Iterable[SOGRatePrior],
    *,
    model_run_timestamp: datetime,
    minimum_player_exposure_minutes: float,
) -> tuple[SOGShrunkRate, ...]:
    """Build available empirical-Bayes player rates for one candidate."""

    shrinkage_results = shrink_player_game_state_rates(
        _rate_observations(candidate, canonical_player_id),
        priors,
        canonical_player_id=canonical_player_id,
        position=candidate.position,
        model_run_timestamp=model_run_timestamp,
        minimum_player_exposure_minutes=minimum_player_exposure_minutes,
    )

    return tuple(
        result.shrunk_rate
        for game_state in SOGGameState
        if (
            result := shrinkage_results[game_state]
        ).shrunk_rate is not None
    )


def build_sog_projection_inputs(
    candidate_results: Iterable[SOGProjectionCandidateResult],
    identity_map: CanonicalIdMap,
    priors: Iterable[SOGRatePrior],
    *,
    model_run_timestamp: datetime,
    minimum_player_exposure_minutes: float = 0.0,
) -> tuple[SOGProjectionInput, ...]:
    """Build model-ready inputs for eligible validated-slate candidates."""

    prior_list = tuple(priors)
    projection_inputs: list[SOGProjectionInput] = []

    for candidate_result in candidate_results:
        candidate = candidate_result.candidate

        if candidate is None:
            continue

        canonical_event_id = _canonical_event_id(
            candidate,
            identity_map,
        )
        canonical_player_id = _canonical_player_id(
            candidate,
            identity_map,
        )
        projected_minutes = _projected_minutes(
            candidate,
            canonical_player_id,
        )
        shrunk_rates = _shrunk_rates(
            candidate,
            canonical_player_id,
            prior_list,
            model_run_timestamp=model_run_timestamp,
            minimum_player_exposure_minutes=(
                minimum_player_exposure_minutes
            ),
        )

        projection_inputs.append(
            SOGProjectionInput(
                canonical_event_id=canonical_event_id,
                canonical_player_id=canonical_player_id,
                position=candidate.position,
                model_run_timestamp=model_run_timestamp,
                projected_minutes=projected_minutes,
                shrunk_rates=shrunk_rates,
                contextual_adjustments=(),
                input_quality_flags=(),
            )
        )

    return tuple(projection_inputs)