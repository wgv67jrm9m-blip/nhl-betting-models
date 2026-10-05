"""Convert canonical SOG projection inputs into expected-SOG results."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

from nhl_betting_models.nhl.schemas import (
    ConfidenceTier,
    SOGContextualAdjustment,
    SOGGameState,
    SOGProjectionComponent,
    SOGProjectionInput,
    SOGProjectionResult,
    SOGQualityFlag,
    SOGShrunkRate,
)


def _deduplicate_flags(
    flags: Iterable[SOGQualityFlag],
) -> tuple[SOGQualityFlag, ...]:
    """Return quality flags in deterministic first-seen order."""

    return tuple(dict.fromkeys(flags))


def _rates_by_game_state(
    projection_input: SOGProjectionInput,
) -> dict[SOGGameState, SOGShrunkRate]:
    """Index unique shrunk rates by their modeled game state."""

    rates: dict[SOGGameState, SOGShrunkRate] = {}

    for rate in projection_input.shrunk_rates:
        if rate.game_state in rates:
            raise ValueError(
                "multiple shrunk rates found for the same game state"
            )
        rates[rate.game_state] = rate

    return rates


def _contextual_multiplier(
    adjustments: Iterable[SOGContextualAdjustment],
) -> float:
    """Return the combined finite multiplicative contextual adjustment."""

    multiplier = 1.0

    for adjustment in adjustments:
        if not isfinite(adjustment.applied_multiplier):
            raise ValueError(
                "contextual adjustment multiplier must be finite"
            )

        if adjustment.applied_multiplier <= 0.0:
            raise ValueError(
                "contextual adjustment multiplier must be positive"
            )

        multiplier *= adjustment.applied_multiplier

    if not isfinite(multiplier):
        raise ValueError(
            "combined contextual adjustment multiplier must be finite"
        )

    return multiplier


def _confidence_tier(
    *,
    quality_flags: tuple[SOGQualityFlag, ...],
    has_projection: bool,
) -> ConfidenceTier:
    """Assign an input-completeness tier for one SOG projection."""

    if not has_projection:
        return ConfidenceTier.NO_PROJECTION

    if any(
        flag
        in {
            SOGQualityFlag.MISSING_PROJECTED_MINUTES,
            SOGQualityFlag.MISSING_PLAYER_RATE,
            SOGQualityFlag.MISSING_POSITION_PRIOR,
            SOGQualityFlag.INVALID_CONTEXTUAL_ADJUSTMENT,
            SOGQualityFlag.POST_FORECAST_FEATURE,
        }
        for flag in quality_flags
    ):
        return ConfidenceTier.NO_PROJECTION

    if any(
        flag
        in {
            SOGQualityFlag.UNCERTAIN_ROLE,
            SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE,
            SOGQualityFlag.MISSING_CONTEXT,
            SOGQualityFlag.STALE_FEATURE,
        }
        for flag in quality_flags
    ):
        return ConfidenceTier.LOW

    if SOGQualityFlag.PROJECTED_ROLE in quality_flags:
        return ConfidenceTier.MEDIUM

    return ConfidenceTier.HIGH


def project_sog(
    projection_input: SOGProjectionInput,
) -> SOGProjectionResult:
    """Build one auditable expected-SOG projection from canonical inputs.

    A game-state contribution is projected as:

        projected_minutes / 60 * posterior_sog_per_60

    Contextual multipliers are applied only after all base game-state
    contributions are summed.
    """

    rates_by_state = _rates_by_game_state(projection_input)
    flags: list[SOGQualityFlag] = [
        *projection_input.input_quality_flags,
        *projection_input.projected_minutes.quality_flags,
    ]
    components: list[SOGProjectionComponent] = []
    base_expected_sog = 0.0
    has_required_rate = True

    for game_state in SOGGameState:
        projected_minutes = projection_input.projected_minutes.minutes_for(
            game_state
        )
        rate = rates_by_state.get(game_state)

        if projected_minutes > 0.0 and rate is None:
            flags.append(SOGQualityFlag.MISSING_PLAYER_RATE)
            has_required_rate = False
            continue

        if rate is None:
            continue

        flags.extend(rate.quality_flags)

        expected_sog = (
            projected_minutes
            * rate.posterior_sog_per_60
            / 60.0
        )
        components.append(
            SOGProjectionComponent(
                game_state=game_state,
                projected_minutes=projected_minutes,
                sog_per_60=rate.posterior_sog_per_60,
                expected_sog=expected_sog,
            )
        )
        base_expected_sog += expected_sog

    if not components:
        flags.append(SOGQualityFlag.MISSING_PROJECTED_MINUTES)

    quality_flags = _deduplicate_flags(flags)
    has_projection = has_required_rate and bool(components)

    if not has_projection:
        return SOGProjectionResult(
            canonical_event_id=projection_input.canonical_event_id,
            canonical_player_id=projection_input.canonical_player_id,
            model_run_timestamp=projection_input.model_run_timestamp,
            base_expected_sog=0.0,
            adjusted_expected_sog=0.0,
            components=tuple(components),
            contextual_adjustments=(
                projection_input.contextual_adjustments
            ),
            confidence_tier=_confidence_tier(
                quality_flags=quality_flags,
                has_projection=False,
            ),
            quality_flags=quality_flags,
        )

    try:
        multiplier = _contextual_multiplier(
            projection_input.contextual_adjustments
        )
    except ValueError:
        quality_flags = _deduplicate_flags(
            (
                *quality_flags,
                SOGQualityFlag.INVALID_CONTEXTUAL_ADJUSTMENT,
            )
        )
        return SOGProjectionResult(
            canonical_event_id=projection_input.canonical_event_id,
            canonical_player_id=projection_input.canonical_player_id,
            model_run_timestamp=projection_input.model_run_timestamp,
            base_expected_sog=base_expected_sog,
            adjusted_expected_sog=0.0,
            components=tuple(components),
            contextual_adjustments=(
                projection_input.contextual_adjustments
            ),
            confidence_tier=ConfidenceTier.NO_PROJECTION,
            quality_flags=quality_flags,
        )

    adjusted_expected_sog = base_expected_sog * multiplier

    return SOGProjectionResult(
        canonical_event_id=projection_input.canonical_event_id,
        canonical_player_id=projection_input.canonical_player_id,
        model_run_timestamp=projection_input.model_run_timestamp,
        base_expected_sog=base_expected_sog,
        adjusted_expected_sog=adjusted_expected_sog,
        components=tuple(components),
        contextual_adjustments=projection_input.contextual_adjustments,
        confidence_tier=_confidence_tier(
            quality_flags=quality_flags,
            has_projection=True,
        ),
        quality_flags=quality_flags,
    )