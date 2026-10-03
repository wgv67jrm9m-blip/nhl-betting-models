"""Transparent expected-SOG projection from game-state opportunity and rates."""

from __future__ import annotations

import math
from collections.abc import Iterable

from nhl_betting_models.common.validation import reject_post_forecast_features
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
    """Return flags in deterministic first-seen order."""

    return tuple(dict.fromkeys(flags))


def _rate_by_game_state(
    rates: tuple[SOGShrunkRate, ...],
) -> dict[SOGGameState, SOGShrunkRate]:
    """Index unique shrunk rates by modeled game state."""

    indexed: dict[SOGGameState, SOGShrunkRate] = {}

    for rate in rates:
        if rate.game_state in indexed:
            raise ValueError(
                "multiple shrunk SOG rates supplied for the same game state"
            )

        indexed[rate.game_state] = rate

    return indexed


def _validate_adjustment(
    adjustment: SOGContextualAdjustment,
) -> None:
    """Validate a named bounded contextual adjustment."""

    if not adjustment.adjustment_key.strip():
        raise ValueError(
            "contextual adjustment key must not be empty"
        )

    values = (
        adjustment.requested_multiplier,
        adjustment.lower_bound,
        adjustment.upper_bound,
        adjustment.applied_multiplier,
    )

    if any(not math.isfinite(value) for value in values):
        raise ValueError(
            "contextual adjustment values must be finite"
        )

    bounded_multiplier = min(
        max(
            adjustment.requested_multiplier,
            adjustment.lower_bound,
        ),
        adjustment.upper_bound,
    )

    if not math.isclose(
        adjustment.applied_multiplier,
        bounded_multiplier,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "applied contextual multiplier must equal the bounded "
            "requested multiplier"
        )


def _has_future_dated_inputs(
    projection_input: SOGProjectionInput,
) -> bool:
    """Detect inputs newer than the model-run timestamp."""

    timestamps = [
        projection_input.projected_minutes.as_of_timestamp,
        *(
            rate.as_of_timestamp
            for rate in projection_input.shrunk_rates
        ),
        *(
            adjustment.as_of_timestamp
            for adjustment in projection_input.contextual_adjustments
        ),
    ]

    try:
        reject_post_forecast_features(
            timestamps,
            projection_input.model_run_timestamp,
        )
    except ValueError as exc:
        if "newer than model-run timestamp" in str(exc):
            return True
        raise

    return False


def _no_projection(
    projection_input: SOGProjectionInput,
    *,
    flags: Iterable[SOGQualityFlag],
) -> SOGProjectionResult:
    """Return an explicit no-projection result."""

    return SOGProjectionResult(
        canonical_event_id=projection_input.canonical_event_id,
        canonical_player_id=projection_input.canonical_player_id,
        model_run_timestamp=projection_input.model_run_timestamp,
        base_expected_sog=0.0,
        adjusted_expected_sog=0.0,
        components=(),
        contextual_adjustments=(),
        confidence_tier=ConfidenceTier.NO_PROJECTION,
        quality_flags=_deduplicate_flags(flags),
    )


def _detect_rate_quality(
    rate: SOGShrunkRate,
) -> tuple[SOGQualityFlag, ...]:
    """Collect rate flags and detect a prior-dominant observation."""

    flags = list(rate.quality_flags)

    if (
        rate.observed_exposure_minutes
        < rate.prior_equivalent_minutes
    ):
        flags.append(
            SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE
        )

    return _deduplicate_flags(flags)


def _confidence_from_flags(
    flags: tuple[SOGQualityFlag, ...],
) -> ConfidenceTier:
    """Map input completeness and certainty to a confidence tier."""

    no_projection_flags = {
        SOGQualityFlag.MISSING_PROJECTED_MINUTES,
        SOGQualityFlag.MISSING_PLAYER_RATE,
        SOGQualityFlag.MISSING_POSITION_PRIOR,
        SOGQualityFlag.POST_FORECAST_FEATURE,
        SOGQualityFlag.INVALID_CONTEXTUAL_ADJUSTMENT,
    }

    low_confidence_flags = {
        SOGQualityFlag.UNCERTAIN_ROLE,
        SOGQualityFlag.MISSING_CONTEXT,
        SOGQualityFlag.STALE_FEATURE,
        SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE,
    }

    if set(flags) & no_projection_flags:
        return ConfidenceTier.NO_PROJECTION

    if set(flags) & low_confidence_flags:
        return ConfidenceTier.LOW

    if SOGQualityFlag.PROJECTED_ROLE in flags:
        return ConfidenceTier.MEDIUM

    return ConfidenceTier.HIGH


def project_expected_sog(
    projection_input: SOGProjectionInput,
) -> SOGProjectionResult:
    """Project expected SOG from 5v5, PP, and short-handed opportunity.

    The unadjusted expectation is the sum of each modeled game state's
    projected minutes multiplied by its posterior SOG/60 and divided by 60.
    Valid contextual multipliers are then applied in supplied order.
    """

    flags: list[SOGQualityFlag] = list(
        projection_input.input_quality_flags
    )
    flags.extend(
        projection_input.projected_minutes.quality_flags
    )

    if _has_future_dated_inputs(projection_input):
        flags.append(
            SOGQualityFlag.POST_FORECAST_FEATURE
        )
        return _no_projection(
            projection_input,
            flags=flags,
        )

    total_minutes = (
        projection_input.projected_minutes.total_minutes
    )

    if not math.isfinite(total_minutes):
        raise ValueError(
            "projected total minutes must be finite"
        )

    if total_minutes <= 0.0:
        flags.append(
            SOGQualityFlag.MISSING_PROJECTED_MINUTES
        )
        return _no_projection(
            projection_input,
            flags=flags,
        )

    rates = _rate_by_game_state(
        projection_input.shrunk_rates
    )

    components: list[SOGProjectionComponent] = []

    for game_state in SOGGameState:
        projected_minutes = (
            projection_input.projected_minutes.minutes_for(
                game_state
            )
        )

        if not math.isfinite(projected_minutes):
            raise ValueError(
                f"projected minutes for {game_state.value} "
                "must be finite"
            )

        if projected_minutes == 0.0:
            components.append(
                SOGProjectionComponent(
                    game_state=game_state,
                    projected_minutes=0.0,
                    sog_per_60=0.0,
                    expected_sog=0.0,
                )
            )
            continue

        rate = rates.get(game_state)

        if rate is None:
            flags.append(
                SOGQualityFlag.MISSING_PLAYER_RATE
            )
            return _no_projection(
                projection_input,
                flags=flags,
            )

        if not math.isfinite(
            rate.posterior_sog_per_60
        ):
            flags.append(
                SOGQualityFlag.MISSING_PLAYER_RATE
            )
            return _no_projection(
                projection_input,
                flags=flags,
            )

        flags.extend(
            _detect_rate_quality(rate)
        )

        expected_component = (
            projected_minutes
            * rate.posterior_sog_per_60
            / 60.0
        )

        components.append(
            SOGProjectionComponent(
                game_state=game_state,
                projected_minutes=projected_minutes,
                sog_per_60=rate.posterior_sog_per_60,
                expected_sog=expected_component,
            )
        )

    base_expected_sog = sum(
        component.expected_sog
        for component in components
    )

    adjustment_keys: set[str] = set()
    applied_adjustments: list[
        SOGContextualAdjustment
    ] = []
    adjusted_expected_sog = base_expected_sog

    for adjustment in (
        projection_input.contextual_adjustments
    ):
        if adjustment.adjustment_key in adjustment_keys:
            flags.append(
                SOGQualityFlag.INVALID_CONTEXTUAL_ADJUSTMENT
            )
            return _no_projection(
                projection_input,
                flags=flags,
            )

        adjustment_keys.add(
            adjustment.adjustment_key
        )

        try:
            _validate_adjustment(adjustment)
        except ValueError:
            flags.append(
                SOGQualityFlag.INVALID_CONTEXTUAL_ADJUSTMENT
            )
            return _no_projection(
                projection_input,
                flags=flags,
            )

        adjusted_expected_sog *= (
            adjustment.applied_multiplier
        )
        applied_adjustments.append(adjustment)

    if not math.isfinite(adjusted_expected_sog):
        raise ValueError(
            "adjusted expected SOG must be finite"
        )

    quality_flags = _deduplicate_flags(flags)

    return SOGProjectionResult(
        canonical_event_id=projection_input.canonical_event_id,
        canonical_player_id=projection_input.canonical_player_id,
        model_run_timestamp=projection_input.model_run_timestamp,
        base_expected_sog=base_expected_sog,
        adjusted_expected_sog=adjusted_expected_sog,
        components=tuple(components),
        contextual_adjustments=tuple(
            applied_adjustments
        ),
        confidence_tier=_confidence_from_flags(
            quality_flags
        ),
        quality_flags=quality_flags,
    )