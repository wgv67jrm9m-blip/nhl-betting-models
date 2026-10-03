"""Sportsbook evaluation bridge for NHL player SOG projections."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from nhl_betting_models.common.config import EligibilityThresholds
from nhl_betting_models.common.edge import evaluate_market
from nhl_betting_models.common.market_matching import compare_offers
from nhl_betting_models.common.no_vig import two_way_proportional
from nhl_betting_models.common.schemas import (
    MarketEvaluation,
    MarketSide,
    MarketStatus,
    Offer,
)
from nhl_betting_models.common.validation import reject_post_forecast_features
from nhl_betting_models.nhl.schemas import (
    ConfidenceTier,
    SOGDistributionResult,
    SOGEvaluationResult,
    SOGProjectionResult,
    SOGQualityFlag,
)
from nhl_betting_models.nhl.sog_distribution import settlement_probabilities

ALLOWED_SOG_MARKET_KEYS = frozenset(
    {
        "player_sog",
        "player_sog_alt",
    }
)

_AUTOMATIC_BLOCKING_FLAGS = frozenset(
    {
        "future_dated_input",
        "future_dated_odds",
        "missing_projected_toi",
        "unconfirmed_role",
        "unconfirmed_goalie_or_availability",
        "stale_odds",
        "small_sample",
        "prior_dominant_rate",
        "low_confidence_projection",
        "no_projection",
        "incomplete_market",
        "non_open_market",
        "market_identity_mismatch",
    }
)


def _deduplicate_strings(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Return strings in deterministic first-seen order."""

    return tuple(dict.fromkeys(values))


def _projection_quality_strings(
    projection: SOGProjectionResult,
) -> tuple[str, ...]:
    """Translate typed projection quality into evaluation-level flags."""

    flags: list[str] = [
        flag.value
        for flag in projection.quality_flags
    ]

    if SOGQualityFlag.POST_FORECAST_FEATURE in projection.quality_flags:
        flags.append("future_dated_input")

    if SOGQualityFlag.MISSING_PROJECTED_MINUTES in projection.quality_flags:
        flags.append("missing_projected_toi")

    if (
        SOGQualityFlag.PROJECTED_ROLE in projection.quality_flags
        or SOGQualityFlag.UNCERTAIN_ROLE in projection.quality_flags
    ):
        flags.append("unconfirmed_role")

    if SOGQualityFlag.MISSING_CONTEXT in projection.quality_flags:
        flags.append("unconfirmed_goalie_or_availability")

    if (
        SOGQualityFlag.INSUFFICIENT_PLAYER_EXPOSURE
        in projection.quality_flags
    ):
        flags.extend(
            (
                "small_sample",
                "prior_dominant_rate",
            )
        )

    if projection.confidence_tier is ConfidenceTier.LOW:
        flags.append("low_confidence_projection")

    if projection.confidence_tier is ConfidenceTier.NO_PROJECTION:
        flags.append("no_projection")

    return _deduplicate_strings(flags)


def _offer_quality_strings(
    offer: Offer,
    *,
    model_run_timestamp: datetime,
) -> tuple[str, ...]:
    """Collect timestamp, freshness, status, and source quality flags."""

    flags: list[str] = list(offer.quality_flags)

    if offer.is_stale:
        flags.append("stale_odds")

    if offer.status is not MarketStatus.OPEN:
        flags.append("non_open_market")

    timestamps = [
        offer.retrieval_timestamp,
        offer.as_of_timestamp,
    ]

    if offer.source_timestamp is not None:
        timestamps.append(offer.source_timestamp)

    try:
        reject_post_forecast_features(
            timestamps,
            model_run_timestamp,
        )
    except ValueError as exc:
        if "newer than model-run timestamp" in str(exc):
            flags.append("future_dated_odds")
        else:
            raise

    return _deduplicate_strings(flags)


def _same_market_except_side(
    left: Offer,
    right: Offer,
) -> bool:
    """Apply shared matching semantics while allowing opposite sides."""

    aligned_right = right.model_copy(
        update={"side": left.side}
    )

    return compare_offers(
        left,
        aligned_right,
    ).comparable


def _find_opposite_offer(
    target: Offer,
    offers: Sequence[Offer],
) -> Offer | None:
    """Find the exact opposite side from the same sportsbook snapshot."""

    opposite_side = (
        MarketSide.UNDER
        if target.side is MarketSide.OVER
        else MarketSide.OVER
    )

    matches = [
        candidate
        for candidate in offers
        if candidate.record_id != target.record_id
        and candidate.side is opposite_side
        and candidate.sportsbook == target.sportsbook
        and candidate.retrieval_timestamp == target.retrieval_timestamp
        and _same_market_except_side(
            target,
            candidate,
        )
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            "multiple exact opposite-side offers found for sportsbook snapshot"
        )

    return matches[0]


def _validate_target_offer(
    offer: Offer,
    projection: SOGProjectionResult,
) -> None:
    """Require an offer to reference the projected event and player."""

    if offer.market_key not in ALLOWED_SOG_MARKET_KEYS:
        raise ValueError("offer is not a supported player SOG market")

    if offer.side not in {
        MarketSide.OVER,
        MarketSide.UNDER,
    }:
        raise ValueError("SOG offer side must be over or under")

    if offer.canonical_event_id != projection.canonical_event_id:
        raise ValueError(
            "offer event does not match SOG projection event"
        )

    if offer.canonical_subject_id != projection.canonical_player_id:
        raise ValueError(
            "offer subject does not match SOG projection player"
        )


def _force_ineligible(
    evaluation: MarketEvaluation,
    *,
    failure: str,
    input_quality_flags: tuple[str, ...],
) -> MarketEvaluation:
    """Return a shared evaluation with an explicit eligibility block."""

    failures = _deduplicate_strings(
        (
            *evaluation.threshold_failures,
            failure,
        )
    )

    return evaluation.model_copy(
        update={
            "eligible": False,
            "threshold_failures": failures,
            "input_quality_flags": input_quality_flags,
        }
    )


def _apply_automatic_eligibility_policy(
    evaluation: MarketEvaluation,
    *,
    projection: SOGProjectionResult,
    quality_flags: tuple[str, ...],
    automatic_eligibility: bool,
) -> MarketEvaluation:
    """Apply SOG-specific safety gates without changing shared interfaces."""

    if not automatic_eligibility:
        return _force_ineligible(
            evaluation,
            failure="automatic_eligibility_disabled",
            input_quality_flags=quality_flags,
        )

    if set(quality_flags) & _AUTOMATIC_BLOCKING_FLAGS:
        return _force_ineligible(
            evaluation,
            failure="sog_quality_gate_failed",
            input_quality_flags=quality_flags,
        )

    if projection.confidence_tier in {
        ConfidenceTier.LOW,
        ConfidenceTier.NO_PROJECTION,
    }:
        return _force_ineligible(
            evaluation,
            failure="projection_confidence_too_low",
            input_quality_flags=quality_flags,
        )

    return evaluation.model_copy(
        update={
            "input_quality_flags": quality_flags,
        }
    )


def evaluate_sog_offer(
    *,
    offer: Offer,
    market_offers: Sequence[Offer],
    projection: SOGProjectionResult,
    distribution: SOGDistributionResult,
    thresholds: EligibilityThresholds | None = None,
    automatic_eligibility: bool = False,
) -> tuple[SOGEvaluationResult | None, tuple[str, ...]]:
    """Evaluate one sportsbook SOG offer independently."""

    _validate_target_offer(
        offer,
        projection,
    )

    projection_flags = _projection_quality_strings(
        projection
    )
    offer_flags = _offer_quality_strings(
        offer,
        model_run_timestamp=projection.model_run_timestamp,
    )

    if projection.confidence_tier is ConfidenceTier.NO_PROJECTION:
        return (
            None,
            _deduplicate_strings(
                (
                    *projection_flags,
                    *offer_flags,
                    "no_projection",
                )
            ),
        )

    counterpart = _find_opposite_offer(
        offer,
        market_offers,
    )

    if counterpart is None:
        return (
            None,
            _deduplicate_strings(
                (
                    *projection_flags,
                    *offer_flags,
                    "incomplete_market",
                )
            ),
        )

    counterpart_flags = _offer_quality_strings(
        counterpart,
        model_run_timestamp=projection.model_run_timestamp,
    )

    quality_flags = _deduplicate_strings(
        (
            *projection_flags,
            *offer_flags,
            *counterpart_flags,
        )
    )

    market_blockers = {
        "stale_odds",
        "future_dated_odds",
        "non_open_market",
    }

    if set(quality_flags) & market_blockers:
        return None, quality_flags

    if offer.quality_flags or counterpart.quality_flags:
        return None, quality_flags

    no_vig = two_way_proportional(
        (
            offer,
            counterpart,
        )
    )

    no_vig_probability = no_vig.no_vig_probabilities[
        offer.side.value
    ]

    probabilities = settlement_probabilities(
        distribution,
        side=offer.side,
        line=offer.line,
    )

    shared_evaluation = evaluate_market(
        model_probability=probabilities.win_probability,
        no_vig_market_probability=no_vig_probability,
        offered_american_odds=offer.american_odds,
        offered_decimal_odds=offer.decimal_odds,
        thresholds=(
            EligibilityThresholds()
            if thresholds is None
            else thresholds
        ),
        input_quality_flags=quality_flags,
        loss_probability=probabilities.loss_probability,
        push_probability=probabilities.push_probability,
    )

    final_evaluation = _apply_automatic_eligibility_policy(
        shared_evaluation,
        projection=projection,
        quality_flags=quality_flags,
        automatic_eligibility=automatic_eligibility,
    )

    return (
        SOGEvaluationResult(
            canonical_event_id=offer.canonical_event_id,
            canonical_player_id=offer.canonical_subject_id,
            market_key=offer.market_key,
            side=offer.side,
            line=offer.line,
            probabilities=probabilities,
            market_evaluation=final_evaluation,
            confidence_tier=projection.confidence_tier,
            quality_flags=projection.quality_flags,
        ),
        quality_flags,
    )


def evaluate_sog_offers(
    *,
    offers: Sequence[Offer],
    projection: SOGProjectionResult,
    distribution: SOGDistributionResult,
    thresholds: EligibilityThresholds | None = None,
    automatic_eligibility: bool = False,
) -> tuple[
    tuple[SOGEvaluationResult, ...],
    dict[str, tuple[str, ...]],
]:
    """Evaluate every sportsbook offer independently."""

    evaluations: list[SOGEvaluationResult] = []
    blocked: dict[str, tuple[str, ...]] = {}

    for offer in offers:
        result, quality_flags = evaluate_sog_offer(
            offer=offer,
            market_offers=offers,
            projection=projection,
            distribution=distribution,
            thresholds=thresholds,
            automatic_eligibility=automatic_eligibility,
        )

        if result is None:
            blocked[offer.record_id] = quality_flags
            continue

        evaluations.append(result)

    return tuple(evaluations), blocked