"""Structured data-quality validation for provider-neutral NHL data."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from nhl_betting_models.common.schemas import MarketSide, Offer
from nhl_betting_models.data.base import NormalizedGame
from nhl_betting_models.data.canonical_ids import (
    IdentityResolution,
    IdentityResolutionStatus,
)


class DataQualitySeverity(StrEnum):
    """Severity attached to one structured data-quality issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DataQualityFlag(StrEnum):
    """Provider-neutral ingestion and normalization quality flags."""

    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    UNRESOLVED_IDENTITY = "unresolved_identity"
    MISSING_TIMESTAMP = "missing_timestamp"
    STALE_ODDS = "stale_odds"
    INVALID_ODDS = "invalid_odds"
    INCOMPLETE_TWO_SIDED_MARKET = "incomplete_two_sided_market"
    MISMATCHED_GAME_START_TIME = "mismatched_game_start_time"
    DUPLICATE_OFFER = "duplicate_offer"
    MISMATCHED_PLAYER_IDENTITY = "mismatched_player_identity"
    MISSING_SLATE_GAME = "missing_slate_game"
    MISSING_SLATE_PLAYER = "missing_slate_player"

class DataQualityIssue(BaseModel):
    """One immutable structured data-quality finding."""

    model_config = ConfigDict(frozen=True)

    flag: DataQualityFlag
    severity: DataQualitySeverity
    message: str
    related_record_ids: tuple[str, ...] = ()


class DataQualityResult(BaseModel):
    """Immutable collection of data-quality findings."""

    model_config = ConfigDict(frozen=True)

    issues: tuple[DataQualityIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether no error-severity findings exist."""

        return not any(
            issue.severity is DataQualitySeverity.ERROR
            for issue in self.issues
        )

    @property
    def flags(self) -> tuple[DataQualityFlag, ...]:
        """Return flags in deterministic first-seen order."""

        return tuple(dict.fromkeys(issue.flag for issue in self.issues))


def combine_quality_results(
    *results: DataQualityResult,
) -> DataQualityResult:
    """Combine quality results while preserving issue order."""

    return DataQualityResult(
        issues=tuple(
            issue
            for result in results
            for issue in result.issues
        )
    )


def validate_identity_resolution(
    resolution: IdentityResolution,
) -> DataQualityResult:
    """Convert unresolved identity states to structured quality findings."""

    if resolution.status is IdentityResolutionStatus.RESOLVED:
        return DataQualityResult()

    if resolution.status is IdentityResolutionStatus.AMBIGUOUS:
        return DataQualityResult(
            issues=(
                DataQualityIssue(
                    flag=DataQualityFlag.AMBIGUOUS_IDENTITY,
                    severity=DataQualitySeverity.ERROR,
                    message=(
                        "provider identity maps to multiple canonical IDs"
                    ),
                ),
            )
        )

    return DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.UNRESOLVED_IDENTITY,
                severity=DataQualitySeverity.ERROR,
                message="provider identity has no canonical mapping",
            ),
        )
    )


def validate_offer_timestamps(
    offer: Offer,
) -> DataQualityResult:
    """Require timestamps needed for point-in-time offer evaluation."""

    missing: list[str] = []

    if offer.retrieval_timestamp is None:
        missing.append("retrieval_timestamp")
    if offer.as_of_timestamp is None:
        missing.append("as_of_timestamp")

    if not missing:
        return DataQualityResult()

    return DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.MISSING_TIMESTAMP,
                severity=DataQualitySeverity.ERROR,
                message=(
                    "offer is missing required timestamps: "
                    + ", ".join(missing)
                ),
                related_record_ids=(offer.record_id,),
            ),
        )
    )


def validate_stale_offer(offer: Offer) -> DataQualityResult:
    """Surface the shared Offer stale marker as structured quality."""

    if not offer.is_stale:
        return DataQualityResult()

    return DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.STALE_ODDS,
                severity=DataQualitySeverity.ERROR,
                message="sportsbook offer is marked stale",
                related_record_ids=(offer.record_id,),
            ),
        )
    )


def validate_offer_odds(offer: Offer) -> DataQualityResult:
    """Validate basic American and decimal price invariants."""

    valid_american = (
        offer.american_odds >= 100
        or offer.american_odds <= -100
    )
    valid_decimal = (
        math.isfinite(offer.decimal_odds)
        and offer.decimal_odds > 1.0
    )

    if valid_american and valid_decimal:
        return DataQualityResult()

    return DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.INVALID_ODDS,
                severity=DataQualitySeverity.ERROR,
                message="offer contains invalid American or decimal odds",
                related_record_ids=(offer.record_id,),
            ),
        )
    )


def _market_identity_without_side(
    offer: Offer,
) -> tuple[object, ...]:
    """Build exact two-sided market identity excluding side and price."""

    return (
        offer.sportsbook,
        offer.canonical_event_id,
        offer.canonical_subject_id,
        offer.market_key,
        offer.line,
        offer.settlement_rule_id,
        offer.retrieval_timestamp,
    )


def validate_two_sided_market(
    offers: Sequence[Offer],
) -> DataQualityResult:
    """Require one Over and one Under for each exact SOG snapshot."""

    grouped: dict[tuple[object, ...], list[Offer]] = {}

    for offer in offers:
        grouped.setdefault(
            _market_identity_without_side(offer),
            [],
        ).append(offer)

    issues: list[DataQualityIssue] = []

    for market_offers in grouped.values():
        sides = {offer.side for offer in market_offers}

        if sides != {MarketSide.OVER, MarketSide.UNDER}:
            issues.append(
                DataQualityIssue(
                    flag=DataQualityFlag.INCOMPLETE_TWO_SIDED_MARKET,
                    severity=DataQualitySeverity.ERROR,
                    message=(
                        "sportsbook SOG snapshot does not contain "
                        "the required Over and Under sides"
                    ),
                    related_record_ids=tuple(
                        offer.record_id
                        for offer in market_offers
                    ),
                )
            )

    return DataQualityResult(issues=tuple(issues))


def _offer_duplicate_key(
    offer: Offer,
) -> tuple[object, ...]:
    """Build immutable identity for duplicate-offer detection."""

    return (
        offer.source_name,
        offer.source_record_id,
        offer.sportsbook,
        offer.canonical_event_id,
        offer.canonical_subject_id,
        offer.market_key,
        offer.side,
        offer.line,
        offer.settlement_rule_id,
        offer.american_odds,
        offer.decimal_odds,
        offer.retrieval_timestamp,
        offer.source_timestamp,
        offer.as_of_timestamp,
        offer.status,
    )


def validate_duplicate_offers(
    offers: Sequence[Offer],
) -> DataQualityResult:
    """Detect repeated normalized representations of one offer snapshot."""

    keys = [_offer_duplicate_key(offer) for offer in offers]
    counts = Counter(keys)
    issues: list[DataQualityIssue] = []

    for key, count in counts.items():
        if count <= 1:
            continue

        duplicates = [
            offer
            for offer in offers
            if _offer_duplicate_key(offer) == key
        ]
        issues.append(
            DataQualityIssue(
                flag=DataQualityFlag.DUPLICATE_OFFER,
                severity=DataQualitySeverity.ERROR,
                message="duplicate normalized sportsbook offers detected",
                related_record_ids=tuple(
                    offer.record_id
                    for offer in duplicates
                ),
            )
        )

    return DataQualityResult(issues=tuple(issues))


def validate_game_start_times(
    games: Sequence[NormalizedGame],
) -> DataQualityResult:
    """Detect conflicting start times for one provider game identity."""

    indexed: dict[
        tuple[str, str],
        dict[datetime, list[NormalizedGame]],
    ] = {}

    for game in games:
        key = (
            game.provenance.source_name,
            game.source_game_id,
        )
        indexed.setdefault(key, {}).setdefault(
            game.game_start_timestamp,
            [],
        ).append(game)

    issues: list[DataQualityIssue] = []

    for (_, source_game_id), times in indexed.items():
        if len(times) <= 1:
            continue

        raw_ids = tuple(
            game.provenance.raw_payload_record_id
            for games_at_time in times.values()
            for game in games_at_time
        )
        issues.append(
            DataQualityIssue(
                flag=DataQualityFlag.MISMATCHED_GAME_START_TIME,
                severity=DataQualitySeverity.ERROR,
                message=(
                    "provider game identity has conflicting game start times: "
                    f"{source_game_id}"
                ),
                related_record_ids=raw_ids,
            )
        )

    return DataQualityResult(issues=tuple(issues))


def validate_player_identity(
    *,
    expected_canonical_player_id: str,
    offers: Iterable[Offer],
) -> DataQualityResult:
    """Require supplied offers to reference the expected player."""

    mismatches = tuple(
        offer.record_id
        for offer in offers
        if offer.canonical_subject_id != expected_canonical_player_id
    )

    if not mismatches:
        return DataQualityResult()

    return DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.MISMATCHED_PLAYER_IDENTITY,
                severity=DataQualitySeverity.ERROR,
                message=(
                    "sportsbook offer player identity does not match "
                    "the expected canonical player"
                ),
                related_record_ids=mismatches,
            ),
        )
    )


def validate_offer_set(
    offers: Sequence[Offer],
    *,
    expected_canonical_player_id: str | None = None,
) -> DataQualityResult:
    """Run provider-neutral sportsbook data-quality checks."""

    results: list[DataQualityResult] = []

    for offer in offers:
        results.extend(
            (
                validate_offer_timestamps(offer),
                validate_stale_offer(offer),
                validate_offer_odds(offer),
            )
        )

    results.extend(
        (
            validate_duplicate_offers(offers),
            validate_two_sided_market(offers),
        )
    )

    if expected_canonical_player_id is not None:
        results.append(
            validate_player_identity(
                expected_canonical_player_id=(
                    expected_canonical_player_id
                ),
                offers=offers,
            )
        )

    return combine_quality_results(*results)