"""Vig-removal methods with strict market validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .odds import american_to_implied_probability
from .schemas import MarketStatus, NoVigResult, Offer


class NoVigMethod(ABC):
    """Extension point for proportional, Shin, power, and future methods."""

    name: str

    @abstractmethod
    def calculate(
        self,
        offers: Sequence[Offer],
        required_sides: set[str],
    ) -> NoVigResult:
        """Calculate no-vig probabilities for a complete market."""

        raise NotImplementedError


class ProportionalNoVig(NoVigMethod):
    """Proportional normalization of raw implied probabilities."""

    name = "proportional"

    def calculate(
        self,
        offers: Sequence[Offer],
        required_sides: set[str],
    ) -> NoVigResult:
        _validate_market(offers, required_sides)

        raw = {
            offer.side.value: american_to_implied_probability(
                offer.american_odds,
            )
            for offer in offers
        }
        total = sum(raw.values())

        normalized = {
            side: probability / total
            for side, probability in raw.items()
        }

        return NoVigResult(
            method=self.name,
            raw_implied_probabilities=raw,
            no_vig_probabilities=normalized,
            overround=total - 1.0,
        )


class ShinNoVig(NoVigMethod):
    """Reserved extension interface for a future Shin implementation."""

    name = "shin"

    def calculate(
        self,
        offers: Sequence[Offer],
        required_sides: set[str],
    ) -> NoVigResult:
        raise NotImplementedError(
            "Shin no-vig is reserved for a future validated implementation",
        )


class PowerNoVig(NoVigMethod):
    """Reserved extension interface for a future power implementation."""

    name = "power"

    def calculate(
        self,
        offers: Sequence[Offer],
        required_sides: set[str],
    ) -> NoVigResult:
        raise NotImplementedError(
            "Power no-vig is reserved for a future validated implementation",
        )


def _validate_market(
    offers: Sequence[Offer],
    required_sides: set[str],
) -> None:
    if not offers:
        raise ValueError("Market must contain offers")

    if len(offers) != len(required_sides):
        raise ValueError(
            "Incomplete market: offer count does not match required sides",
        )

    first = offers[0]

    identity = (
        first.canonical_event_id,
        first.canonical_subject_id,
        first.market_key,
        first.line,
        first.settlement_rule_id,
        first.sportsbook,
        first.retrieval_timestamp,
    )

    actual_sides: set[str] = set()

    for offer in offers:
        if offer.status != MarketStatus.OPEN:
            raise ValueError(
                "Market contains a non-open offer",
            )

        if offer.is_stale:
            raise ValueError(
                "Market contains stale odds",
            )

        if offer.quality_flags:
            raise ValueError(
                "Market contains offer quality flags",
            )

        current_identity = (
            offer.canonical_event_id,
            offer.canonical_subject_id,
            offer.market_key,
            offer.line,
            offer.settlement_rule_id,
            offer.sportsbook,
            offer.retrieval_timestamp,
        )

        if current_identity != identity:
            raise ValueError(
                "Market offers are mismatched",
            )

        if offer.side.value in actual_sides:
            raise ValueError(
                "Market contains duplicate sides",
            )

        actual_sides.add(offer.side.value)

    if actual_sides != required_sides:
        raise ValueError(
            "Incomplete market: required sides are missing or unexpected",
        )


def two_way_proportional(
    offers: Sequence[Offer],
) -> NoVigResult:
    """Normalize a complete two-way Over/Under market."""

    return ProportionalNoVig().calculate(
        offers,
        {"over", "under"},
    )


def multiway_proportional(
    offers: Sequence[Offer],
    required_sides: set[str],
) -> NoVigResult:
    """Normalize an explicitly defined complete multiway market."""

    if len(required_sides) < 2:
        raise ValueError(
            "Multiway market requires at least two sides",
        )

    return ProportionalNoVig().calculate(
        offers,
        required_sides,
    )