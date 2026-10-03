"""Exact like-for-like sportsbook offer matching."""

from __future__ import annotations

from .schemas import MatchResult, Offer

_FIELDS = (
    ("canonical_event_id", "event_mismatch"),
    ("canonical_subject_id", "subject_mismatch"),
    ("market_key", "market_mismatch"),
    ("side", "side_mismatch"),
    ("line", "line_mismatch"),
    ("settlement_rule_id", "settlement_rule_mismatch"),
)


def compare_offers(
    left: Offer,
    right: Offer,
) -> MatchResult:
    """Return exact comparability and structured mismatch reasons."""

    reasons = tuple(
        reason
        for field, reason in _FIELDS
        if getattr(left, field) != getattr(right, field)
    )

    return MatchResult(
        comparable=not reasons,
        reasons=reasons,
    )


def require_comparable(
    left: Offer,
    right: Offer,
) -> None:
    """Raise when two offers do not represent identical settlement exposure."""

    result = compare_offers(left, right)

    if not result.comparable:
        raise ValueError(
            f"Offers are not comparable: {', '.join(result.reasons)}",
        )