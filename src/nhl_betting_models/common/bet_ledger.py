"""Bet grading helpers.

Candidate creation, placement, and grading remain separate records.
"""

from __future__ import annotations

from .schemas import BetResult, GradedBet


def grade_bet(
    result: BetResult,
    stake: float,
    decimal_odds: float,
) -> GradedBet:
    """Grade one placed wager without mutating its placement record."""

    if stake <= 0:
        raise ValueError("Stake must be positive")

    if decimal_odds <= 1:
        raise ValueError(
            "Decimal odds must be greater than 1",
        )

    if result == BetResult.WIN:
        return GradedBet(
            result=result,
            profit=stake * (decimal_odds - 1.0),
            returned_stake=stake,
        )

    if result == BetResult.LOSS:
        return GradedBet(
            result=result,
            profit=-stake,
            returned_stake=0.0,
        )

    if result in {
        BetResult.PUSH,
        BetResult.VOID,
    }:
        return GradedBet(
            result=result,
            profit=0.0,
            returned_stake=stake,
        )

    return GradedBet(
        result=result,
        profit=None,
        returned_stake=None,
    )