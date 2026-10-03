"""Synthetic tests for bet settlement grading."""

import pytest

from nhl_betting_models.common.bet_ledger import (
    grade_bet,
)
from nhl_betting_models.common.schemas import (
    BetResult,
)


@pytest.mark.parametrize(
    ("result", "profit", "returned"),
    [
        (
            BetResult.WIN,
            10.0,
            10.0,
        ),
        (
            BetResult.LOSS,
            -10.0,
            0.0,
        ),
        (
            BetResult.PUSH,
            0.0,
            10.0,
        ),
        (
            BetResult.VOID,
            0.0,
            10.0,
        ),
        (
            BetResult.PENDING,
            None,
            None,
        ),
    ],
)
def test_bet_grading(
    result: BetResult,
    profit: float | None,
    returned: float | None,
) -> None:
    graded = grade_bet(
        result,
        10.0,
        2.0,
    )

    assert graded.profit == profit
    assert graded.returned_stake == returned