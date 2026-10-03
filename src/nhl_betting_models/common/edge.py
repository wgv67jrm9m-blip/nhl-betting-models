"""Expected-value, edge, and eligibility calculations."""

from __future__ import annotations

import math

from .config import EligibilityThresholds
from .odds import (
    probability_to_fair_american,
    probability_to_fair_decimal,
    validate_open_probability,
)
from .schemas import MarketEvaluation


def expected_value_no_push(
    win_probability: float,
    decimal_odds: float,
) -> float:
    """Calculate EV per unit staked when no push outcome is possible."""

    validate_open_probability(win_probability)

    return expected_value(
        win_probability=win_probability,
        loss_probability=1.0 - win_probability,
        push_probability=0.0,
        decimal_odds=decimal_odds,
    )


def expected_value(
    win_probability: float,
    loss_probability: float,
    push_probability: float,
    decimal_odds: float,
) -> float:
    """Calculate EV per unit staked for win/loss/push probabilities."""

    probabilities = (
        win_probability,
        loss_probability,
        push_probability,
    )

    if any(
        not math.isfinite(probability)
        or probability < 0.0
        or probability > 1.0
        for probability in probabilities
    ):
        raise ValueError(
            "Outcome probabilities must be finite and within [0, 1]",
        )

    if not math.isclose(
        sum(probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Win, loss, and push probabilities must sum to 1",
        )

    if (
        not math.isfinite(decimal_odds)
        or decimal_odds <= 1.0
    ):
        raise ValueError(
            "Decimal odds must be finite and greater than 1",
        )

    return (
        win_probability * (decimal_odds - 1.0)
        - loss_probability
    )


def diagnostic_edge(
    model_probability: float,
    no_vig_market_probability: float,
) -> float:
    """Return model probability minus market no-vig probability."""

    validate_open_probability(model_probability)
    validate_open_probability(no_vig_market_probability)

    return (
        model_probability
        - no_vig_market_probability
    )


def evaluate_market(
    *,
    model_probability: float,
    no_vig_market_probability: float,
    offered_american_odds: int,
    offered_decimal_odds: float,
    thresholds: EligibilityThresholds,
    input_quality_flags: tuple[str, ...] = (),
    loss_probability: float | None = None,
    push_probability: float = 0.0,
) -> MarketEvaluation:
    """Evaluate price and configurable eligibility gates.

    Positive EV does not independently imply eligibility.
    """

    validate_open_probability(model_probability)
    validate_open_probability(no_vig_market_probability)

    loss = (
        1.0 - model_probability
        if loss_probability is None
        else loss_probability
    )

    ev = expected_value(
        win_probability=model_probability,
        loss_probability=loss,
        push_probability=push_probability,
        decimal_odds=offered_decimal_odds,
    )

    edge = diagnostic_edge(
        model_probability,
        no_vig_market_probability,
    )

    failures: list[str] = []

    if ev < thresholds.min_ev:
        failures.append("ev_below_threshold")

    if edge < thresholds.min_edge:
        failures.append("edge_below_threshold")

    if (
        model_probability
        < thresholds.min_model_probability
    ):
        failures.append(
            "model_probability_below_threshold",
        )

    critical_flags = sorted(
        set(input_quality_flags)
        & thresholds.critical_quality_flags
    )

    failures.extend(
        f"critical_quality:{flag}"
        for flag in critical_flags
    )

    return MarketEvaluation(
        model_probability=model_probability,
        no_vig_market_probability=no_vig_market_probability,
        fair_decimal_odds=probability_to_fair_decimal(
            model_probability,
        ),
        fair_american_odds=probability_to_fair_american(
            model_probability,
        ),
        offered_american_odds=offered_american_odds,
        offered_decimal_odds=offered_decimal_odds,
        expected_value=ev,
        edge=edge,
        eligible=not failures,
        threshold_failures=tuple(failures),
        input_quality_flags=input_quality_flags,
    )