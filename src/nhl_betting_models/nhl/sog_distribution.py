"""Count distributions and settlement probabilities for NHL player SOG."""

from __future__ import annotations

from math import floor, isfinite

from scipy.stats import nbinom, poisson

from nhl_betting_models.common.schemas import MarketSide
from nhl_betting_models.nhl.schemas import (
    SOGDistributionConfig,
    SOGDistributionFamily,
    SOGDistributionResult,
    SOGSideProbabilities,
)


def _validate_expected_sog(expected_sog: float) -> None:
    """Validate a count-distribution mean."""

    if not isfinite(expected_sog):
        raise ValueError("expected_sog must be finite")
    if expected_sog < 0.0:
        raise ValueError("expected_sog must be non-negative")


def _validate_line(line: float) -> None:
    """Validate a SOG settlement line."""

    if not isfinite(line):
        raise ValueError("line must be finite")
    if line < 0.0:
        raise ValueError("line must be non-negative")

    fractional_part = line - floor(line)
    if not (
        abs(fractional_part) <= 1e-12
        or abs(fractional_part - 0.5) <= 1e-12
    ):
        raise ValueError("SOG line must be a whole-number or half-number line")


def build_distribution(
    expected_sog: float,
    config: SOGDistributionConfig,
) -> SOGDistributionResult:
    """Build a SOG distribution from explicit configuration.

    Negative binomial remains the default configured family in
    ``SOGDistributionConfig``. Its dispersion must be supplied explicitly
    because the project has not established a production dispersion prior.
    """

    _validate_expected_sog(expected_sog)

    if config.family is SOGDistributionFamily.POISSON:
        return poisson_distribution(expected_sog)

    dispersion = config.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    return negative_binomial_distribution(
        expected_sog,
        dispersion=dispersion,
    )


def poisson_distribution(expected_sog: float) -> SOGDistributionResult:
    """Create an explicit Poisson baseline with mean equal to expected SOG."""

    _validate_expected_sog(expected_sog)

    return SOGDistributionResult(
        expected_sog=expected_sog,
        family=SOGDistributionFamily.POISSON,
        dispersion=None,
    )


def negative_binomial_distribution(
    expected_sog: float,
    *,
    dispersion: float,
) -> SOGDistributionResult:
    """Create a negative-binomial SOG distribution.

    ``dispersion`` is SciPy's ``n`` parameter. The corresponding probability
    parameter is:

        p = dispersion / (dispersion + expected_sog)

    which gives:

        mean = n * (1 - p) / p = expected_sog

    and:

        variance = expected_sog + expected_sog**2 / dispersion
    """

    _validate_expected_sog(expected_sog)

    if not isfinite(dispersion):
        raise ValueError("dispersion must be finite")
    if dispersion <= 0.0:
        raise ValueError("dispersion must be positive")

    return SOGDistributionResult(
        expected_sog=expected_sog,
        family=SOGDistributionFamily.NEGATIVE_BINOMIAL,
        dispersion=dispersion,
    )


def distribution_mean(distribution: SOGDistributionResult) -> float:
    """Return the mathematical mean of a configured SOG distribution."""

    if distribution.family is SOGDistributionFamily.POISSON:
        return distribution.expected_sog

    dispersion = distribution.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    if distribution.expected_sog == 0.0:
        return 0.0

    probability = dispersion / (
        dispersion + distribution.expected_sog
    )
    return dispersion * (1.0 - probability) / probability


def _negative_binomial_probability(
    distribution: SOGDistributionResult,
) -> float:
    """Return SciPy's negative-binomial probability parameter."""

    dispersion = distribution.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    if distribution.expected_sog == 0.0:
        return 1.0

    return dispersion / (
        dispersion + distribution.expected_sog
    )


def count_probability(
    distribution: SOGDistributionResult,
    count: int,
) -> float:
    """Return P(SOG = count)."""

    if count < 0:
        return 0.0

    mean = distribution.expected_sog

    if distribution.family is SOGDistributionFamily.POISSON:
        return float(poisson.pmf(count, mean))

    dispersion = distribution.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    if mean == 0.0:
        return 1.0 if count == 0 else 0.0

    probability = _negative_binomial_probability(distribution)
    return float(
        nbinom.pmf(
            count,
            dispersion,
            probability,
        )
    )


def cumulative_probability(
    distribution: SOGDistributionResult,
    maximum_count: int,
) -> float:
    """Return P(SOG <= maximum_count)."""

    if maximum_count < 0:
        return 0.0

    mean = distribution.expected_sog

    if distribution.family is SOGDistributionFamily.POISSON:
        return float(poisson.cdf(maximum_count, mean))

    dispersion = distribution.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    if mean == 0.0:
        return 1.0

    probability = _negative_binomial_probability(distribution)
    return float(
        nbinom.cdf(
            maximum_count,
            dispersion,
            probability,
        )
    )


def survival_probability(
    distribution: SOGDistributionResult,
    minimum_exclusive_count: int,
) -> float:
    """Return P(SOG > minimum_exclusive_count)."""

    if minimum_exclusive_count < 0:
        return 1.0

    mean = distribution.expected_sog

    if distribution.family is SOGDistributionFamily.POISSON:
        return float(
            poisson.sf(
                minimum_exclusive_count,
                mean,
            )
        )

    dispersion = distribution.dispersion
    if dispersion is None:
        raise ValueError("negative-binomial distribution requires dispersion")

    if mean == 0.0:
        return 0.0

    probability = _negative_binomial_probability(distribution)
    return float(
        nbinom.sf(
            minimum_exclusive_count,
            dispersion,
            probability,
        )
    )


def settlement_probabilities(
    distribution: SOGDistributionResult,
    *,
    side: MarketSide,
    line: float,
) -> SOGSideProbabilities:
    """Calculate exact win, loss, and push probabilities for a SOG line."""

    _validate_line(line)

    if side not in {
        MarketSide.OVER,
        MarketSide.UNDER,
    }:
        raise ValueError("SOG settlement side must be over or under")

    line_floor = floor(line)
    is_whole_number = abs(line - line_floor) <= 1e-12

    if is_whole_number:
        push_probability = count_probability(
            distribution,
            line_floor,
        )

        if side is MarketSide.OVER:
            win_probability = survival_probability(
                distribution,
                line_floor,
            )
            loss_probability = cumulative_probability(
                distribution,
                line_floor - 1,
            )
        else:
            win_probability = cumulative_probability(
                distribution,
                line_floor - 1,
            )
            loss_probability = survival_probability(
                distribution,
                line_floor,
            )
    else:
        push_probability = 0.0

        if side is MarketSide.OVER:
            win_probability = survival_probability(
                distribution,
                line_floor,
            )
            loss_probability = cumulative_probability(
                distribution,
                line_floor,
            )
        else:
            win_probability = cumulative_probability(
                distribution,
                line_floor,
            )
            loss_probability = survival_probability(
                distribution,
                line_floor,
            )

    total_probability = (
        win_probability
        + loss_probability
        + push_probability
    )

    if total_probability <= 0.0:
        raise ValueError("settlement probabilities have invalid total mass")

    win_probability /= total_probability
    loss_probability /= total_probability
    push_probability /= total_probability

    return SOGSideProbabilities(
        side=side,
        line=line,
        win_probability=win_probability,
        loss_probability=loss_probability,
        push_probability=push_probability,
    )


def over_probabilities(
    distribution: SOGDistributionResult,
    *,
    line: float,
) -> SOGSideProbabilities:
    """Calculate Over settlement probabilities."""

    return settlement_probabilities(
        distribution,
        side=MarketSide.OVER,
        line=line,
    )


def under_probabilities(
    distribution: SOGDistributionResult,
    *,
    line: float,
) -> SOGSideProbabilities:
    """Calculate Under settlement probabilities."""

    return settlement_probabilities(
        distribution,
        side=MarketSide.UNDER,
        line=line,
    )