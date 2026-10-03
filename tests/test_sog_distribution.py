"""Synthetic tests for NHL SOG count distributions and settlement."""

import pytest
from scipy.stats import nbinom, poisson

from nhl_betting_models.common.schemas import MarketSide
from nhl_betting_models.nhl.schemas import (
    SOGDistributionConfig,
    SOGDistributionFamily,
)
from nhl_betting_models.nhl.sog_distribution import (
    build_distribution,
    count_probability,
    cumulative_probability,
    distribution_mean,
    negative_binomial_distribution,
    over_probabilities,
    settlement_probabilities,
    survival_probability,
    under_probabilities,
)


def synthetic_poisson_distribution():
    """SYNTHETIC TEST FIXTURE: no real NHL projection."""

    return build_distribution(
        3.25,
        SOGDistributionConfig(
            family=SOGDistributionFamily.POISSON,
        ),
    )


def synthetic_negative_binomial_distribution():
    """SYNTHETIC TEST FIXTURE: no fitted NHL dispersion."""

    return negative_binomial_distribution(
        4.2,
        dispersion=3.5,
    )


def test_poisson_probabilities_match_scipy() -> None:
    expected_sog = 3.25
    distribution = synthetic_poisson_distribution()

    assert count_probability(
        distribution,
        3,
    ) == pytest.approx(
        float(poisson.pmf(3, expected_sog))
    )
    assert cumulative_probability(
        distribution,
        2,
    ) == pytest.approx(
        float(poisson.cdf(2, expected_sog))
    )
    assert survival_probability(
        distribution,
        2,
    ) == pytest.approx(
        float(poisson.sf(2, expected_sog))
    )


def test_negative_binomial_probabilities_match_scipy() -> None:
    expected_sog = 4.2
    dispersion = 3.5
    probability = dispersion / (dispersion + expected_sog)
    distribution = synthetic_negative_binomial_distribution()

    assert count_probability(
        distribution,
        4,
    ) == pytest.approx(
        float(
            nbinom.pmf(
                4,
                dispersion,
                probability,
            )
        )
    )
    assert cumulative_probability(
        distribution,
        3,
    ) == pytest.approx(
        float(
            nbinom.cdf(
                3,
                dispersion,
                probability,
            )
        )
    )
    assert survival_probability(
        distribution,
        3,
    ) == pytest.approx(
        float(
            nbinom.sf(
                3,
                dispersion,
                probability,
            )
        )
    )


def test_negative_binomial_expected_sog_is_distribution_mean() -> None:
    expected_sog = 4.2
    dispersion = 3.5
    probability = dispersion / (dispersion + expected_sog)
    distribution = synthetic_negative_binomial_distribution()

    assert distribution_mean(distribution) == pytest.approx(
        expected_sog
    )
    assert float(
        nbinom.mean(
            dispersion,
            probability,
        )
    ) == pytest.approx(expected_sog)


def test_default_configured_family_is_negative_binomial() -> None:
    distribution = build_distribution(
        3.0,
        SOGDistributionConfig(
            dispersion=4.0,
        ),
    )

    assert (
        distribution.family
        is SOGDistributionFamily.NEGATIVE_BINOMIAL
    )
    assert distribution.expected_sog == pytest.approx(3.0)
    assert distribution.dispersion == pytest.approx(4.0)


def test_half_line_over_under_settlement_probabilities() -> None:
    expected_sog = 3.0
    distribution = build_distribution(
        expected_sog,
        SOGDistributionConfig(
            family=SOGDistributionFamily.POISSON,
        ),
    )

    over = over_probabilities(
        distribution,
        line=2.5,
    )
    under = under_probabilities(
        distribution,
        line=2.5,
    )

    expected_over = float(poisson.sf(2, expected_sog))
    expected_under = float(poisson.cdf(2, expected_sog))

    assert over.side is MarketSide.OVER
    assert over.win_probability == pytest.approx(expected_over)
    assert over.loss_probability == pytest.approx(expected_under)
    assert over.push_probability == 0.0

    assert under.side is MarketSide.UNDER
    assert under.win_probability == pytest.approx(expected_under)
    assert under.loss_probability == pytest.approx(expected_over)
    assert under.push_probability == 0.0


def test_whole_number_over_settlement_includes_push() -> None:
    expected_sog = 3.0
    distribution = build_distribution(
        expected_sog,
        SOGDistributionConfig(
            family=SOGDistributionFamily.POISSON,
        ),
    )

    probabilities = settlement_probabilities(
        distribution,
        side=MarketSide.OVER,
        line=3.0,
    )

    assert probabilities.win_probability == pytest.approx(
        float(poisson.sf(3, expected_sog))
    )
    assert probabilities.loss_probability == pytest.approx(
        float(poisson.cdf(2, expected_sog))
    )
    assert probabilities.push_probability == pytest.approx(
        float(poisson.pmf(3, expected_sog))
    )


def test_whole_number_under_settlement_includes_push() -> None:
    expected_sog = 3.0
    distribution = build_distribution(
        expected_sog,
        SOGDistributionConfig(
            family=SOGDistributionFamily.POISSON,
        ),
    )

    probabilities = settlement_probabilities(
        distribution,
        side=MarketSide.UNDER,
        line=3.0,
    )

    assert probabilities.win_probability == pytest.approx(
        float(poisson.cdf(2, expected_sog))
    )
    assert probabilities.loss_probability == pytest.approx(
        float(poisson.sf(3, expected_sog))
    )
    assert probabilities.push_probability == pytest.approx(
        float(poisson.pmf(3, expected_sog))
    )


def test_invalid_fractional_line_is_rejected() -> None:
    distribution = synthetic_poisson_distribution()

    with pytest.raises(
        ValueError,
        match="whole-number or half-number",
    ):
        settlement_probabilities(
            distribution,
            side=MarketSide.OVER,
            line=2.25,
        )