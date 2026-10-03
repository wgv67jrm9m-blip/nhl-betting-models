"""Odds and probability conversion utilities."""

from __future__ import annotations

import math


def validate_american_odds(american_odds: int) -> None:
	"""Validate project-supported integer American odds.

	+100 is valid. -100, zero, and all values strictly between -100 and +100
	are rejected.
	"""

	if isinstance(american_odds, bool) or not isinstance(american_odds, int):
		raise TypeError("American odds must be an integer")

	if -100 <= american_odds < 100:
		raise ValueError(
			"American odds must be <= -101 or >= +100",
		)


def american_to_decimal(american_odds: int) -> float:
	"""Convert valid American odds to decimal odds."""

	validate_american_odds(american_odds)

	if american_odds > 0:
		return 1.0 + american_odds / 100.0

	return 1.0 + 100.0 / abs(american_odds)


def decimal_to_american(decimal_odds: float) -> int:
	"""Convert decimal odds to nearest integer American odds."""

	if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
		raise ValueError(
			"Decimal odds must be finite and greater than 1",
		)

	if decimal_odds >= 2.0:
		return int(round((decimal_odds - 1.0) * 100.0))

	return int(round(-100.0 / (decimal_odds - 1.0)))


def american_to_implied_probability(
	american_odds: int,
) -> float:
	"""Convert American odds to raw implied probability."""

	return 1.0 / american_to_decimal(american_odds)


def validate_open_probability(probability: float) -> None:
	"""Require a finite probability strictly inside the unit interval."""

	if (
		not math.isfinite(probability)
		or not 0.0 < probability < 1.0
	):
		raise ValueError(
			"Probability must be finite and strictly between 0 and 1",
		)


def probability_to_fair_decimal(
	probability: float,
) -> float:
	"""Convert probability to fair decimal odds.

	Probabilities equal to 0 or 1 are deliberately rejected because their
	corresponding fair-price representations are infinite or degenerate.
	"""

	validate_open_probability(probability)
	return 1.0 / probability


def probability_to_fair_american(
	probability: float,
) -> int:
	"""Convert probability to nearest integer fair American odds."""

	return decimal_to_american(
		probability_to_fair_decimal(probability),
	)
