"""Typed shared schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MarketSide(StrEnum):
    """Canonical sportsbook market sides."""

    OVER = "over"
    UNDER = "under"
    HOME = "home"
    AWAY = "away"
    YES = "yes"
    NO = "no"
    DRAW = "draw"


class MarketStatus(StrEnum):
    """Canonical sportsbook market statuses."""

    OPEN = "open"
    SUSPENDED = "suspended"
    UNAVAILABLE = "unavailable"
    SETTLED = "settled"
    VOIDED = "voided"
    UNKNOWN = "unknown"


class BetResult(StrEnum):
    """Supported wager settlement states."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"
    PENDING = "pending"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class Offer(BaseModel):
    """Immutable sportsbook offer snapshot used by shared pricing logic."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    source_name: str
    sportsbook: str
    source_record_id: str | None = None
    canonical_event_id: str
    canonical_subject_id: str
    market_key: str
    side: MarketSide
    line: float
    settlement_rule_id: str
    american_odds: int
    decimal_odds: float
    retrieval_timestamp: datetime
    source_timestamp: datetime | None = None
    as_of_timestamp: datetime
    status: MarketStatus = MarketStatus.OPEN
    is_stale: bool = False
    quality_flags: tuple[str, ...] = ()
    data_hash: str | None = None

    @field_validator(
        "retrieval_timestamp",
        "source_timestamp",
        "as_of_timestamp",
    )
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return None if value is None else _utc(value)


class NoVigResult(BaseModel):
    """Result of a no-vig normalization."""

    model_config = ConfigDict(frozen=True)

    method: str
    raw_implied_probabilities: dict[str, float]
    no_vig_probabilities: dict[str, float]
    overround: float


class MatchResult(BaseModel):
    """Structured exact-market comparison result."""

    model_config = ConfigDict(frozen=True)

    comparable: bool
    reasons: tuple[str, ...] = ()


class MarketEvaluation(BaseModel):
    """Structured price/edge/eligibility evaluation."""

    model_config = ConfigDict(frozen=True)

    model_probability: float
    no_vig_market_probability: float
    fair_decimal_odds: float
    fair_american_odds: int
    offered_american_odds: int
    offered_decimal_odds: float
    expected_value: float
    edge: float
    eligible: bool
    threshold_failures: tuple[str, ...]
    input_quality_flags: tuple[str, ...]


class GradedBet(BaseModel):
    """Financial result of a graded wager."""

    model_config = ConfigDict(frozen=True)

    result: BetResult
    profit: float | None
    returned_stake: float | None


class CalibrationBin(BaseModel):
    """One reliability/calibration bucket."""

    model_config = ConfigDict(frozen=True)

    bin_lower: float
    bin_upper: float
    count: int
    mean_prediction: float | None
    observed_rate: float | None


class WalkForwardSplit(BaseModel):
    """Indices and time boundaries for one walk-forward fold."""

    model_config = ConfigDict(frozen=True)

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_end: datetime
    test_start: datetime

    @field_validator("train_end", "test_start")
    @classmethod
    def timestamps_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class PerformanceRecord(BaseModel):
    """Normalized settled-bet record for grouped summaries."""

    model_config = ConfigDict(frozen=True)

    model_version: str
    sportsbook: str
    market_key: str
    market_side: str
    line: float
    american_odds: int
    stake: float = Field(gt=0)
    profit: float
    result: BetResult
    settled_at: datetime
    model_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    outcome: int | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("settled_at")
    @classmethod
    def settled_utc(cls, value: datetime) -> datetime:
        return _utc(value)