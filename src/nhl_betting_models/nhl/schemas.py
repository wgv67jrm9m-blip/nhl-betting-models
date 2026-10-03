"""Typed schemas for NHL player shots-on-goal modeling."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nhl_betting_models.common.schemas import MarketEvaluation, MarketSide


def _utc(value: datetime) -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class PlayerPosition(StrEnum):
    """Position groups used for SOG priors and matchup features."""

    FORWARD = "forward"
    DEFENSEMAN = "defenseman"


class SOGGameState(StrEnum):
    """Game states modeled independently for player SOG opportunity."""

    FIVE_ON_FIVE = "5v5"
    POWER_PLAY = "power_play"
    SHORT_HANDED = "short_handed"


class SOGDistributionFamily(StrEnum):
    """Supported count-distribution families."""

    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"


class ConfidenceTier(StrEnum):
    """Model-input completeness and certainty, not wager win confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_PROJECTION = "no_projection"


class SOGQualityFlag(StrEnum):
    """SOG-specific quality conditions that can affect projection confidence."""

    MISSING_PROJECTED_MINUTES = "missing_projected_minutes"
    MISSING_PLAYER_RATE = "missing_player_rate"
    MISSING_POSITION_PRIOR = "missing_position_prior"
    INSUFFICIENT_PLAYER_EXPOSURE = "insufficient_player_exposure"
    PROJECTED_ROLE = "projected_role"
    UNCERTAIN_ROLE = "uncertain_role"
    STALE_FEATURE = "stale_feature"
    POST_FORECAST_FEATURE = "post_forecast_feature"
    MISSING_CONTEXT = "missing_context"
    INVALID_CONTEXTUAL_ADJUSTMENT = "invalid_contextual_adjustment"
    INVALID_DISTRIBUTION_PARAMETERS = "invalid_distribution_parameters"


class SOGRateObservation(BaseModel):
    """Timestamped player SOG-rate observation for one game state."""

    model_config = ConfigDict(frozen=True)

    canonical_player_id: str
    position: PlayerPosition
    game_state: SOGGameState
    sog_per_60: float = Field(ge=0.0)
    exposure_minutes: float = Field(ge=0.0)
    as_of_timestamp: datetime

    @field_validator("as_of_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class SOGRatePrior(BaseModel):
    """Position/game-state prior used for empirical-Bayes shrinkage."""

    model_config = ConfigDict(frozen=True)

    position: PlayerPosition
    game_state: SOGGameState
    mean_sog_per_60: float = Field(ge=0.0)
    prior_equivalent_minutes: float = Field(gt=0.0)
    prior_version: str


class SOGShrunkRate(BaseModel):
    """Auditable empirical-Bayes player rate after prior shrinkage."""

    model_config = ConfigDict(frozen=True)

    canonical_player_id: str
    position: PlayerPosition
    game_state: SOGGameState
    observed_sog_per_60: float = Field(ge=0.0)
    observed_exposure_minutes: float = Field(ge=0.0)
    prior_sog_per_60: float = Field(ge=0.0)
    prior_equivalent_minutes: float = Field(gt=0.0)
    posterior_sog_per_60: float = Field(ge=0.0)
    as_of_timestamp: datetime
    quality_flags: tuple[SOGQualityFlag, ...] = ()

    @field_validator("as_of_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class SOGProjectedMinutes(BaseModel):
    """Projected player minutes split by modeled game state."""

    model_config = ConfigDict(frozen=True)

    canonical_player_id: str
    five_on_five_minutes: float = Field(ge=0.0)
    power_play_minutes: float = Field(ge=0.0)
    short_handed_minutes: float = Field(ge=0.0)
    as_of_timestamp: datetime
    quality_flags: tuple[SOGQualityFlag, ...] = ()

    @field_validator("as_of_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    @property
    def total_minutes(self) -> float:
        """Return total projected minutes represented by the three states."""

        return (
            self.five_on_five_minutes
            + self.power_play_minutes
            + self.short_handed_minutes
        )

    def minutes_for(self, game_state: SOGGameState) -> float:
        """Return projected minutes for one modeled game state."""

        if game_state is SOGGameState.FIVE_ON_FIVE:
            return self.five_on_five_minutes
        if game_state is SOGGameState.POWER_PLAY:
            return self.power_play_minutes
        return self.short_handed_minutes


class SOGContextualAdjustment(BaseModel):
    """One transparent, bounded multiplicative SOG adjustment."""

    model_config = ConfigDict(frozen=True)

    adjustment_key: str
    requested_multiplier: float = Field(gt=0.0)
    lower_bound: float = Field(gt=0.0)
    upper_bound: float = Field(gt=0.0)
    applied_multiplier: float = Field(gt=0.0)
    as_of_timestamp: datetime

    @field_validator("as_of_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_bounds(self) -> SOGContextualAdjustment:
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        if not self.lower_bound <= self.applied_multiplier <= self.upper_bound:
            raise ValueError("applied_multiplier must be within configured bounds")
        return self


class SOGContextualAdjustmentBounds(BaseModel):
    """Configured bounds for one contextual adjustment type."""

    model_config = ConfigDict(frozen=True)

    adjustment_key: str
    lower_bound: float = Field(gt=0.0)
    upper_bound: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> SOGContextualAdjustmentBounds:
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class SOGProjectionComponent(BaseModel):
    """Expected-SOG contribution from one game state."""

    model_config = ConfigDict(frozen=True)

    game_state: SOGGameState
    projected_minutes: float = Field(ge=0.0)
    sog_per_60: float = Field(ge=0.0)
    expected_sog: float = Field(ge=0.0)


class SOGProjectionInput(BaseModel):
    """Complete timestamped input required for one SOG expectation."""

    model_config = ConfigDict(frozen=True)

    canonical_event_id: str
    canonical_player_id: str
    position: PlayerPosition
    model_run_timestamp: datetime
    projected_minutes: SOGProjectedMinutes
    shrunk_rates: tuple[SOGShrunkRate, ...]
    contextual_adjustments: tuple[SOGContextualAdjustment, ...] = ()
    input_quality_flags: tuple[SOGQualityFlag, ...] = ()

    @field_validator("model_run_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_player_identity(self) -> SOGProjectionInput:
        if self.projected_minutes.canonical_player_id != self.canonical_player_id:
            raise ValueError("projected-minutes player does not match projection player")

        for rate in self.shrunk_rates:
            if rate.canonical_player_id != self.canonical_player_id:
                raise ValueError("shrunk-rate player does not match projection player")
            if rate.position is not self.position:
                raise ValueError("shrunk-rate position does not match projection position")

        return self


class SOGProjectionResult(BaseModel):
    """Auditable expected-SOG result before market pricing."""

    model_config = ConfigDict(frozen=True)

    canonical_event_id: str
    canonical_player_id: str
    model_run_timestamp: datetime
    base_expected_sog: float = Field(ge=0.0)
    adjusted_expected_sog: float = Field(ge=0.0)
    components: tuple[SOGProjectionComponent, ...]
    contextual_adjustments: tuple[SOGContextualAdjustment, ...] = ()
    confidence_tier: ConfidenceTier
    quality_flags: tuple[SOGQualityFlag, ...] = ()

    @field_validator("model_run_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class SOGDistributionConfig(BaseModel):
    """Configuration for converting expected SOG into count probabilities."""

    model_config = ConfigDict(frozen=True)

    family: SOGDistributionFamily = SOGDistributionFamily.NEGATIVE_BINOMIAL
    dispersion: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_dispersion(self) -> SOGDistributionConfig:
        if (
            self.family is SOGDistributionFamily.NEGATIVE_BINOMIAL
            and self.dispersion is None
        ):
            raise ValueError("negative-binomial distribution requires dispersion")

        if self.family is SOGDistributionFamily.POISSON and self.dispersion is not None:
            raise ValueError("Poisson distribution must not specify dispersion")

        return self


class SOGDistributionResult(BaseModel):
    """Count-distribution parameters tied to one SOG expectation."""

    model_config = ConfigDict(frozen=True)

    expected_sog: float = Field(ge=0.0)
    family: SOGDistributionFamily
    dispersion: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_dispersion(self) -> SOGDistributionResult:
        if (
            self.family is SOGDistributionFamily.NEGATIVE_BINOMIAL
            and self.dispersion is None
        ):
            raise ValueError("negative-binomial result requires dispersion")

        if self.family is SOGDistributionFamily.POISSON and self.dispersion is not None:
            raise ValueError("Poisson result must not specify dispersion")

        return self


class SOGSideProbabilities(BaseModel):
    """Win, loss, and push probabilities for one SOG market side."""

    model_config = ConfigDict(frozen=True)

    side: MarketSide
    line: float = Field(ge=0.0)
    win_probability: float = Field(ge=0.0, le=1.0)
    loss_probability: float = Field(ge=0.0, le=1.0)
    push_probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sog_side_probabilities(self) -> SOGSideProbabilities:
        if self.side not in {MarketSide.OVER, MarketSide.UNDER}:
            raise ValueError("SOG side must be over or under")

        total = self.win_probability + self.loss_probability + self.push_probability
        if abs(total - 1.0) > 1e-12:
            raise ValueError("win, loss, and push probabilities must sum to 1")

        return self


class SOGEvaluationResult(BaseModel):
    """SOG-specific wrapper around shared market evaluation output."""

    model_config = ConfigDict(frozen=True)

    canonical_event_id: str
    canonical_player_id: str
    market_key: str
    side: MarketSide
    line: float = Field(ge=0.0)
    probabilities: SOGSideProbabilities
    market_evaluation: MarketEvaluation
    confidence_tier: ConfidenceTier
    quality_flags: tuple[SOGQualityFlag, ...] = ()

    @model_validator(mode="after")
    def validate_side_and_line(self) -> SOGEvaluationResult:
        if self.side is not self.probabilities.side:
            raise ValueError("evaluation side does not match probability side")
        if self.line != self.probabilities.line:
            raise ValueError("evaluation line does not match probability line")
        return self