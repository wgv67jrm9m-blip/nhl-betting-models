"""Provider-neutral normalized records and provider interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nhl_betting_models.common.schemas import Offer
from nhl_betting_models.data.provenance import SourceProvenance
from nhl_betting_models.nhl.schemas import PlayerPosition, SOGGameState


def _utc(value: datetime) -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class ConfirmationStatus(StrEnum):
    """Provider-neutral certainty attached to projected information."""

    CONFIRMED = "confirmed"
    PROJECTED = "projected"
    UNCONFIRMED = "unconfirmed"
    RULED_OUT = "ruled_out"
    UNKNOWN = "unknown"


class AvailabilityStatus(StrEnum):
    """Provider-neutral player availability state."""

    AVAILABLE = "available"
    INJURED = "injured"
    SCRATCHED = "scratched"
    CALLED_UP = "called_up"
    RULED_OUT = "ruled_out"
    UNKNOWN = "unknown"


class NormalizedGame(BaseModel):
    """Provider-neutral NHL game record."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_home_team_id: str
    source_away_team_id: str
    game_start_timestamp: datetime
    provenance: SourceProvenance

    @field_validator("game_start_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime) -> datetime:
        """Require an aware game-start timestamp and normalize it to UTC."""

        return _utc(value)


class NormalizedRosterPlayer(BaseModel):
    """Provider-neutral roster membership record."""

    model_config = ConfigDict(frozen=True)

    source_player_id: str
    source_team_id: str
    display_name: str
    position: PlayerPosition
    provenance: SourceProvenance


class NormalizedPlayerGameStateStats(BaseModel):
    """Historical player SOG and exposure for one modeled game state."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_player_id: str
    source_team_id: str
    position: PlayerPosition
    game_state: SOGGameState
    shots_on_goal: int = Field(ge=0)
    exposure_minutes: float = Field(ge=0.0)
    provenance: SourceProvenance


class NormalizedPlayByPlayEvent(BaseModel):
    """Provider-neutral play-by-play event relevant to SOG features."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_event_id: str
    source_team_id: str | None = None
    source_player_id: str | None = None
    event_type: str
    game_state: SOGGameState | None = None
    event_timestamp: datetime | None = None
    provenance: SourceProvenance

    @field_validator("event_timestamp")
    @classmethod
    def timestamp_utc(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Normalize an optional event timestamp to UTC."""

        if value is None:
            return None
        return _utc(value)


class NormalizedRoleProjection(BaseModel):
    """Projected player deployment before an NHL game."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_player_id: str
    source_team_id: str
    projected_line: int | None = Field(default=None, ge=1)
    power_play_unit: int | None = Field(default=None, ge=1)
    five_on_five_minutes: float | None = Field(default=None, ge=0.0)
    power_play_minutes: float | None = Field(default=None, ge=0.0)
    short_handed_minutes: float | None = Field(default=None, ge=0.0)
    confirmation_status: ConfirmationStatus
    provenance: SourceProvenance


class NormalizedGoalieStatus(BaseModel):
    """Provider-neutral starting-goalie status."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_team_id: str
    source_goalie_id: str | None = None
    confirmation_status: ConfirmationStatus
    provenance: SourceProvenance


class NormalizedAvailability(BaseModel):
    """Provider-neutral player availability record."""

    model_config = ConfigDict(frozen=True)

    source_player_id: str
    source_team_id: str
    source_game_id: str | None = None
    availability_status: AvailabilityStatus
    confirmation_status: ConfirmationStatus
    provenance: SourceProvenance


class NormalizedOpponentSOGProfile(BaseModel):
    """Opponent SOG allowed profile by position and game state."""

    model_config = ConfigDict(frozen=True)

    source_team_id: str
    opponent_position: PlayerPosition
    game_state: SOGGameState
    sog_allowed_per_60: float = Field(ge=0.0)
    exposure_minutes: float = Field(ge=0.0)
    provenance: SourceProvenance


class NormalizedFinalSOGOutcome(BaseModel):
    """Official final player SOG outcome."""

    model_config = ConfigDict(frozen=True)

    source_game_id: str
    source_player_id: str
    final_sog: int = Field(ge=0)
    confirmation_status: ConfirmationStatus
    provenance: SourceProvenance


class ScheduleProvider(Protocol):
    """Supply normalized games and schedule identity."""

    def games(self) -> Sequence[NormalizedGame]:
        """Return normalized NHL games."""
        ...


class RosterProvider(Protocol):
    """Supply normalized NHL roster membership."""

    def roster(self) -> Sequence[NormalizedRosterPlayer]:
        """Return normalized roster records."""
        ...


class PlayerGameStatsProvider(Protocol):
    """Supply normalized historical player game-state statistics."""

    def player_game_stats(
        self,
    ) -> Sequence[NormalizedPlayerGameStateStats]:
        """Return normalized player statistics."""
        ...


class PlayByPlayProvider(Protocol):
    """Supply normalized play-by-play events."""

    def play_by_play(self) -> Sequence[NormalizedPlayByPlayEvent]:
        """Return normalized play-by-play events."""
        ...


class RoleProjectionProvider(Protocol):
    """Supply normalized line, PP-role, and TOI projections."""

    def role_projections(
        self,
    ) -> Sequence[NormalizedRoleProjection]:
        """Return normalized role projections."""
        ...


class StartingGoalieProvider(Protocol):
    """Supply normalized starting-goalie states."""

    def starting_goalies(
        self,
    ) -> Sequence[NormalizedGoalieStatus]:
        """Return normalized starting-goalie records."""
        ...


class AvailabilityProvider(Protocol):
    """Supply normalized player availability information."""

    def availability(
        self,
    ) -> Sequence[NormalizedAvailability]:
        """Return normalized player availability records."""
        ...


class OpponentSOGProfileProvider(Protocol):
    """Supply normalized opponent SOG profiles."""

    def opponent_sog_profiles(
        self,
    ) -> Sequence[NormalizedOpponentSOGProfile]:
        """Return normalized opponent SOG profiles."""
        ...


class SportsbookOfferProvider(Protocol):
    """Supply normalized sportsbook offers using the shared Offer schema."""

    def offers(self) -> Sequence[Offer]:
        """Return normalized immutable sportsbook offers."""
        ...


class OfficialOutcomeProvider(Protocol):
    """Supply official final player SOG outcomes."""

    def final_sog_outcomes(
        self,
    ) -> Sequence[NormalizedFinalSOGOutcome]:
        """Return normalized official SOG outcomes."""
        ...


class ClosingOddsProvider(Protocol):
    """Supply immutable sportsbook snapshots usable for CLV."""

    def closing_offers(self) -> Sequence[Offer]:
        """Return normalized historical closing-price candidates."""
        ...