"""Assembly of normalized NHL SOG slate inputs from local files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nhl_betting_models.common.schemas import Offer
from nhl_betting_models.data.base import (
    NormalizedAvailability,
    NormalizedGame,
    NormalizedGoalieStatus,
    NormalizedOpponentSOGProfile,
    NormalizedPlayerGameStateStats,
    NormalizedRoleProjection,
)
from nhl_betting_models.data.canonical_ids import CanonicalIdMap
from nhl_betting_models.data.file_adapters import (
    CsvOpponentSOGProfileFileAdapter,
    CsvPlayerGameStatsFileAdapter,
    CsvRoleProjectionFileAdapter,
    JsonAvailabilityFileAdapter,
    JsonScheduleFileAdapter,
    JsonSportsbookOfferFileAdapter,
    JsonStartingGoalieFileAdapter,
)
from nhl_betting_models.data.quality import DataQualityResult


@dataclass(frozen=True)
class SOGSlate:
    """Normalized inputs needed to prepare an NHL SOG slate."""

    games: tuple[NormalizedGame, ...]
    player_game_stats: tuple[NormalizedPlayerGameStateStats, ...]
    role_projections: tuple[NormalizedRoleProjection, ...]
    starting_goalies: tuple[NormalizedGoalieStatus, ...]
    availability: tuple[NormalizedAvailability, ...]
    opponent_sog_profiles: tuple[NormalizedOpponentSOGProfile, ...]
    sportsbook_offers: tuple[Offer, ...]


@dataclass(frozen=True)
class SOGSlateAssemblyResult:
    """One assembled slate with its cross-record quality findings."""

    slate: SOGSlate
    quality: DataQualityResult

    @property
    def is_valid(self) -> bool:
        """Return whether the assembled slate has no validation errors."""

        return self.quality.is_valid


@dataclass(frozen=True)
class SOGSlatePaths:
    """Local files required to assemble one NHL SOG slate."""

    schedule: Path
    player_game_stats: Path
    role_projections: Path
    starting_goalies: Path
    availability: Path
    opponent_sog_profiles: Path
    sportsbook_offers: Path


def assemble_sog_slate(
    paths: SOGSlatePaths,
    identity_map: CanonicalIdMap,
) -> SOGSlate:
    """Read and normalize all local inputs for an NHL SOG slate."""

    return SOGSlate(
        games=JsonScheduleFileAdapter(paths.schedule).games(),
        player_game_stats=CsvPlayerGameStatsFileAdapter(
            paths.player_game_stats
        ).player_game_stats(),
        role_projections=CsvRoleProjectionFileAdapter(
            paths.role_projections
        ).role_projections(),
        starting_goalies=JsonStartingGoalieFileAdapter(
            paths.starting_goalies
        ).starting_goalies(),
        availability=JsonAvailabilityFileAdapter(
            paths.availability
        ).availability(),
        opponent_sog_profiles=CsvOpponentSOGProfileFileAdapter(
            paths.opponent_sog_profiles
        ).opponent_sog_profiles(),
        sportsbook_offers=JsonSportsbookOfferFileAdapter(
            paths.sportsbook_offers,
            identity_map,
        ).offers(),
    )


def assemble_validated_sog_slate(
    paths: SOGSlatePaths,
    identity_map: CanonicalIdMap,
) -> SOGSlateAssemblyResult:
    """Assemble an NHL SOG slate and return its quality findings."""

    from nhl_betting_models.data.slate_validation import (
        validate_sog_slate_links,
    )

    slate = assemble_sog_slate(paths, identity_map)

    return SOGSlateAssemblyResult(
        slate=slate,
        quality=validate_sog_slate_links(slate),
    )