"""Build source-linked candidate rows for NHL SOG projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nhl_betting_models.data.base import (
    AvailabilityStatus,
    NormalizedAvailability,
    NormalizedPlayerGameStateStats,
    NormalizedRoleProjection,
)
from nhl_betting_models.data.slate import SOGSlateAssemblyResult
from nhl_betting_models.nhl.schemas import PlayerPosition


class SOGProjectionCandidateExclusionReason(StrEnum):
    """Reasons one player cannot become a projection candidate."""

    INVALID_SLATE = "invalid_slate"
    UNAVAILABLE_PLAYER = "unavailable_player"
    MISSING_PLAYER_STATS = "missing_player_stats"


@dataclass(frozen=True)
class SOGProjectionCandidate:
    """Source-linked player context before canonical projection assembly."""

    source_game_id: str
    source_player_id: str
    source_team_id: str
    position: PlayerPosition
    role_projection: NormalizedRoleProjection
    player_game_stats: tuple[NormalizedPlayerGameStateStats, ...]
    availability: NormalizedAvailability | None


@dataclass(frozen=True)
class SOGProjectionCandidateResult:
    """One candidate or one explicit exclusion from the projection workflow."""

    candidate: SOGProjectionCandidate | None
    exclusion_reason: SOGProjectionCandidateExclusionReason | None

    @property
    def is_eligible(self) -> bool:
        """Return whether this result contains a usable candidate."""

        return self.candidate is not None


def _availability_by_player_game(
    assembly_result: SOGSlateAssemblyResult,
) -> dict[tuple[str, str], NormalizedAvailability]:
    """Index game-specific availability by source game and player."""

    indexed: dict[tuple[str, str], NormalizedAvailability] = {}

    for availability in assembly_result.slate.availability:
        if availability.source_game_id is None:
            continue

        key = (
            availability.source_game_id,
            availability.source_player_id,
        )

        if key in indexed:
            raise ValueError(
                "multiple availability records found for one player and game"
            )

        indexed[key] = availability

    return indexed


def _player_stats_by_player_game(
    assembly_result: SOGSlateAssemblyResult,
) -> dict[
    tuple[str, str],
    tuple[NormalizedPlayerGameStateStats, ...],
]:
    """Index player game-state statistics by source game and player."""

    grouped: dict[
        tuple[str, str],
        list[NormalizedPlayerGameStateStats],
    ] = {}

    for player_stat in assembly_result.slate.player_game_stats:
        key = (
            player_stat.source_game_id,
            player_stat.source_player_id,
        )
        grouped.setdefault(key, []).append(player_stat)

    return {
        key: tuple(player_stats)
        for key, player_stats in grouped.items()
    }


def build_sog_projection_candidates(
    assembly_result: SOGSlateAssemblyResult,
) -> tuple[SOGProjectionCandidateResult, ...]:
    """Build candidates from a valid slate and explain all exclusions."""

    role_projections = assembly_result.slate.role_projections

    if not assembly_result.is_valid:
        return tuple(
            SOGProjectionCandidateResult(
                candidate=None,
                exclusion_reason=(
                    SOGProjectionCandidateExclusionReason.INVALID_SLATE
                ),
            )
            for _ in role_projections
        )

    availability_by_player_game = _availability_by_player_game(
        assembly_result
    )
    stats_by_player_game = _player_stats_by_player_game(
        assembly_result
    )
    results: list[SOGProjectionCandidateResult] = []

    excluded_statuses = {
        AvailabilityStatus.INJURED,
        AvailabilityStatus.SCRATCHED,
        AvailabilityStatus.RULED_OUT,
    }

    for role_projection in role_projections:
        key = (
            role_projection.source_game_id,
            role_projection.source_player_id,
        )
        availability = availability_by_player_game.get(key)

        if (
            availability is not None
            and availability.availability_status in excluded_statuses
        ):
            results.append(
                SOGProjectionCandidateResult(
                    candidate=None,
                    exclusion_reason=(
                        SOGProjectionCandidateExclusionReason
                        .UNAVAILABLE_PLAYER
                    ),
                )
            )
            continue

        player_game_stats = stats_by_player_game.get(key, ())

        if not player_game_stats:
            results.append(
                SOGProjectionCandidateResult(
                    candidate=None,
                    exclusion_reason=(
                        SOGProjectionCandidateExclusionReason
                        .MISSING_PLAYER_STATS
                    ),
                )
            )
            continue

        positions = {
            player_stat.position
            for player_stat in player_game_stats
        }

        if len(positions) != 1:
            raise ValueError(
                "player game-state statistics must use one position "
                "per player and game"
            )

        results.append(
            SOGProjectionCandidateResult(
                candidate=SOGProjectionCandidate(
                    source_game_id=role_projection.source_game_id,
                    source_player_id=role_projection.source_player_id,
                    source_team_id=role_projection.source_team_id,
                    position=next(iter(positions)),
                    role_projection=role_projection,
                    player_game_stats=player_game_stats,
                    availability=availability,
                ),
                exclusion_reason=None,
            )
        )

    return tuple(results)