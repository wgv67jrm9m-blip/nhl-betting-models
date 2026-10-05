"""Cross-record validation for normalized NHL SOG slates."""

from __future__ import annotations

from nhl_betting_models.data.base import (
    NormalizedAvailability,
    NormalizedGoalieStatus,
    NormalizedPlayerGameStateStats,
    NormalizedRoleProjection,
)
from nhl_betting_models.data.quality import (
    DataQualityFlag,
    DataQualityIssue,
    DataQualityResult,
    DataQualitySeverity,
)
from nhl_betting_models.data.slate import SOGSlate


def _missing_game_issue(
    *,
    source_game_id: str | None,
    raw_payload_record_id: str,
    game_ids: set[str],
    record_type: str,
) -> DataQualityIssue | None:
    """Return an issue when one record references an unscheduled game."""

    if source_game_id is None or source_game_id in game_ids:
        return None

    return DataQualityIssue(
        flag=DataQualityFlag.MISSING_SLATE_GAME,
        severity=DataQualitySeverity.ERROR,
        message=(
            f"{record_type} references a game not present "
            f"in the loaded schedule: {source_game_id}"
        ),
        related_record_ids=(raw_payload_record_id,),
    )


def _missing_game_issues_for_player_stats(
    records: tuple[NormalizedPlayerGameStateStats, ...],
    *,
    game_ids: set[str],
) -> list[DataQualityIssue]:
    """Validate player-stat references against scheduled games."""

    issues: list[DataQualityIssue] = []

    for player_stat in records:
        issue = _missing_game_issue(
            source_game_id=player_stat.source_game_id,
            raw_payload_record_id=(
                player_stat.provenance.raw_payload_record_id
            ),
            game_ids=game_ids,
            record_type="player game statistics",
        )

        if issue is not None:
            issues.append(issue)

    return issues


def _missing_game_issues_for_role_projections(
    records: tuple[NormalizedRoleProjection, ...],
    *,
    game_ids: set[str],
) -> list[DataQualityIssue]:
    """Validate role-projection references against scheduled games."""

    issues: list[DataQualityIssue] = []

    for role_projection in records:
        issue = _missing_game_issue(
            source_game_id=role_projection.source_game_id,
            raw_payload_record_id=(
                role_projection.provenance.raw_payload_record_id
            ),
            game_ids=game_ids,
            record_type="role projection",
        )

        if issue is not None:
            issues.append(issue)

    return issues


def _missing_game_issues_for_goalies(
    records: tuple[NormalizedGoalieStatus, ...],
    *,
    game_ids: set[str],
) -> list[DataQualityIssue]:
    """Validate goalie-status references against scheduled games."""

    issues: list[DataQualityIssue] = []

    for goalie_status in records:
        issue = _missing_game_issue(
            source_game_id=goalie_status.source_game_id,
            raw_payload_record_id=(
                goalie_status.provenance.raw_payload_record_id
            ),
            game_ids=game_ids,
            record_type="starting goalie status",
        )

        if issue is not None:
            issues.append(issue)

    return issues


def _missing_game_issues_for_availability(
    records: tuple[NormalizedAvailability, ...],
    *,
    game_ids: set[str],
) -> list[DataQualityIssue]:
    """Validate optional availability-game references against schedule."""

    issues: list[DataQualityIssue] = []

    for availability in records:
        issue = _missing_game_issue(
            source_game_id=availability.source_game_id,
            raw_payload_record_id=(
                availability.provenance.raw_payload_record_id
            ),
            game_ids=game_ids,
            record_type="availability record",
        )

        if issue is not None:
            issues.append(issue)

    return issues


def validate_sog_slate_links(slate: SOGSlate) -> DataQualityResult:
    """Validate that slate records link to scheduled games and players."""

    scheduled_game_ids = {
        game.source_game_id
        for game in slate.games
    }

    issues = [
        *_missing_game_issues_for_player_stats(
            slate.player_game_stats,
            game_ids=scheduled_game_ids,
        ),
        *_missing_game_issues_for_role_projections(
            slate.role_projections,
            game_ids=scheduled_game_ids,
        ),
        *_missing_game_issues_for_goalies(
            slate.starting_goalies,
            game_ids=scheduled_game_ids,
        ),
        *_missing_game_issues_for_availability(
            slate.availability,
            game_ids=scheduled_game_ids,
        ),
    ]

    player_stat_keys = {
        (player_stat.source_game_id, player_stat.source_player_id)
        for player_stat in slate.player_game_stats
    }
    role_projection_keys = {
        (
            role_projection.source_game_id,
            role_projection.source_player_id,
        )
        for role_projection in slate.role_projections
    }

    for role_projection in slate.role_projections:
        key = (
            role_projection.source_game_id,
            role_projection.source_player_id,
        )

        if key not in player_stat_keys:
            issues.append(
                DataQualityIssue(
                    flag=DataQualityFlag.MISSING_SLATE_PLAYER,
                    severity=DataQualitySeverity.WARNING,
                    message=(
                        "role projection has no corresponding "
                        "player game-stat record"
                    ),
                    related_record_ids=(
                        role_projection.provenance.raw_payload_record_id,
                    ),
                )
            )

    for player_stat in slate.player_game_stats:
        key = (
            player_stat.source_game_id,
            player_stat.source_player_id,
        )

        if key not in role_projection_keys:
            issues.append(
                DataQualityIssue(
                    flag=DataQualityFlag.MISSING_SLATE_PLAYER,
                    severity=DataQualitySeverity.WARNING,
                    message=(
                        "player game-stat record has no corresponding "
                        "role projection"
                    ),
                    related_record_ids=(
                        player_stat.provenance.raw_payload_record_id,
                    ),
                )
            )

    return DataQualityResult(issues=tuple(issues))