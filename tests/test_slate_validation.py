"""Tests for cross-record normalized NHL SOG slate validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nhl_betting_models.data.quality import (
    DataQualityFlag,
    DataQualitySeverity,
)
from nhl_betting_models.data.slate import (
    SOGSlate,
    SOGSlatePaths,
    assemble_sog_slate,
)
from nhl_betting_models.data.slate_validation import (
    validate_sog_slate_links,
)
from tests.test_slate import synthetic_identity_map

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_slate() -> SOGSlate:
    """Return the fully assembled synthetic slate used for validation."""

    paths = SOGSlatePaths(
        schedule=FIXTURES / "synthetic_schedule.json",
        player_game_stats=FIXTURES / "synthetic_player_game_logs.csv",
        role_projections=FIXTURES / "synthetic_role_projections.csv",
        starting_goalies=FIXTURES / "synthetic_goalie_status.json",
        availability=FIXTURES / "synthetic_injury_status.json",
        opponent_sog_profiles=(
            FIXTURES / "synthetic_opponent_sog_profiles.csv"
        ),
        sportsbook_offers=(
            FIXTURES / "synthetic_sportsbook_sog_offers.json"
        ),
    )

    return assemble_sog_slate(paths, synthetic_identity_map())


def test_synthetic_slate_has_no_missing_game_errors() -> None:
    """Accept records whose source game IDs occur in the schedule."""

    result = validate_sog_slate_links(synthetic_slate())

    missing_game_issues = [
        issue
        for issue in result.issues
        if issue.flag is DataQualityFlag.MISSING_SLATE_GAME
    ]

    assert missing_game_issues == []
    assert result.is_valid


def test_missing_game_reference_is_returned_as_error() -> None:
    """Flag a player-stat record that references an unscheduled game."""

    slate = synthetic_slate()
    bad_stat = slate.player_game_stats[0].model_copy(
        update={"source_game_id": "missing-game"}
    )
    bad_slate = replace(
        slate,
        player_game_stats=(bad_stat, *slate.player_game_stats[1:]),
    )

    result = validate_sog_slate_links(bad_slate)

    assert result.is_valid is False
    assert result.flags == (
        DataQualityFlag.MISSING_SLATE_GAME,
        DataQualityFlag.MISSING_SLATE_PLAYER,
    )

    missing_game_issue = result.issues[0]
    assert missing_game_issue.flag is DataQualityFlag.MISSING_SLATE_GAME
    assert missing_game_issue.severity is DataQualitySeverity.ERROR
    assert missing_game_issue.related_record_ids == ("raw-stats-1",)
    assert "missing-game" in missing_game_issue.message