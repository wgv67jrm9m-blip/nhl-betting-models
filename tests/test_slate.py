"""Integration tests for normalized NHL SOG slate assembly."""

from __future__ import annotations

from pathlib import Path

from nhl_betting_models.data.slate import (
    SOGSlatePaths,
    assemble_sog_slate,
)
from tests.test_file_adapters import synthetic_identity_map

FIXTURES = Path(__file__).parent / "fixtures"


def test_assemble_sog_slate_loads_all_synthetic_inputs() -> None:
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

    slate = assemble_sog_slate(paths, synthetic_identity_map())

    assert len(slate.games) == 3
    assert len(slate.player_game_stats) == 3
    assert len(slate.role_projections) == 2
    assert len(slate.starting_goalies) == 2
    assert len(slate.availability) == 3
    assert len(slate.opponent_sog_profiles) == 3
    assert len(slate.sportsbook_offers) == 9

    first_offer = slate.sportsbook_offers[0]
    assert first_offer.canonical_event_id == "canonical-game-a"
    assert first_offer.canonical_subject_id == "canonical-player-a"
    assert first_offer.sportsbook == "canonical-book-a"