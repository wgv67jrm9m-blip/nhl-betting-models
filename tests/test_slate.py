"""Integration tests for normalized NHL SOG slate assembly."""

from __future__ import annotations

from pathlib import Path

from nhl_betting_models.data.canonical_ids import (
    CanonicalEntityType,
    CanonicalIdMap,
    CanonicalIdMapping,
)
from nhl_betting_models.data.slate import (
    SOGSlatePaths,
    assemble_sog_slate,
    assemble_validated_sog_slate,
)

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_identity_map() -> CanonicalIdMap:
    """Return canonical IDs required by the synthetic odds fixture."""

    mappings = [
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.GAME,
            source_name="synthetic-odds",
            source_entity_id="game-a",
            canonical_id="canonical-game-a",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.GAME,
            source_name="synthetic-odds",
            source_entity_id="game-b",
            canonical_id="canonical-game-b",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.PLAYER,
            source_name="synthetic-odds",
            source_entity_id="player-a",
            canonical_id="canonical-player-a",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.PLAYER,
            source_name="synthetic-odds",
            source_entity_id="player-b",
            canonical_id="canonical-player-b",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.SPORTSBOOK,
            source_name="synthetic-odds",
            source_entity_id="book-a",
            canonical_id="canonical-book-a",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.SPORTSBOOK,
            source_name="synthetic-odds",
            source_entity_id="book-b",
            canonical_id="canonical-book-b",
        ),
    ]

    return CanonicalIdMap(mappings)


def synthetic_slate_paths() -> SOGSlatePaths:
    """Return local fixture paths for one synthetic NHL SOG slate."""

    return SOGSlatePaths(
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


def test_assemble_sog_slate_loads_all_synthetic_inputs() -> None:
    """Load every synthetic input type into one normalized slate."""

    slate = assemble_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )

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


def test_assemble_validated_sog_slate_returns_quality_result() -> None:
    """Return a usable slate together with cross-record validation."""

    result = assemble_validated_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )

    assert result.is_valid
    assert result.quality.is_valid
    assert len(result.slate.games) == 3
    assert len(result.slate.sportsbook_offers) == 9