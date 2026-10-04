"""Synthetic tests for local Part 2 NHL file adapters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nhl_betting_models.common.market_matching import compare_offers
from nhl_betting_models.common.schemas import MarketSide, Offer
from nhl_betting_models.data.base import (
    AvailabilityStatus,
    ConfirmationStatus,
)
from nhl_betting_models.data.canonical_ids import (
    CanonicalEntityType,
    CanonicalIdMap,
    CanonicalIdMapping,
    IdentityResolutionStatus,
)
from nhl_betting_models.data.file_adapters import (
    CsvFinalSOGOutcomeFileAdapter,
    CsvOpponentSOGProfileFileAdapter,
    CsvPlayerGameStatsFileAdapter,
    CsvRoleProjectionFileAdapter,
    JsonAvailabilityFileAdapter,
    JsonClosingOddsFileAdapter,
    JsonScheduleFileAdapter,
    JsonSportsbookOfferFileAdapter,
    JsonStartingGoalieFileAdapter,
)
from nhl_betting_models.data.normalizers import (
    normalize_sportsbook_offer,
)
from nhl_betting_models.data.quality import (
    DataQualityFlag,
    validate_game_start_times,
    validate_identity_resolution,
    validate_offer_set,
    validate_two_sided_market,
)
from nhl_betting_models.nhl.schemas import (
    PlayerPosition,
    SOGGameState,
)

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_identity_map() -> CanonicalIdMap:
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


def test_schedule_adapter_returns_real_records_and_flags_conflicting_start(
) -> None:
    adapter = JsonScheduleFileAdapter(
        FIXTURES / "synthetic_schedule.json"
    )
    games = adapter.games()

    assert len(games) == 3
    assert games[0].source_game_id == "game-a"
    assert games[0].game_start_timestamp == datetime(
        2026,
        1,
        11,
        0,
        tzinfo=UTC,
    )
    assert (
        games[0].provenance.raw_payload_record_id
        == "raw-schedule-1"
    )

    quality = validate_game_start_times(games)
    assert (
        DataQualityFlag.MISMATCHED_GAME_START_TIME
        in quality.flags
    )


def test_player_game_stats_adapter_returns_typed_game_states() -> None:
    rows = CsvPlayerGameStatsFileAdapter(
        FIXTURES / "synthetic_player_game_logs.csv"
    ).player_game_stats()

    assert len(rows) == 3
    assert rows[0].position is PlayerPosition.FORWARD
    assert rows[1].game_state is SOGGameState.POWER_PLAY
    assert rows[2].shots_on_goal == 0
    assert (
        rows[0].provenance.raw_payload_record_id
        == "raw-stats-1"
    )


def test_role_projection_adapter_preserves_projected_usage() -> None:
    rows = CsvRoleProjectionFileAdapter(
        FIXTURES / "synthetic_role_projections.csv"
    ).role_projections()

    assert len(rows) == 2
    assert rows[0].five_on_five_minutes == 15.5
    assert (
        rows[0].confirmation_status
        is ConfirmationStatus.CONFIRMED
    )
    assert (
        rows[1].confirmation_status
        is ConfirmationStatus.PROJECTED
    )


def test_goalie_adapter_contains_confirmed_and_projected_statuses(
) -> None:
    rows = JsonStartingGoalieFileAdapter(
        FIXTURES / "synthetic_goalie_status.json"
    ).starting_goalies()

    assert len(rows) == 2
    assert {
        row.confirmation_status for row in rows
    } == {
        ConfirmationStatus.CONFIRMED,
        ConfirmationStatus.PROJECTED,
    }
    assert rows[0].source_goalie_id == "goalie-a"


def test_availability_adapter_preserves_supported_states_and_raw_linkage(
) -> None:
    rows = JsonAvailabilityFileAdapter(
        FIXTURES / "synthetic_injury_status.json"
    ).availability()

    assert len(rows) == 3
    assert [
        row.availability_status for row in rows
    ] == [
        AvailabilityStatus.AVAILABLE,
        AvailabilityStatus.INJURED,
        AvailabilityStatus.SCRATCHED,
    ]
    assert (
        rows[2].provenance.raw_payload_record_id
        == "raw-availability-3"
    )

    raw = json.loads(
        (
            FIXTURES / "synthetic_injury_status.json"
        ).read_text(encoding="utf-8")
    )
    assert [
        row["provider_label"] for row in raw
    ] == [
        "active",
        "questionable",
        "scratched",
    ]


def test_opponent_profile_adapter_returns_typed_profiles() -> None:
    rows = CsvOpponentSOGProfileFileAdapter(
        FIXTURES / "synthetic_opponent_sog_profiles.csv"
    ).opponent_sog_profiles()

    assert len(rows) == 3
    assert rows[0].sog_allowed_per_60 == 8.4
    assert (
        rows[2].opponent_position
        is PlayerPosition.DEFENSEMAN
    )


def test_final_sog_adapter_returns_confirmed_results() -> None:
    rows = CsvFinalSOGOutcomeFileAdapter(
        FIXTURES / "synthetic_sog_results.csv"
    ).final_sog_outcomes()

    assert len(rows) == 2
    assert [row.final_sog for row in rows] == [4, 2]
    assert all(
        row.confirmation_status
        is ConfirmationStatus.CONFIRMED
        for row in rows
    )


def test_sportsbook_adapter_returns_existing_offer_schema_and_canonical_ids(
) -> None:
    offers = JsonSportsbookOfferFileAdapter(
        FIXTURES / "synthetic_sportsbook_sog_offers.json",
        synthetic_identity_map(),
    ).offers()

    assert len(offers) == 9
    assert all(isinstance(offer, Offer) for offer in offers)
    assert offers[0].canonical_event_id == "canonical-game-a"
    assert (
        offers[0].canonical_subject_id
        == "canonical-player-a"
    )
    assert offers[0].sportsbook == "canonical-book-a"
    assert {offer.line for offer in offers} >= {2.5, 3.0}


def test_offer_quality_uses_existing_validators_for_complete_and_bad_cases(
) -> None:
    offers = JsonSportsbookOfferFileAdapter(
        FIXTURES / "synthetic_sportsbook_sog_offers.json",
        synthetic_identity_map(),
    ).offers()

    complete = offers[:2]
    assert validate_two_sided_market(complete).is_valid

    quality = validate_offer_set(
        offers,
        expected_canonical_player_id="canonical-player-a",
    )

    assert not quality.is_valid
    assert DataQualityFlag.STALE_ODDS in quality.flags
    assert (
        DataQualityFlag.INCOMPLETE_TWO_SIDED_MARKET
        in quality.flags
    )
    assert DataQualityFlag.DUPLICATE_OFFER in quality.flags
    assert DataQualityFlag.INVALID_ODDS in quality.flags
    assert (
        DataQualityFlag.MISMATCHED_PLAYER_IDENTITY
        in quality.flags
    )


def test_closing_adapter_exact_comparability_and_game_start_cutoff(
) -> None:
    offers = JsonClosingOddsFileAdapter(
        FIXTURES / "synthetic_closing_odds.json",
        synthetic_identity_map(),
    ).closing_offers()

    assert len(offers) == 8

    reference = offers[0]
    game_start = datetime(
        2026,
        1,
        11,
        0,
        tzinfo=UTC,
    )

    comparable_pregame = [
        offer
        for offer in offers
        if (
            compare_offers(reference, offer).comparable
            and offer.retrieval_timestamp < game_start
        )
    ]
    latest = max(
        comparable_pregame,
        key=lambda offer: offer.retrieval_timestamp,
    )

    assert latest.record_id == "close-latest-comparable"

    wrong_line = next(
        offer
        for offer in offers
        if offer.record_id == "close-wrong-line"
    )
    wrong_side = next(
        offer
        for offer in offers
        if offer.record_id == "close-wrong-side"
    )
    wrong_player = next(
        offer
        for offer in offers
        if offer.record_id == "close-wrong-player"
    )
    wrong_settlement = next(
        offer
        for offer in offers
        if offer.record_id == "close-wrong-settlement"
    )
    post_start = next(
        offer
        for offer in offers
        if offer.record_id == "close-post-start"
    )

    assert not compare_offers(
        reference,
        wrong_line,
    ).comparable
    assert not compare_offers(
        reference,
        wrong_side,
    ).comparable
    assert not compare_offers(
        reference,
        wrong_player,
    ).comparable
    assert not compare_offers(
        reference,
        wrong_settlement,
    ).comparable

    assert compare_offers(
        reference,
        post_start,
    ).comparable
    assert post_start.retrieval_timestamp >= game_start


def test_adapter_output_is_deterministic() -> None:
    adapter = JsonSportsbookOfferFileAdapter(
        FIXTURES / "synthetic_sportsbook_sog_offers.json",
        synthetic_identity_map(),
    )

    assert adapter.offers() == adapter.offers()

    schedule = JsonScheduleFileAdapter(
        FIXTURES / "synthetic_schedule.json"
    )
    assert schedule.games() == schedule.games()


def test_missing_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    path.write_text(
        '[{"source_game_id":"game-a"}]',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        JsonScheduleFileAdapter(path).games()


def test_missing_csv_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text(
        "source_game_id,source_player_id\n"
        "game-a,player-a\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        CsvPlayerGameStatsFileAdapter(
            path
        ).player_game_stats()


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-10T12:00:00",
        "not-a-timestamp",
    ],
)
def test_naive_or_malformed_timestamp_is_rejected(
    tmp_path: Path,
    timestamp: str,
) -> None:
    row = json.loads(
        (
            FIXTURES / "synthetic_schedule.json"
        ).read_text(encoding="utf-8")
    )[0]
    row["retrieval_timestamp"] = timestamp

    path = tmp_path / "bad-time.json"
    path.write_text(
        json.dumps([row]),
        encoding="utf-8",
    )

    with pytest.raises((ValueError, ValidationError)):
        JsonScheduleFileAdapter(path).games()


def test_as_of_after_retrieval_is_rejected(
    tmp_path: Path,
) -> None:
    row = json.loads(
        (
            FIXTURES / "synthetic_schedule.json"
        ).read_text(encoding="utf-8")
    )[0]
    row["retrieval_timestamp"] = (
        "2026-01-10T12:00:00-05:00"
    )
    row["as_of_timestamp"] = (
        "2026-01-10T12:01:00-05:00"
    )

    path = tmp_path / "future-as-of.json"
    path.write_text(
        json.dumps([row]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="as_of_timestamp",
    ):
        JsonScheduleFileAdapter(path).games()


def test_offer_missing_required_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    row = json.loads(
        (
            FIXTURES
            / "synthetic_sportsbook_sog_offers.json"
        ).read_text(encoding="utf-8")
    )[0]
    del row["retrieval_timestamp"]

    path = tmp_path / "missing-offer-time.json"
    path.write_text(
        json.dumps([row]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="retrieval_timestamp",
    ):
        JsonSportsbookOfferFileAdapter(
            path,
            synthetic_identity_map(),
        ).offers()


def test_unresolved_identity_is_not_guessed() -> None:
    identity_map = synthetic_identity_map()

    resolution = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-odds",
        source_entity_id="unknown-player",
    )

    assert (
        resolution.status
        is IdentityResolutionStatus.UNRESOLVED
    )
    assert resolution.canonical_id is None
    assert (
        DataQualityFlag.UNRESOLVED_IDENTITY
        in validate_identity_resolution(
            resolution
        ).flags
    )

    row = json.loads(
        (
            FIXTURES
            / "synthetic_sportsbook_sog_offers.json"
        ).read_text(encoding="utf-8")
    )[0]
    row["source_player_id"] = "unknown-player"

    with pytest.raises(
        ValueError,
        match="unresolved_identity",
    ):
        normalize_sportsbook_offer(
            row,
            identity_map,
        )


def test_ambiguous_identity_is_not_guessed_and_candidates_are_deterministic(
) -> None:
    mappings = [
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.GAME,
            source_name="synthetic-odds",
            source_entity_id="game-a",
            canonical_id="canonical-game-a",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.PLAYER,
            source_name="synthetic-odds",
            source_entity_id="player-a",
            canonical_id="canonical-player-z",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.PLAYER,
            source_name="synthetic-odds",
            source_entity_id="player-a",
            canonical_id="canonical-player-a",
        ),
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.SPORTSBOOK,
            source_name="synthetic-odds",
            source_entity_id="book-a",
            canonical_id="canonical-book-a",
        ),
    ]
    identity_map = CanonicalIdMap(mappings)

    resolution = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-odds",
        source_entity_id="player-a",
    )

    assert (
        resolution.status
        is IdentityResolutionStatus.AMBIGUOUS
    )
    assert resolution.canonical_id is None
    assert resolution.candidate_canonical_ids == (
        "canonical-player-a",
        "canonical-player-z",
    )
    assert (
        DataQualityFlag.AMBIGUOUS_IDENTITY
        in validate_identity_resolution(
            resolution
        ).flags
    )

    row = json.loads(
        (
            FIXTURES
            / "synthetic_sportsbook_sog_offers.json"
        ).read_text(encoding="utf-8")
    )[0]

    with pytest.raises(
        ValueError,
        match="ambiguous_identity",
    ):
        normalize_sportsbook_offer(
            row,
            identity_map,
        )


def test_offer_as_of_after_retrieval_follows_existing_offer_contract(
) -> None:
    row = json.loads(
        (
            FIXTURES
            / "synthetic_sportsbook_sog_offers.json"
        ).read_text(encoding="utf-8")
    )[0]
    row["retrieval_timestamp"] = (
        "2026-01-10T12:00:00-05:00"
    )
    row["as_of_timestamp"] = (
        "2026-01-10T12:01:00-05:00"
    )

    offer = normalize_sportsbook_offer(
        row,
        synthetic_identity_map(),
    )

    assert (
        offer.as_of_timestamp
        > offer.retrieval_timestamp
    )


def test_offer_side_and_line_are_exactly_typed() -> None:
    offers = JsonSportsbookOfferFileAdapter(
        FIXTURES / "synthetic_sportsbook_sog_offers.json",
        synthetic_identity_map(),
    ).offers()

    assert offers[0].side is MarketSide.OVER
    assert offers[0].line == 2.5

    integer_offer = next(
        offer
        for offer in offers
        if offer.record_id == "offer-integer-over"
    )
    assert integer_offer.line == 3.0