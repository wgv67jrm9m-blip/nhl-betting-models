"""Synthetic tests for provider-neutral NHL data-quality validation."""

from datetime import UTC, datetime

import pytest

from nhl_betting_models.common.odds import american_to_decimal
from nhl_betting_models.common.schemas import (
    MarketSide,
    MarketStatus,
    Offer,
)
from nhl_betting_models.data.base import NormalizedGame
from nhl_betting_models.data.provenance import SourceProvenance
from nhl_betting_models.data.quality import (
    DataQualityFlag,
    validate_duplicate_offers,
    validate_game_start_times,
    validate_offer_odds,
    validate_offer_set,
    validate_player_identity,
    validate_stale_offer,
    validate_two_sided_market,
)

ODDS_TIME = datetime(2026, 1, 1, 17, tzinfo=UTC)
GAME_TIME = datetime(2026, 1, 1, 19, tzinfo=UTC)


def synthetic_provenance(
    *,
    raw_payload_record_id: str = "synthetic-raw-1",
) -> SourceProvenance:
    """SYNTHETIC TEST FIXTURE: no real provider provenance."""

    return SourceProvenance(
        source_name="synthetic-source",
        source_record_id="synthetic-source-record-1",
        retrieval_timestamp=ODDS_TIME,
        source_timestamp=ODDS_TIME,
        as_of_timestamp=ODDS_TIME,
        raw_payload_record_id=raw_payload_record_id,
    )


def synthetic_offer(
    *,
    record_id: str,
    side: MarketSide,
    american_odds: int,
    canonical_subject_id: str = "synthetic-player-1",
    line: float = 3.0,
    retrieval_timestamp: datetime = ODDS_TIME,
    is_stale: bool = False,
    source_record_id: str | None = None,
    decimal_odds: float | None = None,
) -> Offer:
    """SYNTHETIC TEST FIXTURE: no live sportsbook data."""

    resolved_decimal_odds = (
        american_to_decimal(american_odds)
        if decimal_odds is None
        else decimal_odds
    )

    return Offer(
        record_id=record_id,
        source_name="synthetic-odds-source",
        sportsbook="SyntheticBook",
        source_record_id=(
            source_record_id
            if source_record_id is not None
            else record_id
        ),
        canonical_event_id="synthetic-game-1",
        canonical_subject_id=canonical_subject_id,
        market_key="player_sog",
        side=side,
        line=line,
        settlement_rule_id="synthetic-sog-rule-v1",
        american_odds=american_odds,
        decimal_odds=resolved_decimal_odds,
        retrieval_timestamp=retrieval_timestamp,
        source_timestamp=retrieval_timestamp,
        as_of_timestamp=retrieval_timestamp,
        status=MarketStatus.OPEN,
        is_stale=is_stale,
    )


def test_stale_odds_are_flagged() -> None:
    offer = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
        is_stale=True,
    )

    result = validate_stale_offer(offer)

    assert not result.is_valid
    assert DataQualityFlag.STALE_ODDS in result.flags


@pytest.mark.parametrize(
    ("american_odds", "decimal_odds"),
    (
        (0, 2.0),
        (99, 1.99),
        (-99, 1.99),
        (100, 1.0),
        (100, float("inf")),
    ),
)
def test_invalid_odds_are_flagged(
    american_odds: int,
    decimal_odds: float,
) -> None:
    offer = synthetic_offer(
        record_id="synthetic-invalid",
        side=MarketSide.OVER,
        american_odds=american_odds,
        decimal_odds=decimal_odds,
    )

    result = validate_offer_odds(offer)

    assert not result.is_valid
    assert DataQualityFlag.INVALID_ODDS in result.flags


def test_complete_two_sided_market_passes() -> None:
    over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
    )
    under = synthetic_offer(
        record_id="synthetic-under",
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    result = validate_two_sided_market((over, under))

    assert result.is_valid
    assert not result.issues


def test_single_sided_market_is_flagged() -> None:
    over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
    )

    result = validate_two_sided_market((over,))

    assert not result.is_valid
    assert (
        DataQualityFlag.INCOMPLETE_TWO_SIDED_MARKET
        in result.flags
    )


def test_different_lines_do_not_form_two_sided_market() -> None:
    over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
        line=3.0,
    )
    under = synthetic_offer(
        record_id="synthetic-under",
        side=MarketSide.UNDER,
        american_odds=-120,
        line=4.0,
    )

    result = validate_two_sided_market((over, under))

    assert not result.is_valid
    assert len(result.issues) == 2


def test_different_retrieval_times_do_not_form_snapshot() -> None:
    over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
        retrieval_timestamp=datetime(
            2026,
            1,
            1,
            17,
            0,
            tzinfo=UTC,
        ),
    )
    under = synthetic_offer(
        record_id="synthetic-under",
        side=MarketSide.UNDER,
        american_odds=-120,
        retrieval_timestamp=datetime(
            2026,
            1,
            1,
            17,
            1,
            tzinfo=UTC,
        ),
    )

    result = validate_two_sided_market((over, under))

    assert not result.is_valid
    assert len(result.issues) == 2


def test_duplicate_offer_is_flagged() -> None:
    first = synthetic_offer(
        record_id="synthetic-record-a",
        source_record_id="provider-record-1",
        side=MarketSide.OVER,
        american_odds=100,
    )
    duplicate = synthetic_offer(
        record_id="synthetic-record-b",
        source_record_id="provider-record-1",
        side=MarketSide.OVER,
        american_odds=100,
    )

    result = validate_duplicate_offers((first, duplicate))

    assert not result.is_valid
    assert DataQualityFlag.DUPLICATE_OFFER in result.flags
    assert result.issues[0].related_record_ids == (
        "synthetic-record-a",
        "synthetic-record-b",
    )


def test_different_prices_are_not_duplicate_offers() -> None:
    first = synthetic_offer(
        record_id="synthetic-record-a",
        source_record_id="provider-record-1",
        side=MarketSide.OVER,
        american_odds=100,
    )
    second = synthetic_offer(
        record_id="synthetic-record-b",
        source_record_id="provider-record-2",
        side=MarketSide.OVER,
        american_odds=110,
    )

    result = validate_duplicate_offers((first, second))

    assert result.is_valid


def test_mismatched_player_identity_is_flagged() -> None:
    matching = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
    )
    mismatching = synthetic_offer(
        record_id="synthetic-under",
        side=MarketSide.UNDER,
        american_odds=-120,
        canonical_subject_id="synthetic-player-2",
    )

    result = validate_player_identity(
        expected_canonical_player_id="synthetic-player-1",
        offers=(matching, mismatching),
    )

    assert not result.is_valid
    assert (
        DataQualityFlag.MISMATCHED_PLAYER_IDENTITY
        in result.flags
    )
    assert result.issues[0].related_record_ids == (
        "synthetic-under",
    )


def test_conflicting_game_start_times_are_flagged() -> None:
    first = NormalizedGame(
        source_game_id="synthetic-source-game-1",
        source_home_team_id="synthetic-home",
        source_away_team_id="synthetic-away",
        game_start_timestamp=GAME_TIME,
        provenance=synthetic_provenance(
            raw_payload_record_id="synthetic-raw-1"
        ),
    )
    second = NormalizedGame(
        source_game_id="synthetic-source-game-1",
        source_home_team_id="synthetic-home",
        source_away_team_id="synthetic-away",
        game_start_timestamp=datetime(
            2026,
            1,
            1,
            19,
            30,
            tzinfo=UTC,
        ),
        provenance=synthetic_provenance(
            raw_payload_record_id="synthetic-raw-2"
        ),
    )

    result = validate_game_start_times((first, second))

    assert not result.is_valid
    assert (
        DataQualityFlag.MISMATCHED_GAME_START_TIME
        in result.flags
    )


def test_same_game_start_time_passes() -> None:
    first = NormalizedGame(
        source_game_id="synthetic-source-game-1",
        source_home_team_id="synthetic-home",
        source_away_team_id="synthetic-away",
        game_start_timestamp=GAME_TIME,
        provenance=synthetic_provenance(
            raw_payload_record_id="synthetic-raw-1"
        ),
    )
    second = NormalizedGame(
        source_game_id="synthetic-source-game-1",
        source_home_team_id="synthetic-home",
        source_away_team_id="synthetic-away",
        game_start_timestamp=GAME_TIME,
        provenance=synthetic_provenance(
            raw_payload_record_id="synthetic-raw-2"
        ),
    )

    result = validate_game_start_times((first, second))

    assert result.is_valid


def test_source_provenance_requires_timezone_aware_timestamps() -> None:
    with pytest.raises(
        ValueError,
        match="timestamp must be timezone-aware",
    ):
        SourceProvenance(
            source_name="synthetic-source",
            source_record_id="synthetic-record",
            retrieval_timestamp=datetime(2026, 1, 1, 17),
            source_timestamp=None,
            as_of_timestamp=ODDS_TIME,
            raw_payload_record_id="synthetic-raw",
        )


def test_source_provenance_requires_raw_payload_linkage() -> None:
    with pytest.raises(
        ValueError,
        match="provenance identifier must not be empty",
    ):
        SourceProvenance(
            source_name="synthetic-source",
            source_record_id="synthetic-record",
            retrieval_timestamp=ODDS_TIME,
            source_timestamp=ODDS_TIME,
            as_of_timestamp=ODDS_TIME,
            raw_payload_record_id=" ",
        )


def test_as_of_timestamp_cannot_follow_retrieval() -> None:
    with pytest.raises(
        ValueError,
        match="as_of_timestamp must not be later",
    ):
        SourceProvenance(
            source_name="synthetic-source",
            source_record_id="synthetic-record",
            retrieval_timestamp=datetime(
                2026,
                1,
                1,
                17,
                tzinfo=UTC,
            ),
            source_timestamp=None,
            as_of_timestamp=datetime(
                2026,
                1,
                1,
                18,
                tzinfo=UTC,
            ),
            raw_payload_record_id="synthetic-raw",
        )


def test_complete_valid_offer_set_passes_quality_validation() -> None:
    over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
    )
    under = synthetic_offer(
        record_id="synthetic-under",
        side=MarketSide.UNDER,
        american_odds=-120,
    )

    result = validate_offer_set(
        (over, under),
        expected_canonical_player_id="synthetic-player-1",
    )

    assert result.is_valid
    assert not result.issues


def test_offer_set_accumulates_independent_quality_failures() -> None:
    stale_over = synthetic_offer(
        record_id="synthetic-over",
        side=MarketSide.OVER,
        american_odds=100,
        is_stale=True,
        canonical_subject_id="synthetic-player-2",
    )

    result = validate_offer_set(
        (stale_over,),
        expected_canonical_player_id="synthetic-player-1",
    )

    assert not result.is_valid
    assert DataQualityFlag.STALE_ODDS in result.flags
    assert (
        DataQualityFlag.INCOMPLETE_TWO_SIDED_MARKET
        in result.flags
    )
    assert (
        DataQualityFlag.MISMATCHED_PLAYER_IDENTITY
        in result.flags
    )