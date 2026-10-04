"""Pure normalization helpers for local NHL CSV/JSON file adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from nhl_betting_models.common.schemas import (
    MarketSide,
    MarketStatus,
    Offer,
)
from nhl_betting_models.data.base import (
    AvailabilityStatus,
    ConfirmationStatus,
    NormalizedAvailability,
    NormalizedFinalSOGOutcome,
    NormalizedGame,
    NormalizedGoalieStatus,
    NormalizedOpponentSOGProfile,
    NormalizedPlayerGameStateStats,
    NormalizedRoleProjection,
)
from nhl_betting_models.data.canonical_ids import (
    CanonicalEntityType,
    CanonicalIdMap,
    IdentityResolutionStatus,
)
from nhl_betting_models.data.provenance import SourceProvenance
from nhl_betting_models.data.quality import validate_identity_resolution
from nhl_betting_models.nhl.schemas import PlayerPosition, SOGGameState

Row = Mapping[str, Any]


def _required(row: Row, key: str) -> Any:
    if key not in row or row[key] is None or str(row[key]).strip() == "":
        raise ValueError(f"missing required field: {key}")
    return row[key]


def _optional(row: Row, key: str) -> Any | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return value


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value

    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc


def _optional_datetime(value: Any | None) -> datetime | None:
    return None if value is None else _datetime(value)


def _provenance(row: Row) -> SourceProvenance:
    source_record_id = _optional(row, "source_record_id")

    return SourceProvenance(
        source_name=str(_required(row, "source_name")),
        source_record_id=(
            None if source_record_id is None else str(source_record_id)
        ),
        raw_payload_record_id=str(
            _required(row, "raw_payload_record_id")
        ),
        source_timestamp=_optional_datetime(
            _optional(row, "source_timestamp")
        ),
        retrieval_timestamp=_datetime(
            _required(row, "retrieval_timestamp")
        ),
        as_of_timestamp=_datetime(
            _required(row, "as_of_timestamp")
        ),
    )


def _optional_int(row: Row, key: str) -> int | None:
    value = _optional(row, key)
    return None if value is None else int(value)


def _optional_float(row: Row, key: str) -> float | None:
    value = _optional(row, key)
    return None if value is None else float(value)


def _resolve_required(
    identity_map: CanonicalIdMap,
    *,
    entity_type: CanonicalEntityType,
    source_name: str,
    source_entity_id: str,
) -> str:
    resolution = identity_map.resolve(
        entity_type=entity_type,
        source_name=source_name,
        source_entity_id=source_entity_id,
    )
    quality = validate_identity_resolution(resolution)

    if resolution.status is not IdentityResolutionStatus.RESOLVED:
        flag = quality.flags[0].value
        candidates = ",".join(resolution.candidate_canonical_ids)
        suffix = f" candidates={candidates}" if candidates else ""
        raise ValueError(
            f"{flag}: {entity_type.value} {source_entity_id}{suffix}"
        )

    assert resolution.canonical_id is not None
    return resolution.canonical_id


def _stable_offer_record_id(row: Row) -> str:
    supplied = _optional(row, "record_id")
    if supplied is not None:
        return str(supplied)

    payload = json.dumps(
        dict(row),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"offer-{digest[:24]}"


def normalize_game(row: Row) -> NormalizedGame:
    return NormalizedGame(
        source_game_id=str(_required(row, "source_game_id")),
        source_home_team_id=str(
            _required(row, "source_home_team_id")
        ),
        source_away_team_id=str(
            _required(row, "source_away_team_id")
        ),
        game_start_timestamp=_datetime(
            _required(row, "game_start_timestamp")
        ),
        provenance=_provenance(row),
    )


def normalize_player_game_state_stats(
    row: Row,
) -> NormalizedPlayerGameStateStats:
    return NormalizedPlayerGameStateStats(
        source_game_id=str(_required(row, "source_game_id")),
        source_player_id=str(_required(row, "source_player_id")),
        source_team_id=str(_required(row, "source_team_id")),
        position=PlayerPosition(str(_required(row, "position"))),
        game_state=SOGGameState(
            str(_required(row, "game_state"))
        ),
        shots_on_goal=int(_required(row, "shots_on_goal")),
        exposure_minutes=float(
            _required(row, "exposure_minutes")
        ),
        provenance=_provenance(row),
    )


def normalize_role_projection(
    row: Row,
) -> NormalizedRoleProjection:
    return NormalizedRoleProjection(
        source_game_id=str(_required(row, "source_game_id")),
        source_player_id=str(_required(row, "source_player_id")),
        source_team_id=str(_required(row, "source_team_id")),
        projected_line=_optional_int(row, "projected_line"),
        power_play_unit=_optional_int(row, "power_play_unit"),
        five_on_five_minutes=_optional_float(
            row,
            "five_on_five_minutes",
        ),
        power_play_minutes=_optional_float(
            row,
            "power_play_minutes",
        ),
        short_handed_minutes=_optional_float(
            row,
            "short_handed_minutes",
        ),
        confirmation_status=ConfirmationStatus(
            str(_required(row, "confirmation_status"))
        ),
        provenance=_provenance(row),
    )


def normalize_goalie_status(row: Row) -> NormalizedGoalieStatus:
    goalie = _optional(row, "source_goalie_id")

    return NormalizedGoalieStatus(
        source_game_id=str(_required(row, "source_game_id")),
        source_team_id=str(_required(row, "source_team_id")),
        source_goalie_id=None if goalie is None else str(goalie),
        confirmation_status=ConfirmationStatus(
            str(_required(row, "confirmation_status"))
        ),
        provenance=_provenance(row),
    )


def normalize_availability(row: Row) -> NormalizedAvailability:
    game = _optional(row, "source_game_id")

    return NormalizedAvailability(
        source_player_id=str(_required(row, "source_player_id")),
        source_team_id=str(_required(row, "source_team_id")),
        source_game_id=None if game is None else str(game),
        availability_status=AvailabilityStatus(
            str(_required(row, "availability_status"))
        ),
        confirmation_status=ConfirmationStatus(
            str(_required(row, "confirmation_status"))
        ),
        provenance=_provenance(row),
    )


def normalize_opponent_sog_profile(
    row: Row,
) -> NormalizedOpponentSOGProfile:
    return NormalizedOpponentSOGProfile(
        source_team_id=str(_required(row, "source_team_id")),
        opponent_position=PlayerPosition(
            str(_required(row, "opponent_position"))
        ),
        game_state=SOGGameState(
            str(_required(row, "game_state"))
        ),
        sog_allowed_per_60=float(
            _required(row, "sog_allowed_per_60")
        ),
        exposure_minutes=float(
            _required(row, "exposure_minutes")
        ),
        provenance=_provenance(row),
    )


def normalize_sportsbook_offer(
    row: Row,
    identity_map: CanonicalIdMap,
) -> Offer:
    source_name = str(_required(row, "source_name"))

    canonical_event_id = _resolve_required(
        identity_map,
        entity_type=CanonicalEntityType.GAME,
        source_name=source_name,
        source_entity_id=str(
            _required(row, "source_game_id")
        ),
    )
    canonical_subject_id = _resolve_required(
        identity_map,
        entity_type=CanonicalEntityType.PLAYER,
        source_name=source_name,
        source_entity_id=str(
            _required(row, "source_player_id")
        ),
    )
    canonical_sportsbook_id = _resolve_required(
        identity_map,
        entity_type=CanonicalEntityType.SPORTSBOOK,
        source_name=source_name,
        source_entity_id=str(
            _required(row, "source_sportsbook_id")
        ),
    )

    source_record_id = _optional(row, "source_record_id")
    data_hash = _optional(row, "data_hash")

    return Offer(
        record_id=_stable_offer_record_id(row),
        source_name=source_name,
        sportsbook=canonical_sportsbook_id,
        source_record_id=(
            None
            if source_record_id is None
            else str(source_record_id)
        ),
        canonical_event_id=canonical_event_id,
        canonical_subject_id=canonical_subject_id,
        market_key=str(_required(row, "market_key")),
        side=MarketSide(str(_required(row, "side"))),
        line=float(_required(row, "line")),
        settlement_rule_id=str(
            _required(row, "settlement_rule_id")
        ),
        american_odds=int(_required(row, "american_odds")),
        decimal_odds=float(_required(row, "decimal_odds")),
        retrieval_timestamp=_datetime(
            _required(row, "retrieval_timestamp")
        ),
        source_timestamp=_optional_datetime(
            _optional(row, "source_timestamp")
        ),
        as_of_timestamp=_datetime(
            _required(row, "as_of_timestamp")
        ),
        status=MarketStatus(
            str(row.get("status", MarketStatus.OPEN.value))
        ),
        is_stale=str(
            row.get("is_stale", "false")
        ).lower() in {"1", "true", "yes"},
        quality_flags=tuple(row.get("quality_flags", ())),
        data_hash=None if data_hash is None else str(data_hash),
    )


def normalize_final_sog_outcome(
    row: Row,
) -> NormalizedFinalSOGOutcome:
    return NormalizedFinalSOGOutcome(
        source_game_id=str(_required(row, "source_game_id")),
        source_player_id=str(_required(row, "source_player_id")),
        final_sog=int(_required(row, "final_sog")),
        confirmation_status=ConfirmationStatus(
            str(_required(row, "confirmation_status"))
        ),
        provenance=_provenance(row),
    )