"""Local UTF-8 CSV/JSON adapters for provider-neutral NHL records."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nhl_betting_models.common.schemas import Offer
from nhl_betting_models.data.base import (
    NormalizedAvailability,
    NormalizedFinalSOGOutcome,
    NormalizedGame,
    NormalizedGoalieStatus,
    NormalizedOpponentSOGProfile,
    NormalizedPlayerGameStateStats,
    NormalizedRoleProjection,
)
from nhl_betting_models.data.canonical_ids import CanonicalIdMap
from nhl_betting_models.data.normalizers import (
    normalize_availability,
    normalize_final_sog_outcome,
    normalize_game,
    normalize_goalie_status,
    normalize_opponent_sog_profile,
    normalize_player_game_state_stats,
    normalize_role_projection,
    normalize_sportsbook_offer,
)


def _json_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("JSON fixture must contain a top-level array")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("JSON fixture rows must be objects")

    return payload


def _csv_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV fixture must contain a header row")
        return list(reader)


def _normalize_rows[T](
        rows: list[Mapping[str, Any]],
        normalizer: Callable[[Mapping[str, Any]], T],
    ) -> tuple[T, ...]:
        return tuple(normalizer(row) for row in rows)


class JsonScheduleFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def games(self) -> tuple[NormalizedGame, ...]:
        return _normalize_rows(_json_rows(self._path), normalize_game)


class CsvPlayerGameStatsFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def player_game_stats(
        self,
    ) -> tuple[NormalizedPlayerGameStateStats, ...]:
        return _normalize_rows(
            _csv_rows(self._path),
            normalize_player_game_state_stats,
        )


class CsvRoleProjectionFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def role_projections(self) -> tuple[NormalizedRoleProjection, ...]:
        return _normalize_rows(
            _csv_rows(self._path),
            normalize_role_projection,
        )


class JsonStartingGoalieFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def starting_goalies(self) -> tuple[NormalizedGoalieStatus, ...]:
        return _normalize_rows(
            _json_rows(self._path),
            normalize_goalie_status,
        )


class JsonAvailabilityFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def availability(self) -> tuple[NormalizedAvailability, ...]:
        return _normalize_rows(
            _json_rows(self._path),
            normalize_availability,
        )


class CsvOpponentSOGProfileFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def opponent_sog_profiles(
        self,
    ) -> tuple[NormalizedOpponentSOGProfile, ...]:
        return _normalize_rows(
            _csv_rows(self._path),
            normalize_opponent_sog_profile,
        )


class JsonSportsbookOfferFileAdapter:
    def __init__(
        self,
        path: Path,
        identity_map: CanonicalIdMap,
    ) -> None:
        self._path = Path(path)
        self._identity_map = identity_map

    def offers(self) -> tuple[Offer, ...]:
        return tuple(
            normalize_sportsbook_offer(row, self._identity_map)
            for row in _json_rows(self._path)
        )


class CsvFinalSOGOutcomeFileAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def final_sog_outcomes(
        self,
    ) -> tuple[NormalizedFinalSOGOutcome, ...]:
        return _normalize_rows(
            _csv_rows(self._path),
            normalize_final_sog_outcome,
        )


class JsonClosingOddsFileAdapter:
    def __init__(
        self,
        path: Path,
        identity_map: CanonicalIdMap,
    ) -> None:
        self._path = Path(path)
        self._identity_map = identity_map

    def closing_offers(self) -> tuple[Offer, ...]:
        return tuple(
            normalize_sportsbook_offer(row, self._identity_map)
            for row in _json_rows(self._path)
        )