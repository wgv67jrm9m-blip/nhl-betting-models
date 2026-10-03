"""Ingestion-run audit persistence for local synthetic fixture ingestion."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import uuid4

import duckdb
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
    model_validator,
)

from nhl_betting_models.data.persistence import (
    initialize_ingestion_persistence,
)
from nhl_betting_models.data.quality import DataQualityIssue


def _utc(value: datetime) -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class IngestionRunStatus(StrEnum):
    """Lifecycle status for one local ingestion execution."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RecordTypeCount(BaseModel):
    """Immutable count for one attempted or normalized record type."""

    model_config = ConfigDict(frozen=True)

    record_type: str
    count: int

    @field_validator("record_type")
    @classmethod
    def require_record_type(cls, value: str) -> str:
        """Require a nonempty record-type label."""

        if not value.strip():
            raise ValueError("record_type must not be empty")
        return value

    @field_validator("count")
    @classmethod
    def require_nonnegative_count(cls, value: int) -> int:
        """Reject negative record counts."""

        if value < 0:
            raise ValueError("record count must not be negative")
        return value


class IngestionRunStart(BaseModel):
    """Immutable request to create an ingestion-run audit record."""

    model_config = ConfigDict(frozen=True)

    run_started_at: datetime
    input_names: tuple[Path, ...]
    run_id: str | None = None
    code_version: str | None = None
    config_version: str | None = None

    @field_validator("run_started_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require an aware run timestamp and normalize it to UTC."""

        return _utc(value)

    @field_validator(
        "run_id",
        "code_version",
        "config_version",
    )
    @classmethod
    def reject_blank_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject explicitly supplied blank audit identifiers."""

        if value is not None and not value.strip():
            raise ValueError("audit identifier must not be empty")
        return value

    @field_validator("input_names")
    @classmethod
    def require_input_names(
        cls,
        value: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        """Require at least one nonempty logical input path."""

        if not value:
            raise ValueError("input_names must not be empty")

        if any(
            not str(item).strip() or str(item) == "."
            for item in value
        ):
            raise ValueError("input name must not be empty")

        return value


class IngestionRunFinalization(BaseModel):
    """Immutable final audit outcome for an ingestion run."""

    model_config = ConfigDict(frozen=True)

    run_ended_at: datetime
    status: IngestionRunStatus
    record_counts: tuple[RecordTypeCount, ...] = ()
    accepted_count: int = 0
    rejected_count: int = 0
    flagged_count: int = 0
    quality_issues: tuple[DataQualityIssue, ...] = ()
    error_summaries: tuple[str, ...] = ()

    @field_validator("run_ended_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require an aware run timestamp and normalize it to UTC."""

        return _utc(value)

    @field_validator(
        "accepted_count",
        "rejected_count",
        "flagged_count",
    )
    @classmethod
    def require_nonnegative_count(cls, value: int) -> int:
        """Reject negative audit counts."""

        if value < 0:
            raise ValueError("audit count must not be negative")
        return value

    @field_validator("error_summaries")
    @classmethod
    def reject_blank_errors(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Reject blank persisted error summaries."""

        if any(not item.strip() for item in value):
            raise ValueError("error summary must not be empty")
        return value

    @model_validator(mode="after")
    def require_final_status(
        self,
    ) -> IngestionRunFinalization:
        """Prevent finalization back to the running state."""

        if self.status is IngestionRunStatus.RUNNING:
            raise ValueError(
                "finalized ingestion run cannot have running status"
            )
        return self


class IngestionRunRecord(BaseModel):
    """Immutable persisted ingestion-run audit record."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_started_at: datetime
    run_ended_at: datetime | None = None
    status: IngestionRunStatus
    input_names: tuple[str, ...]
    record_counts: tuple[RecordTypeCount, ...] = ()
    accepted_count: int = 0
    rejected_count: int = 0
    flagged_count: int = 0
    quality_issues: tuple[DataQualityIssue, ...] = ()
    error_summaries: tuple[str, ...] = ()
    code_version: str | None = None
    config_version: str | None = None

    @field_validator("run_started_at", "run_ended_at")
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Normalize timestamps returned by DuckDB to UTC."""

        if value is None:
            return None
        return _utc(value)


_RECORD_COUNTS_ADAPTER = TypeAdapter(
    tuple[RecordTypeCount, ...]
)
_QUALITY_ISSUES_ADAPTER = TypeAdapter(
    tuple[DataQualityIssue, ...]
)
_STRING_TUPLE_ADAPTER = TypeAdapter(tuple[str, ...])


def _json_tuple(values: tuple[str, ...]) -> str:
    """Serialize an immutable string tuple deterministically."""

    return json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _record_counts_json(
    values: tuple[RecordTypeCount, ...],
) -> str:
    """Serialize typed record counts."""

    return _RECORD_COUNTS_ADAPTER.dump_json(
        values
    ).decode("utf-8")


def _quality_issues_json(
    values: tuple[DataQualityIssue, ...],
) -> str:
    """Serialize existing typed quality issues without alteration."""

    return _QUALITY_ISSUES_ADAPTER.dump_json(
        values
    ).decode("utf-8")


_RUN_COLUMNS = """
run_id,
run_started_at,
run_ended_at,
run_status,
input_names_json,
record_counts_json,
accepted_count,
rejected_count,
flagged_count,
quality_issues_json,
error_summaries_json,
code_version,
config_version
"""


def _row_to_run(
    row: tuple[object, ...],
) -> IngestionRunRecord:
    """Convert one DuckDB row to the immutable public run type."""

    return IngestionRunRecord(
        run_id=cast(str, row[0]),
        run_started_at=cast(datetime, row[1]),
        run_ended_at=cast(datetime | None, row[2]),
        status=IngestionRunStatus(cast(str, row[3])),
        input_names=_STRING_TUPLE_ADAPTER.validate_json(
            cast(str, row[4])
        ),
        record_counts=_RECORD_COUNTS_ADAPTER.validate_json(
            cast(str, row[5])
        ),
        accepted_count=cast(int, row[6]),
        rejected_count=cast(int, row[7]),
        flagged_count=cast(int, row[8]),
        quality_issues=_QUALITY_ISSUES_ADAPTER.validate_json(
            cast(str, row[9])
        ),
        error_summaries=_STRING_TUPLE_ADAPTER.validate_json(
            cast(str, row[10])
        ),
        code_version=cast(str | None, row[11]),
        config_version=cast(str | None, row[12]),
    )


def start_ingestion_run(
    path: str | Path,
    request: IngestionRunStart,
) -> IngestionRunRecord:
    """Create one auditable ingestion run."""

    initialize_ingestion_persistence(path)

    run_id = request.run_id or f"ingest_{uuid4().hex}"
    input_names = tuple(
        item.as_posix()
        for item in request.input_names
    )

    with duckdb.connect(str(path)) as connection:
        try:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id,
                    run_started_at,
                    run_status,
                    input_names_json,
                    record_counts_json,
                    accepted_count,
                    rejected_count,
                    flagged_count,
                    quality_issues_json,
                    error_summaries_json,
                    code_version,
                    config_version
                )
                VALUES (?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    request.run_started_at,
                    IngestionRunStatus.RUNNING.value,
                    _json_tuple(input_names),
                    _record_counts_json(()),
                    _quality_issues_json(()),
                    _json_tuple(()),
                    request.code_version,
                    request.config_version,
                ],
            )
        except duckdb.ConstraintException as exc:
            raise ValueError(
                f"ingestion run already exists: {run_id}"
            ) from exc

    record = get_ingestion_run(path, run_id)

    if record is None:
        raise RuntimeError(
            "created ingestion run could not be read back"
        )

    return record


def get_ingestion_run(
    path: str | Path,
    run_id: str,
) -> IngestionRunRecord | None:
    """Return one persisted ingestion-run audit record."""

    with duckdb.connect(str(path)) as connection:
        row = connection.execute(
            f"""
            SELECT {_RUN_COLUMNS}
            FROM ingestion_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

    if row is None:
        return None

    return _row_to_run(cast(tuple[object, ...], row))


def finalize_ingestion_run(
    path: str | Path,
    run_id: str,
    finalization: IngestionRunFinalization,
) -> IngestionRunRecord:
    """Finalize a run once while permitting exact idempotent retries."""

    current = get_ingestion_run(path, run_id)

    if current is None:
        raise ValueError(f"unknown ingestion run: {run_id}")

    if finalization.run_ended_at < current.run_started_at:
        raise ValueError(
            "run_ended_at must not be earlier than run_started_at"
        )

    desired = IngestionRunRecord(
        run_id=current.run_id,
        run_started_at=current.run_started_at,
        run_ended_at=finalization.run_ended_at,
        status=finalization.status,
        input_names=current.input_names,
        record_counts=finalization.record_counts,
        accepted_count=finalization.accepted_count,
        rejected_count=finalization.rejected_count,
        flagged_count=finalization.flagged_count,
        quality_issues=finalization.quality_issues,
        error_summaries=finalization.error_summaries,
        code_version=current.code_version,
        config_version=current.config_version,
    )

    if current.status is not IngestionRunStatus.RUNNING:
        if current == desired:
            return current

        raise ValueError(
            f"ingestion run is already finalized: {run_id}"
        )

    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            UPDATE ingestion_runs
            SET
                run_ended_at = ?,
                run_status = ?,
                record_counts_json = ?,
                accepted_count = ?,
                rejected_count = ?,
                flagged_count = ?,
                quality_issues_json = ?,
                error_summaries_json = ?
            WHERE run_id = ?
              AND run_status = ?
            """,
            [
                finalization.run_ended_at,
                finalization.status.value,
                _record_counts_json(
                    finalization.record_counts
                ),
                finalization.accepted_count,
                finalization.rejected_count,
                finalization.flagged_count,
                _quality_issues_json(
                    finalization.quality_issues
                ),
                _json_tuple(
                    finalization.error_summaries
                ),
                run_id,
                IngestionRunStatus.RUNNING.value,
            ],
        )

    persisted = get_ingestion_run(path, run_id)

    if persisted is None:
        raise RuntimeError(
            "finalized ingestion run could not be read back"
        )

    if persisted != desired:
        raise RuntimeError(
            "ingestion run finalization did not persist "
            "deterministically"
        )

    return persisted


def raw_payload_ids_for_run(
    path: str | Path,
    run_id: str,
) -> tuple[str, ...]:
    """Return ordered raw payload IDs linked to one ingestion run."""

    with duckdb.connect(str(path)) as connection:
        rows = connection.execute(
            """
            SELECT raw_payload_record_id
            FROM ingestion_run_payloads
            WHERE run_id = ?
            ORDER BY raw_payload_record_id
            """,
            [run_id],
        ).fetchall()

    return tuple(
        cast(str, row[0])
        for row in rows
    )