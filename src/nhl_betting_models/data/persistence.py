"""Immutable local persistence for synthetic ingestion payloads."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import duckdb
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

PART3A_DDL = r"""
CREATE TABLE IF NOT EXISTS ingestion_raw_payloads (
    record_id VARCHAR PRIMARY KEY,
    content_hash VARCHAR NOT NULL,
    source_name VARCHAR NOT NULL,
    source_record_id VARCHAR,
    logical_input_path VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL,
    payload_kind VARCHAR NOT NULL,
    raw_payload_text VARCHAR,
    raw_payload_bytes BLOB,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ NOT NULL,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    first_ingestion_run_id VARCHAR NOT NULL,
    created_at TIMESTAMPTZ DEFAULT current_timestamp,
    CHECK (payload_kind IN ('text', 'bytes')),
    CHECK (
        (
            payload_kind = 'text'
            AND raw_payload_text IS NOT NULL
            AND raw_payload_bytes IS NULL
        )
        OR
        (
            payload_kind = 'bytes'
            AND raw_payload_text IS NULL
            AND raw_payload_bytes IS NOT NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id VARCHAR PRIMARY KEY,
    run_started_at TIMESTAMPTZ NOT NULL,
    run_ended_at TIMESTAMPTZ,
    run_status VARCHAR NOT NULL,
    input_names_json JSON NOT NULL,
    record_counts_json JSON NOT NULL,
    accepted_count INTEGER NOT NULL,
    rejected_count INTEGER NOT NULL,
    flagged_count INTEGER NOT NULL,
    quality_issues_json JSON NOT NULL,
    error_summaries_json JSON NOT NULL,
    code_version VARCHAR,
    config_version VARCHAR,
    created_at TIMESTAMPTZ DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ingestion_run_payloads (
    run_id VARCHAR NOT NULL,
    raw_payload_record_id VARCHAR NOT NULL,
    PRIMARY KEY (run_id, raw_payload_record_id)
);
"""


def _utc(value: datetime) -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class RawPayloadKind(StrEnum):
    """Stored representation for one immutable raw payload."""

    TEXT = "text"
    BYTES = "bytes"


class RawPayloadInput(BaseModel):
    """Immutable request to persist one local synthetic raw payload."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    source_record_id: str | None = None
    logical_input_path: Path
    media_type: str
    payload: str | bytes
    source_timestamp: datetime | None = None
    retrieval_timestamp: datetime
    as_of_timestamp: datetime
    ingestion_run_id: str

    @field_validator("source_name", "media_type", "ingestion_run_id")
    @classmethod
    def require_nonempty_identifier(cls, value: str) -> str:
        """Require identifiers needed for raw-payload auditability."""

        if not value.strip():
            raise ValueError("raw payload identifier must not be empty")
        return value

    @field_validator("source_record_id")
    @classmethod
    def validate_optional_source_record_id(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject an explicitly supplied empty source record identifier."""

        if value is not None and not value.strip():
            raise ValueError("source_record_id must not be empty")
        return value

    @field_validator("logical_input_path")
    @classmethod
    def require_logical_input_path(cls, value: Path) -> Path:
        """Require a nonempty logical input path without performing I/O."""

        if not str(value).strip() or str(value) == ".":
            raise ValueError("logical_input_path must not be empty")
        return value

    @field_validator(
        "source_timestamp",
        "retrieval_timestamp",
        "as_of_timestamp",
    )
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Require supplied timestamps to be aware and normalize them to UTC."""

        if value is None:
            return None
        return _utc(value)

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> RawPayloadInput:
        """Require information to be effective no later than retrieval."""

        if self.as_of_timestamp > self.retrieval_timestamp:
            raise ValueError(
                "as_of_timestamp must not be later than retrieval_timestamp"
            )
        return self


class RawPayloadRecord(BaseModel):
    """Immutable persisted raw-payload record."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    content_hash: str
    source_name: str
    source_record_id: str | None = None
    logical_input_path: str
    media_type: str
    payload_kind: RawPayloadKind
    raw_payload_text: str | None = None
    raw_payload_bytes: bytes | None = None
    source_timestamp: datetime | None = None
    retrieval_timestamp: datetime
    as_of_timestamp: datetime
    first_ingestion_run_id: str

    @field_validator(
        "source_timestamp",
        "retrieval_timestamp",
        "as_of_timestamp",
    )
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Normalize timestamps returned by DuckDB to UTC."""

        if value is None:
            return None
        return _utc(value)


class PersistRawPayloadResult(BaseModel):
    """Immutable result of idempotent raw-payload persistence."""

    model_config = ConfigDict(frozen=True)

    record: RawPayloadRecord
    inserted: bool


def initialize_ingestion_persistence(path: str | Path) -> None:
    """Create Part 3A tables safely without modifying shared tables."""

    with duckdb.connect(str(path)) as connection:
        connection.execute(PART3A_DDL)


def _payload_bytes(payload: str | bytes) -> bytes:
    """Return the exact bytes used for content hashing."""

    if isinstance(payload, str):
        return payload.encode("utf-8")
    return payload


def _content_hash(payload: str | bytes) -> str:
    """Return the deterministic SHA-256 hash of exact payload content."""

    return hashlib.sha256(_payload_bytes(payload)).hexdigest()


def _raw_payload_record_id(
    payload: RawPayloadInput,
    content_hash: str,
) -> str:
    """Create deterministic identity for equivalent immutable payloads."""

    identity = json.dumps(
        {
            "source_name": payload.source_name,
            "source_record_id": payload.source_record_id,
            "logical_input_path": payload.logical_input_path.as_posix(),
            "media_type": payload.media_type,
            "content_hash": content_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return f"raw_{hashlib.sha256(identity).hexdigest()}"


_RAW_COLUMNS = """
record_id,
content_hash,
source_name,
source_record_id,
logical_input_path,
media_type,
payload_kind,
raw_payload_text,
raw_payload_bytes,
source_timestamp,
retrieval_timestamp,
as_of_timestamp,
first_ingestion_run_id
"""


def _row_to_raw_payload(
    row: tuple[object, ...],
) -> RawPayloadRecord:
    """Convert one DuckDB row to the immutable public record."""

    return RawPayloadRecord(
        record_id=cast(str, row[0]),
        content_hash=cast(str, row[1]),
        source_name=cast(str, row[2]),
        source_record_id=cast(str | None, row[3]),
        logical_input_path=cast(str, row[4]),
        media_type=cast(str, row[5]),
        payload_kind=RawPayloadKind(cast(str, row[6])),
        raw_payload_text=cast(str | None, row[7]),
        raw_payload_bytes=cast(bytes | None, row[8]),
        source_timestamp=cast(datetime | None, row[9]),
        retrieval_timestamp=cast(datetime, row[10]),
        as_of_timestamp=cast(datetime, row[11]),
        first_ingestion_run_id=cast(str, row[12]),
    )


def get_raw_payload(
    path: str | Path,
    record_id: str,
) -> RawPayloadRecord | None:
    """Return one immutable raw payload by deterministic record ID."""

    with duckdb.connect(str(path)) as connection:
        row = connection.execute(
            f"""
            SELECT {_RAW_COLUMNS}
            FROM ingestion_raw_payloads
            WHERE record_id = ?
            """,
            [record_id],
        ).fetchone()

    if row is None:
        return None

    return _row_to_raw_payload(cast(tuple[object, ...], row))


def persist_raw_payload(
    path: str | Path,
    payload: RawPayloadInput,
) -> PersistRawPayloadResult:
    """Persist an immutable raw payload and link it to its ingestion run."""

    content_hash = _content_hash(payload.payload)
    record_id = _raw_payload_record_id(payload, content_hash)

    if isinstance(payload.payload, str):
        payload_kind = RawPayloadKind.TEXT
        raw_text: str | None = payload.payload
        raw_bytes: bytes | None = None
    else:
        payload_kind = RawPayloadKind.BYTES
        raw_text = None
        raw_bytes = payload.payload

    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")

        try:
            run_exists = connection.execute(
                """
                SELECT 1
                FROM ingestion_runs
                WHERE run_id = ?
                """,
                [payload.ingestion_run_id],
            ).fetchone()

            if run_exists is None:
                raise ValueError(
                    f"unknown ingestion run: {payload.ingestion_run_id}"
                )

            existing = connection.execute(
                f"""
                SELECT {_RAW_COLUMNS}
                FROM ingestion_raw_payloads
                WHERE record_id = ?
                """,
                [record_id],
            ).fetchone()

            inserted = existing is None

            if inserted:
                connection.execute(
                    """
                    INSERT INTO ingestion_raw_payloads (
                        record_id,
                        content_hash,
                        source_name,
                        source_record_id,
                        logical_input_path,
                        media_type,
                        payload_kind,
                        raw_payload_text,
                        raw_payload_bytes,
                        source_timestamp,
                        retrieval_timestamp,
                        as_of_timestamp,
                        first_ingestion_run_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record_id,
                        content_hash,
                        payload.source_name,
                        payload.source_record_id,
                        payload.logical_input_path.as_posix(),
                        payload.media_type,
                        payload_kind.value,
                        raw_text,
                        raw_bytes,
                        payload.source_timestamp,
                        payload.retrieval_timestamp,
                        payload.as_of_timestamp,
                        payload.ingestion_run_id,
                    ],
                )

            connection.execute(
                """
                INSERT INTO ingestion_run_payloads (
                    run_id,
                    raw_payload_record_id
                )
                VALUES (?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    payload.ingestion_run_id,
                    record_id,
                ],
            )

            row = connection.execute(
                f"""
                SELECT {_RAW_COLUMNS}
                FROM ingestion_raw_payloads
                WHERE record_id = ?
                """,
                [record_id],
            ).fetchone()

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    if row is None:
        raise RuntimeError(
            "persisted raw payload could not be read back"
        )

    return PersistRawPayloadResult(
        record=_row_to_raw_payload(
            cast(tuple[object, ...], row)
        ),
        inserted=inserted,
    )