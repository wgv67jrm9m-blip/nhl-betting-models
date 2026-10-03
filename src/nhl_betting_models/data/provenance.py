"""Provider-neutral provenance metadata for normalized NHL data."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


def _utc(value: datetime) -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class SourceProvenance(BaseModel):
    """Immutable provenance for one normalized provider record."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    source_record_id: str | None = None
    raw_payload_record_id: str
    source_timestamp: datetime | None = None
    retrieval_timestamp: datetime
    as_of_timestamp: datetime

    @field_validator("source_name", "raw_payload_record_id")
    @classmethod
    def require_nonempty_identifier(cls, value: str) -> str:
        """Require identifiers needed for source auditability."""

        if not value.strip():
            raise ValueError("provenance identifier must not be empty")
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
    def validate_temporal_consistency(self) -> SourceProvenance:
        """Require information to be effective no later than retrieval."""

        if self.as_of_timestamp > self.retrieval_timestamp:
            raise ValueError(
                "as_of_timestamp must not be later than retrieval_timestamp"
            )
        return self