"""Synthetic tests for immutable Part 3A ingestion persistence."""

from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from nhl_betting_models.data.ingestion_runs import (
    IngestionRunFinalization,
    IngestionRunStart,
    IngestionRunStatus,
    RecordTypeCount,
    finalize_ingestion_run,
    get_ingestion_run,
    raw_payload_ids_for_run,
    start_ingestion_run,
)
from nhl_betting_models.data.persistence import (
    RawPayloadInput,
    RawPayloadKind,
    get_raw_payload,
    initialize_ingestion_persistence,
    persist_raw_payload,
)
from nhl_betting_models.data.quality import (
    DataQualityFlag,
    DataQualityIssue,
    DataQualitySeverity,
)

START = datetime(2026, 1, 1, 17, tzinfo=UTC)
END = datetime(2026, 1, 1, 17, 1, tzinfo=UTC)


def start_run(
    database: Path,
    *,
    run_id: str,
) -> None:
    """Create one deterministic synthetic ingestion run."""

    start_ingestion_run(
        database,
        IngestionRunStart(
            run_id=run_id,
            run_started_at=START,
            input_names=(
                Path("fixtures/synthetic_schedule.json"),
            ),
            code_version="synthetic-test-code-v1",
            config_version="synthetic-test-config-v1",
        ),
    )


def raw_input(
    *,
    run_id: str,
    payload: str | bytes = (
        '{"game":"synthetic-game-1"}'
    ),
    retrieval_timestamp: datetime = START,
    as_of_timestamp: datetime = START,
) -> RawPayloadInput:
    """Build synthetic raw-payload metadata without fixture I/O."""

    return RawPayloadInput(
        source_name="synthetic-source",
        source_record_id="synthetic-source-record-1",
        logical_input_path=Path(
            "fixtures/synthetic_schedule.json"
        ),
        media_type="application/json",
        payload=payload,
        source_timestamp=START - timedelta(minutes=1),
        retrieval_timestamp=retrieval_timestamp,
        as_of_timestamp=as_of_timestamp,
        ingestion_run_id=run_id,
    )


def test_initialization_is_safe_and_repeatable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"

    initialize_ingestion_persistence(database)
    initialize_ingestion_persistence(database)

    with duckdb.connect(str(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SHOW TABLES"
            ).fetchall()
        }

    assert tables == {
        "ingestion_raw_payloads",
        "ingestion_run_payloads",
        "ingestion_runs",
    }


def test_new_raw_payload_persists_hash_and_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )
    request = raw_input(
        run_id="synthetic-run-1"
    )

    result = persist_raw_payload(
        database,
        request,
    )

    assert result.inserted
    assert isinstance(request.payload, str)
    assert result.record.content_hash == sha256(
        request.payload.encode("utf-8")
    ).hexdigest()
    assert result.record.source_name == request.source_name
    assert (
        result.record.source_record_id
        == request.source_record_id
    )
    assert (
        result.record.logical_input_path
        == request.logical_input_path.as_posix()
    )
    assert result.record.media_type == request.media_type
    assert (
        result.record.payload_kind
        is RawPayloadKind.TEXT
    )
    assert (
        result.record.raw_payload_text
        == request.payload
    )
    assert result.record.raw_payload_bytes is None
    assert (
        result.record.first_ingestion_run_id
        == "synthetic-run-1"
    )
    assert raw_payload_ids_for_run(
        database,
        "synthetic-run-1",
    ) == (result.record.record_id,)


def test_exact_duplicate_payload_is_deduplicated_across_runs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )
    start_run(
        database,
        run_id="synthetic-run-2",
    )

    first = persist_raw_payload(
        database,
        raw_input(run_id="synthetic-run-1"),
    )
    second = persist_raw_payload(
        database,
        raw_input(run_id="synthetic-run-2"),
    )

    assert first.inserted
    assert not second.inserted
    assert second.record == first.record

    assert raw_payload_ids_for_run(
        database,
        "synthetic-run-1",
    ) == (first.record.record_id,)

    assert raw_payload_ids_for_run(
        database,
        "synthetic-run-2",
    ) == (first.record.record_id,)

    with duckdb.connect(str(database)) as connection:
        count = connection.execute(
            """
            SELECT count(*)
            FROM ingestion_raw_payloads
            """
        ).fetchone()

    assert count == (1,)


def test_changed_content_creates_new_immutable_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )
    start_run(
        database,
        run_id="synthetic-run-2",
    )

    first = persist_raw_payload(
        database,
        raw_input(
            run_id="synthetic-run-1",
            payload='{"version":1}',
        ),
    )
    second = persist_raw_payload(
        database,
        raw_input(
            run_id="synthetic-run-2",
            payload='{"version":2}',
        ),
    )

    assert (
        first.record.record_id
        != second.record.record_id
    )
    assert (
        first.record.content_hash
        != second.record.content_hash
    )
    assert (
        get_raw_payload(
            database,
            first.record.record_id,
        )
        == first.record
    )
    assert (
        get_raw_payload(
            database,
            second.record.record_id,
        )
        == second.record
    )


def test_bytes_payload_and_provenance_timestamps_round_trip(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )

    eastern = timezone(timedelta(hours=-5))
    retrieval = datetime(
        2026,
        1,
        1,
        12,
        tzinfo=eastern,
    )
    as_of = datetime(
        2026,
        1,
        1,
        11,
        59,
        tzinfo=eastern,
    )

    result = persist_raw_payload(
        database,
        raw_input(
            run_id="synthetic-run-1",
            payload=b"synthetic,csv\n1,2\n",
            retrieval_timestamp=retrieval,
            as_of_timestamp=as_of,
        ),
    )

    persisted = get_raw_payload(
        database,
        result.record.record_id,
    )

    assert persisted is not None
    assert (
        persisted.payload_kind
        is RawPayloadKind.BYTES
    )
    assert (
        persisted.raw_payload_bytes
        == b"synthetic,csv\n1,2\n"
    )
    assert persisted.raw_payload_text is None
    assert (
        persisted.retrieval_timestamp
        == retrieval.astimezone(UTC)
    )
    assert (
        persisted.as_of_timestamp
        == as_of.astimezone(UTC)
    )
    assert (
        persisted.source_timestamp
        == START - timedelta(minutes=1)
    )


def test_run_finalization_persists_counts_quality_and_errors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )

    issue = DataQualityIssue(
        flag=DataQualityFlag.UNRESOLVED_IDENTITY,
        severity=DataQualitySeverity.ERROR,
        message="synthetic unresolved player identity",
        related_record_ids=(
            "synthetic-source-record-1",
        ),
    )

    finalization = IngestionRunFinalization(
        run_ended_at=END,
        status=IngestionRunStatus.FAILED,
        record_counts=(
            RecordTypeCount(
                record_type="game",
                count=3,
            ),
        ),
        accepted_count=1,
        rejected_count=1,
        flagged_count=1,
        quality_issues=(issue,),
        error_summaries=(
            "synthetic parser failure",
        ),
    )

    finalized = finalize_ingestion_run(
        database,
        "synthetic-run-1",
        finalization,
    )
    persisted = get_ingestion_run(
        database,
        "synthetic-run-1",
    )

    assert persisted == finalized
    assert finalized.run_started_at == START
    assert finalized.run_ended_at == END
    assert (
        finalized.status
        is IngestionRunStatus.FAILED
    )
    assert finalized.record_counts == (
        RecordTypeCount(
            record_type="game",
            count=3,
        ),
    )
    assert finalized.accepted_count == 1
    assert finalized.rejected_count == 1
    assert finalized.flagged_count == 1
    assert finalized.quality_issues == (issue,)
    assert finalized.error_summaries == (
        "synthetic parser failure",
    )
    assert (
        finalized.code_version
        == "synthetic-test-code-v1"
    )
    assert (
        finalized.config_version
        == "synthetic-test-config-v1"
    )


def test_exact_finalization_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )

    finalization = IngestionRunFinalization(
        run_ended_at=END,
        status=IngestionRunStatus.SUCCEEDED,
        accepted_count=1,
    )

    first = finalize_ingestion_run(
        database,
        "synthetic-run-1",
        finalization,
    )
    second = finalize_ingestion_run(
        database,
        "synthetic-run-1",
        finalization,
    )

    assert second == first


def test_conflicting_second_finalization_is_rejected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    start_run(
        database,
        run_id="synthetic-run-1",
    )

    finalize_ingestion_run(
        database,
        "synthetic-run-1",
        IngestionRunFinalization(
            run_ended_at=END,
            status=IngestionRunStatus.SUCCEEDED,
        ),
    )

    with pytest.raises(
        ValueError,
        match="already finalized",
    ):
        finalize_ingestion_run(
            database,
            "synthetic-run-1",
            IngestionRunFinalization(
                run_ended_at=END,
                status=IngestionRunStatus.FAILED,
                error_summaries=(
                    "different result",
                ),
            ),
        )


def test_duplicate_insertion_and_initialization_are_deterministic(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"

    initialize_ingestion_persistence(database)
    start_run(
        database,
        run_id="synthetic-run-1",
    )

    first = persist_raw_payload(
        database,
        raw_input(run_id="synthetic-run-1"),
    )

    initialize_ingestion_persistence(database)

    second = persist_raw_payload(
        database,
        raw_input(run_id="synthetic-run-1"),
    )

    assert first.record == second.record
    assert first.inserted
    assert not second.inserted
    assert raw_payload_ids_for_run(
        database,
        "synthetic-run-1",
    ) == (first.record.record_id,)


def test_unknown_run_cannot_receive_raw_payload(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"
    initialize_ingestion_persistence(database)

    with pytest.raises(
        ValueError,
        match="unknown ingestion run",
    ):
        persist_raw_payload(
            database,
            raw_input(run_id="missing-run"),
        )


@pytest.mark.parametrize(
    (
        "retrieval_timestamp",
        "as_of_timestamp",
    ),
    (
        (
            datetime(2026, 1, 1, 17),
            START,
        ),
        (
            START,
            datetime(2026, 1, 1, 17),
        ),
        (
            START,
            START + timedelta(seconds=1),
        ),
    ),
)
def test_invalid_raw_payload_timestamps_are_rejected(
    retrieval_timestamp: datetime,
    as_of_timestamp: datetime,
) -> None:
    with pytest.raises(ValidationError):
        raw_input(
            run_id="synthetic-run-1",
            retrieval_timestamp=retrieval_timestamp,
            as_of_timestamp=as_of_timestamp,
        )


def test_malformed_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RawPayloadInput.model_validate(
            {
                "source_name": "synthetic-source",
                "source_record_id": (
                    "synthetic-record-1"
                ),
                "logical_input_path": (
                    "fixtures/synthetic.json"
                ),
                "media_type": "application/json",
                "payload": "{}",
                "retrieval_timestamp": (
                    "not-a-timestamp"
                ),
                "as_of_timestamp": START,
                "ingestion_run_id": (
                    "synthetic-run-1"
                ),
            }
        )


def test_run_timestamps_follow_existing_validation_rules(
    tmp_path: Path,
) -> None:
    database = tmp_path / "part3a.duckdb"

    with pytest.raises(ValidationError):
        IngestionRunStart(
            run_id="synthetic-run-1",
            run_started_at=datetime(
                2026,
                1,
                1,
                17,
            ),
            input_names=(
                Path("fixtures/synthetic.json"),
            ),
        )

    start_run(
        database,
        run_id="synthetic-run-1",
    )

    with pytest.raises(
        ValueError,
        match="earlier than run_started_at",
    ):
        finalize_ingestion_run(
            database,
            "synthetic-run-1",
            IngestionRunFinalization(
                run_ended_at=(
                    START - timedelta(seconds=1)
                ),
                status=IngestionRunStatus.FAILED,
            ),
        )