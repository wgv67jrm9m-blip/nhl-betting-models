"""Idempotent DuckDB schema initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

DDL = r"""
CREATE TABLE IF NOT EXISTS raw_source_payloads (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR NOT NULL,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR,
    canonical_subject_id VARCHAR,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ NOT NULL,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    payload_json JSON NOT NULL,
    data_hash VARCHAR NOT NULL,
    created_at TIMESTAMPTZ DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR NOT NULL,
    source_record_id VARCHAR,
    sportsbook VARCHAR NOT NULL,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR NOT NULL,
    market_key VARCHAR NOT NULL,
    market_side VARCHAR NOT NULL,
    line DOUBLE NOT NULL,
    settlement_rule_id VARCHAR NOT NULL,
    american_odds INTEGER NOT NULL,
    decimal_odds DOUBLE NOT NULL,
    implied_probability DOUBLE NOT NULL,
    market_status VARCHAR NOT NULL,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ NOT NULL,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    data_hash VARCHAR,
    quality_flags JSON,
    is_boosted BOOLEAN DEFAULT FALSE,
    is_promotion BOOLEAN DEFAULT FALSE,
    is_insured BOOLEAN DEFAULT FALSE,
    is_parlay_only BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR,
    model_version VARCHAR NOT NULL,
    feature_schema_version VARCHAR NOT NULL,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ NOT NULL,
    features_json JSON NOT NULL,
    input_manifest_json JSON NOT NULL,
    data_hash VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR,
    model_version VARCHAR NOT NULL,
    model_run_id VARCHAR NOT NULL,
    feature_snapshot_id VARCHAR NOT NULL,
    market_key VARCHAR,
    market_side VARCHAR,
    line DOUBLE,
    model_probability DOUBLE,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ NOT NULL,
    prediction_json JSON,
    data_hash VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS bet_candidates (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR,
    model_version VARCHAR NOT NULL,
    prediction_id VARCHAR NOT NULL,
    odds_snapshot_id VARCHAR NOT NULL,
    sportsbook VARCHAR NOT NULL,
    market_key VARCHAR NOT NULL,
    market_side VARCHAR NOT NULL,
    line DOUBLE NOT NULL,
    settlement_rule_id VARCHAR NOT NULL,
    model_probability DOUBLE NOT NULL,
    no_vig_market_probability DOUBLE,
    expected_value DOUBLE NOT NULL,
    edge DOUBLE,
    eligible BOOLEAN NOT NULL,
    threshold_failures JSON,
    input_quality_flags JSON,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ NOT NULL,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ NOT NULL,
    data_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS placed_bets (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR,
    model_version VARCHAR,
    candidate_id VARCHAR,
    prediction_id VARCHAR,
    sportsbook VARCHAR NOT NULL,
    market_key VARCHAR NOT NULL,
    market_side VARCHAR NOT NULL,
    line DOUBLE NOT NULL,
    settlement_rule_id VARCHAR NOT NULL,
    american_odds INTEGER NOT NULL,
    decimal_odds DOUBLE NOT NULL,
    stake DOUBLE NOT NULL,
    placed_timestamp TIMESTAMPTZ NOT NULL,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ,
    data_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS graded_bets (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR NOT NULL,
    canonical_subject_id VARCHAR,
    model_version VARCHAR,
    placed_bet_id VARCHAR NOT NULL,
    result VARCHAR NOT NULL,
    profit DOUBLE,
    returned_stake DOUBLE,
    settlement_value DOUBLE,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ,
    graded_timestamp TIMESTAMPTZ NOT NULL,
    data_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS model_runs (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR,
    canonical_subject_id VARCHAR,
    model_version VARCHAR NOT NULL,
    run_status VARCHAR NOT NULL,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ NOT NULL,
    config_json JSON NOT NULL,
    training_data_hash VARCHAR,
    code_version VARCHAR,
    data_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    record_id VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    source_record_id VARCHAR,
    canonical_event_id VARCHAR,
    canonical_subject_id VARCHAR,
    model_version VARCHAR,
    severity VARCHAR NOT NULL,
    quality_flag VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    related_record_id VARCHAR,
    source_timestamp TIMESTAMPTZ,
    retrieval_timestamp TIMESTAMPTZ,
    as_of_timestamp TIMESTAMPTZ NOT NULL,
    model_run_timestamp TIMESTAMPTZ,
    data_hash VARCHAR
);
"""


def initialize_database(path: str | Path) -> None:
    """Create all shared tables without modifying existing records."""

    with duckdb.connect(str(path)) as connection:
        connection.execute(DDL)


def main() -> None:
    """CLI entry point for local database initialization."""

    parser = argparse.ArgumentParser(
        description="Initialize NHL betting-model DuckDB tables.",
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    initialize_database(args.database)


if __name__ == "__main__":
    main()