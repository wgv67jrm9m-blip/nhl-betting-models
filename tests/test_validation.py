"""Synthetic tests for forecast-time validation."""

from datetime import UTC, datetime, timedelta

import pytest

from nhl_betting_models.common.validation import (
    reject_post_forecast_features,
    walk_forward_splits,
)


def test_walk_forward_split_integrity() -> None:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    timestamps = [
        start + timedelta(days=index)
        for index in range(8)
    ]

    splits = walk_forward_splits(
        timestamps,
        min_train_size=4,
        test_size=2,
    )

    assert len(splits) == 2

    for split in splits:
        assert split.train_end < split.test_start
        assert (
            max(split.train_indices)
            < min(split.test_indices)
        )


def test_walk_forward_rejects_duplicate_boundary() -> None:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    timestamps = [
        start,
        start + timedelta(days=1),
        start + timedelta(days=1),
        start + timedelta(days=2),
    ]

    with pytest.raises(
        ValueError,
        match=r"max\(train\)",
    ):
        walk_forward_splits(
            timestamps,
            min_train_size=2,
            test_size=1,
        )


def test_walk_forward_rejects_unsorted_timestamps() -> None:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    timestamps = [
        start,
        start + timedelta(days=2),
        start + timedelta(days=1),
        start + timedelta(days=3),
    ]

    with pytest.raises(
        ValueError,
        match="sorted",
    ):
        walk_forward_splits(
            timestamps,
            min_train_size=2,
            test_size=1,
        )


def test_reject_post_forecast_feature_timestamps() -> None:
    run = datetime(
        2026,
        1,
        1,
        18,
        tzinfo=UTC,
    )

    reject_post_forecast_features(
        [
            run - timedelta(minutes=1),
            run,
        ],
        run,
    )

    with pytest.raises(
        ValueError,
        match="newer",
    ):
        reject_post_forecast_features(
            [
                run + timedelta(seconds=1),
            ],
            run,
        )