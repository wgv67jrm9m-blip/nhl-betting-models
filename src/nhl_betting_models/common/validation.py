"""Point-in-time validation and walk-forward splitting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .schemas import WalkForwardSplit


def _validate_timestamps(
    timestamps: Sequence[datetime],
) -> None:
    if not timestamps:
        raise ValueError(
            "Timestamps must not be empty",
        )

    for timestamp in timestamps:
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError(
                "All timestamps must be timezone-aware",
            )


def walk_forward_splits(
    timestamps: Sequence[datetime],
    *,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create expanding-window time-series splits.

    Input records must already be ordered by forecast timestamp. Every fold
    enforces max(training timestamp) < min(test timestamp). Random splitting
    is intentionally unsupported.
    """

    _validate_timestamps(timestamps)

    if min_train_size < 1 or test_size < 1:
        raise ValueError(
            "min_train_size and test_size must be positive",
        )

    step = (
        test_size
        if step_size is None
        else step_size
    )

    if step < 1:
        raise ValueError(
            "step_size must be positive",
        )

    if any(
        left > right
        for left, right in zip(
            timestamps,
            timestamps[1:],
            strict=False,
        )
    ):
        raise ValueError(
            "Timestamps must be sorted ascending",
        )

    splits: list[WalkForwardSplit] = []
    test_start_index = min_train_size

    while (
        test_start_index + test_size
        <= len(timestamps)
    ):
        train = tuple(
            range(0, test_start_index)
        )
        test = tuple(
            range(
                test_start_index,
                test_start_index + test_size,
            )
        )

        train_end = timestamps[train[-1]]
        test_start = timestamps[test[0]]

        if train_end >= test_start:
            raise ValueError(
                "Walk-forward split requires "
                "max(train) < min(test)",
            )

        splits.append(
            WalkForwardSplit(
                train_indices=train,
                test_indices=test,
                train_end=train_end,
                test_start=test_start,
            )
        )

        test_start_index += step

    if not splits:
        raise ValueError(
            "Not enough observations for "
            "the requested walk-forward split",
        )

    return splits


def reject_post_forecast_features(
    feature_timestamps: Sequence[datetime],
    model_run_timestamp: datetime,
) -> None:
    """Reject any feature observation newer than forecast execution."""

    _validate_timestamps(feature_timestamps)

    if (
        model_run_timestamp.tzinfo is None
        or model_run_timestamp.utcoffset() is None
    ):
        raise ValueError(
            "Model-run timestamp must be timezone-aware",
        )

    newer = [
        timestamp
        for timestamp in feature_timestamps
        if timestamp > model_run_timestamp
    ]

    if newer:
        raise ValueError(
            f"Detected {len(newer)} feature record(s) "
            "newer than model-run timestamp",
        )