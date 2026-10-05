"""Tests for source-linked NHL SOG projection candidates."""

from __future__ import annotations

from dataclasses import replace

from nhl_betting_models.data.quality import (
    DataQualityFlag,
    DataQualityIssue,
    DataQualityResult,
    DataQualitySeverity,
)
from nhl_betting_models.data.slate import (
    SOGSlateAssemblyResult,
    assemble_validated_sog_slate,
)
from nhl_betting_models.nhl.sog_projection_candidates import (
    SOGProjectionCandidateExclusionReason,
    build_sog_projection_candidates,
)
from tests.test_slate import (
    synthetic_identity_map,
    synthetic_slate_paths,
)


def test_build_candidates_keeps_available_player_and_excludes_injured_player(
) -> None:
    """Build a candidate for player-a and exclude injured player-b."""

    assembly_result = assemble_validated_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )

    candidates = build_sog_projection_candidates(assembly_result)

    assert len(candidates) == 2

    available_result = candidates[0]
    assert available_result.is_eligible
    assert available_result.exclusion_reason is None
    assert available_result.candidate is not None
    assert available_result.candidate.source_game_id == "game-a"
    assert available_result.candidate.source_player_id == "player-a"
    assert len(available_result.candidate.player_game_stats) == 2
    assert available_result.candidate.availability is not None
    assert (
        available_result.candidate.availability.availability_status.value
        == "available"
    )

    injured_result = candidates[1]
    assert injured_result.is_eligible is False
    assert injured_result.candidate is None
    assert injured_result.exclusion_reason is (
        SOGProjectionCandidateExclusionReason.UNAVAILABLE_PLAYER
    )


def test_build_candidates_rejects_invalid_slate() -> None:
    """Return explicit invalid-slate exclusions for invalid input."""

    valid_assembly_result = assemble_validated_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )
    invalid_quality = DataQualityResult(
        issues=(
            DataQualityIssue(
                flag=DataQualityFlag.MISSING_SLATE_GAME,
                severity=DataQualitySeverity.ERROR,
                message="synthetic invalid slate",
            ),
        )
    )
    invalid_assembly_result = replace(
        valid_assembly_result,
        quality=invalid_quality,
    )

    candidates = build_sog_projection_candidates(
        invalid_assembly_result
    )

    assert len(candidates) == 2
    assert all(
        candidate.is_eligible is False
        for candidate in candidates
    )
    assert {
        candidate.exclusion_reason
        for candidate in candidates
    } == {
        SOGProjectionCandidateExclusionReason.INVALID_SLATE,
    }