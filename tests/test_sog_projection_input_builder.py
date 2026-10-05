"""Tests for canonical NHL SOG projection-input construction."""

from __future__ import annotations

from datetime import datetime

from nhl_betting_models.data.slate import (
    assemble_validated_sog_slate,
)
from nhl_betting_models.nhl.schemas import (
    PlayerPosition,
    SOGGameState,
    SOGQualityFlag,
    SOGRatePrior,
)
from nhl_betting_models.nhl.sog_projection_candidates import (
    build_sog_projection_candidates,
)
from nhl_betting_models.nhl.sog_projection_input_builder import (
    build_sog_projection_inputs,
)
from tests.test_slate import (
    synthetic_identity_map,
    synthetic_slate_paths,
)


def test_build_projection_inputs_for_available_candidate() -> None:
    """Convert an eligible source-linked candidate into model input."""

    assembly_result = assemble_validated_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )
    candidate_results = build_sog_projection_candidates(
        assembly_result
    )
    priors = (
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.FIVE_ON_FIVE,
            mean_sog_per_60=8.0,
            prior_equivalent_minutes=120.0,
            prior_version="synthetic-v1",
        ),
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.POWER_PLAY,
            mean_sog_per_60=10.0,
            prior_equivalent_minutes=60.0,
            prior_version="synthetic-v1",
        ),
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.SHORT_HANDED,
            mean_sog_per_60=2.0,
            prior_equivalent_minutes=60.0,
            prior_version="synthetic-v1",
        ),
    )

    inputs = build_sog_projection_inputs(
        candidate_results,
        synthetic_identity_map(),
        priors,
        model_run_timestamp=datetime.fromisoformat(
    "2026-01-11T04:00:00+00:00"
),
    )

    assert len(inputs) == 1

    projection_input = inputs[0]
    assert projection_input.canonical_event_id == "canonical-game-a"
    assert projection_input.canonical_player_id == "canonical-player-a"
    assert projection_input.position is PlayerPosition.FORWARD
    assert projection_input.projected_minutes.five_on_five_minutes == 15.5
    assert projection_input.projected_minutes.power_play_minutes == 3.5
    assert projection_input.projected_minutes.short_handed_minutes == 0.5
    assert projection_input.projected_minutes.quality_flags == ()
    assert {
        rate.game_state for rate in projection_input.shrunk_rates
    } == {
        SOGGameState.FIVE_ON_FIVE,
        SOGGameState.POWER_PLAY,
    }


def test_projection_input_marks_projected_role() -> None:
    """Preserve projected-role uncertainty in the player-minute input."""

    assembly_result = assemble_validated_sog_slate(
        synthetic_slate_paths(),
        synthetic_identity_map(),
    )
    candidate_results = build_sog_projection_candidates(
        assembly_result
    )
    priors = (
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.FIVE_ON_FIVE,
            mean_sog_per_60=8.0,
            prior_equivalent_minutes=120.0,
            prior_version="synthetic-v1",
        ),
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.POWER_PLAY,
            mean_sog_per_60=10.0,
            prior_equivalent_minutes=60.0,
            prior_version="synthetic-v1",
        ),
        SOGRatePrior(
            position=PlayerPosition.FORWARD,
            game_state=SOGGameState.SHORT_HANDED,
            mean_sog_per_60=2.0,
            prior_equivalent_minutes=60.0,
            prior_version="synthetic-v1",
        ),
    )

    inputs = build_sog_projection_inputs(
        candidate_results,
        synthetic_identity_map(),
        priors,
        model_run_timestamp=datetime.fromisoformat(
            "2026-01-12T00:00:00+00:00"
        ),
    )

    assert len(inputs) == 1
    assert (
        SOGQualityFlag.PROJECTED_ROLE
        not in inputs[0].projected_minutes.quality_flags
    )