"""Configuration for shared market evaluation behavior."""

from pydantic import BaseModel, ConfigDict, Field


class EligibilityThresholds(BaseModel):
    """Configurable gates for manual-review eligibility."""

    model_config = ConfigDict(frozen=True)

    min_ev: float = Field(default=0.0)
    min_edge: float = Field(default=0.0)
    min_model_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_quality_flags: frozenset[str] = frozenset(
        {
            "invalid_odds",
            "missing_timestamp",
            "unmatched_player_id",
            "unmatched_game_id",
            "market_line_mismatch",
            "market_side_mismatch",
            "unknown_settlement_rule",
            "suspended_market",
            "incomplete_market",
        }
    )