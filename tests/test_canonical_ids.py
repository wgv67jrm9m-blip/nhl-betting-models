"""Synthetic tests for provider-neutral canonical identity resolution."""

import pytest

from nhl_betting_models.data.canonical_ids import (
    CanonicalEntityType,
    CanonicalIdMap,
    CanonicalIdMapping,
    IdentityResolutionStatus,
)
from nhl_betting_models.data.quality import (
    DataQualityFlag,
    validate_identity_resolution,
)


@pytest.mark.parametrize(
    ("entity_type", "source_id", "canonical_id"),
    (
        (
            CanonicalEntityType.GAME,
            "synthetic-source-game-1",
            "synthetic-game-1",
        ),
        (
            CanonicalEntityType.PLAYER,
            "synthetic-source-player-1",
            "synthetic-player-1",
        ),
        (
            CanonicalEntityType.TEAM,
            "synthetic-source-team-1",
            "synthetic-team-1",
        ),
        (
            CanonicalEntityType.SPORTSBOOK,
            "synthetic-source-book-1",
            "synthetic-book-1",
        ),
    ),
)
def test_exact_provider_id_resolves_to_canonical_id(
    entity_type: CanonicalEntityType,
    source_id: str,
    canonical_id: str,
) -> None:
    """SYNTHETIC TEST FIXTURE: no real provider identifiers."""

    identity_map = CanonicalIdMap(
        (
            CanonicalIdMapping(
                entity_type=entity_type,
                source_name="synthetic-source",
                source_entity_id=source_id,
                canonical_id=canonical_id,
            ),
        )
    )

    result = identity_map.resolve(
        entity_type=entity_type,
        source_name="synthetic-source",
        source_entity_id=source_id,
    )

    assert result.status is IdentityResolutionStatus.RESOLVED
    assert result.canonical_id == canonical_id
    assert not result.candidate_canonical_ids


def test_unknown_provider_id_remains_unresolved() -> None:
    """SYNTHETIC TEST FIXTURE: unknown IDs are never guessed."""

    identity_map = CanonicalIdMap(())

    result = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-source",
        source_entity_id="unknown-player",
    )

    assert result.status is IdentityResolutionStatus.UNRESOLVED
    assert result.canonical_id is None

    quality = validate_identity_resolution(result)

    assert not quality.is_valid
    assert DataQualityFlag.UNRESOLVED_IDENTITY in quality.flags


def test_ambiguous_provider_id_is_not_resolved() -> None:
    """SYNTHETIC TEST FIXTURE: ambiguous identity must remain blocked."""

    identity_map = CanonicalIdMap(
        (
            CanonicalIdMapping(
                entity_type=CanonicalEntityType.PLAYER,
                source_name="synthetic-source",
                source_entity_id="ambiguous-player",
                canonical_id="synthetic-player-1",
            ),
            CanonicalIdMapping(
                entity_type=CanonicalEntityType.PLAYER,
                source_name="synthetic-source",
                source_entity_id="ambiguous-player",
                canonical_id="synthetic-player-2",
            ),
        )
    )

    result = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-source",
        source_entity_id="ambiguous-player",
    )

    assert result.status is IdentityResolutionStatus.AMBIGUOUS
    assert result.canonical_id is None
    assert result.candidate_canonical_ids == (
        "synthetic-player-1",
        "synthetic-player-2",
    )

    quality = validate_identity_resolution(result)

    assert not quality.is_valid
    assert DataQualityFlag.AMBIGUOUS_IDENTITY in quality.flags


def test_entity_types_do_not_cross_resolve() -> None:
    """SYNTHETIC TEST FIXTURE: provider IDs are scoped by entity type."""

    identity_map = CanonicalIdMap(
        (
            CanonicalIdMapping(
                entity_type=CanonicalEntityType.TEAM,
                source_name="synthetic-source",
                source_entity_id="shared-id",
                canonical_id="synthetic-team-1",
            ),
        )
    )

    result = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-source",
        source_entity_id="shared-id",
    )

    assert result.status is IdentityResolutionStatus.UNRESOLVED


def test_provider_namespaces_do_not_cross_resolve() -> None:
    """SYNTHETIC TEST FIXTURE: IDs are scoped to their source."""

    identity_map = CanonicalIdMap(
        (
            CanonicalIdMapping(
                entity_type=CanonicalEntityType.PLAYER,
                source_name="synthetic-source-a",
                source_entity_id="player-1",
                canonical_id="synthetic-player-1",
            ),
        )
    )

    result = identity_map.resolve(
        entity_type=CanonicalEntityType.PLAYER,
        source_name="synthetic-source-b",
        source_entity_id="player-1",
    )

    assert result.status is IdentityResolutionStatus.UNRESOLVED


def test_blank_mapping_identifier_is_rejected() -> None:
    """SYNTHETIC TEST FIXTURE: malformed mappings fail explicitly."""

    with pytest.raises(
        ValueError,
        match="identity mapping identifiers must not be empty",
    ):
        CanonicalIdMapping(
            entity_type=CanonicalEntityType.PLAYER,
            source_name="synthetic-source",
            source_entity_id=" ",
            canonical_id="synthetic-player-1",
        )