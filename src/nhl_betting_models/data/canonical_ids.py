"""Canonical identity mapping for provider-neutral NHL data."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class CanonicalEntityType(StrEnum):
    """Entity types supported by canonical identity resolution."""

    GAME = "game"
    PLAYER = "player"
    TEAM = "team"
    SPORTSBOOK = "sportsbook"


class IdentityResolutionStatus(StrEnum):
    """Outcome of resolving one provider identifier."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class CanonicalIdMapping(BaseModel):
    """One immutable provider-to-canonical identifier mapping."""

    model_config = ConfigDict(frozen=True)

    entity_type: CanonicalEntityType
    source_name: str
    source_entity_id: str
    canonical_id: str

    @model_validator(mode="after")
    def validate_identifiers(self) -> CanonicalIdMapping:
        """Reject blank mapping identifiers."""

        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.source_entity_id.strip():
            raise ValueError("source_entity_id must not be empty")
        if not self.canonical_id.strip():
            raise ValueError("canonical_id must not be empty")
        return self


class IdentityResolution(BaseModel):
    """Structured immutable result of one canonical identity lookup."""

    model_config = ConfigDict(frozen=True)

    entity_type: CanonicalEntityType
    source_name: str
    source_entity_id: str
    status: IdentityResolutionStatus
    canonical_id: str | None = None
    candidate_canonical_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_resolution(self) -> IdentityResolution:
        """Require status, selected ID, and candidates to agree."""

        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.source_entity_id.strip():
            raise ValueError("source_entity_id must not be empty")

        if self.status is IdentityResolutionStatus.RESOLVED:
            if self.canonical_id is None or not self.canonical_id.strip():
                raise ValueError(
                    "resolved identity requires canonical_id"
                )
            if self.candidate_canonical_ids:
                raise ValueError(
                    "resolved identity must not contain candidates"
                )
            return self

        if self.canonical_id is not None:
            raise ValueError(
                "unresolved or ambiguous identity must not select "
                "canonical_id"
            )

        if self.status is IdentityResolutionStatus.UNRESOLVED:
            if self.candidate_canonical_ids:
                raise ValueError(
                    "unresolved identity must not contain candidates"
                )
            return self

        if len(self.candidate_canonical_ids) < 2:
            raise ValueError(
                "ambiguous identity requires at least two candidates"
            )

        return self


class CanonicalIdMap:
    """Lookup index over explicit provider-to-canonical mappings."""

    def __init__(
        self,
        mappings: Iterable[CanonicalIdMapping],
    ) -> None:
        indexed: dict[
            tuple[CanonicalEntityType, str, str],
            set[str],
        ] = {}

        for mapping in mappings:
            key = (
                mapping.entity_type,
                mapping.source_name,
                mapping.source_entity_id,
            )
            indexed.setdefault(key, set()).add(mapping.canonical_id)

        self._mappings = {
            key: tuple(sorted(canonical_ids))
            for key, canonical_ids in indexed.items()
        }

    def resolve(
        self,
        *,
        entity_type: CanonicalEntityType,
        source_name: str,
        source_entity_id: str,
    ) -> IdentityResolution:
        """Resolve an explicit provider identifier without guessing."""

        candidates = self._mappings.get(
            (
                entity_type,
                source_name,
                source_entity_id,
            ),
            (),
        )

        if not candidates:
            return IdentityResolution(
                entity_type=entity_type,
                source_name=source_name,
                source_entity_id=source_entity_id,
                status=IdentityResolutionStatus.UNRESOLVED,
            )

        if len(candidates) > 1:
            return IdentityResolution(
                entity_type=entity_type,
                source_name=source_name,
                source_entity_id=source_entity_id,
                status=IdentityResolutionStatus.AMBIGUOUS,
                candidate_canonical_ids=candidates,
            )

        return IdentityResolution(
            entity_type=entity_type,
            source_name=source_name,
            source_entity_id=source_entity_id,
            status=IdentityResolutionStatus.RESOLVED,
            canonical_id=candidates[0],
        )