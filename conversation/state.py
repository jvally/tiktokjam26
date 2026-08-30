from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})


@dataclass(frozen=True)
class BudgetRange:
    """Normalized customer budget bounds.

    Either bound may be absent. ``target`` is retained for "around $X" requests so
    ranking can prefer products near the requested price rather than treating it as
    a strict upper limit.
    """

    minimum: float | None = None
    maximum: float | None = None
    target: float | None = None

    def __post_init__(self) -> None:
        for value in (self.minimum, self.maximum, self.target):
            if value is not None and value < 0:
                raise ValueError("Budget values must be non-negative")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Budget minimum cannot exceed maximum")


@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: str | float | BudgetRange
    phrase: str
    kind: str = "hard"
    turn: int = 1
    confidence: float = 1.0
    intent_version: int = 0

    def __post_init__(self) -> None:
        if self.attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"Unsupported constraint attribute: {self.attribute}")
        if self.kind not in {"hard", "soft", "negative"}:
            raise ValueError("Constraint kind must be hard, soft, or negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Constraint confidence must be between zero and one")


@dataclass
class ConversationState:
    user_profile: dict[str, Any]
    turn: int = 0
    intent: str = "browsing"
    category: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    asked_counts: dict[str, int] = field(default_factory=dict)
    exhausted_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    boundary_seen: bool = False
    messages: list[str] = field(default_factory=list)
    retracted_constraints: list[Constraint] = field(default_factory=list)
    intent_version: int = 0
    override_count: int = 0
    rejected_result_sets: int = 0
    recommendation_history: list[tuple[int, int, tuple[str, ...]]] = field(default_factory=list)

    @property
    def query(self) -> str:
        parts = [self.category]
        parts.extend(item.phrase for item in self.constraints if item.kind != "negative")
        unique: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = " ".join(str(part).split()).strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return " ".join(unique)

    def known_attributes(self) -> set[str]:
        return {item.attribute for item in self.constraints if item.kind != "negative"}

    def note_question(self, attribute: str | None) -> None:
        self.last_asked_attribute = attribute
        if attribute:
            self.asked_counts[attribute] = self.asked_counts.get(attribute, 0) + 1

    def note_recommendations(self, identifiers: list[str]) -> None:
        unique = tuple(dict.fromkeys(str(item) for item in identifiers if str(item)))
        if unique:
            self.recommendation_history.append((self.intent_version, self.turn, unique))

    def shown_ids(self) -> set[str]:
        """Products already rejected implicitly during the current shopping intent."""

        return {
            identifier
            for version, _, identifiers in self.recommendation_history
            if version == self.intent_version
            for identifier in identifiers
        }

    def replace(self, attribute: str, kind: str | None = None) -> None:
        retained: list[Constraint] = []
        for item in self.constraints:
            remove = item.attribute == attribute and (kind is None or item.kind == kind)
            if remove:
                self.retracted_constraints.append(item)
            else:
                retained.append(item)
        self.constraints = retained

    def clear_soft_constraints(self) -> None:
        for item in self.constraints:
            if item.kind == "soft":
                self.retracted_constraints.append(item)
        self.constraints = [item for item in self.constraints if item.kind != "soft"]

    def clear_active_constraints(self) -> None:
        self.retracted_constraints.extend(self.constraints)
        self.constraints = []

    def begin_override(self) -> None:
        self.intent_version += 1
        self.override_count += 1
        self.clear_soft_constraints()

    def conflicting_attributes(self) -> set[str]:
        positive = {
            (item.attribute, str(item.value).casefold())
            for item in self.constraints if item.kind != "negative"
        }
        negative = {
            (item.attribute, str(item.value).casefold())
            for item in self.constraints if item.kind == "negative"
        }
        return {attribute for attribute, _ in positive & negative}

    def add(self, constraint: Constraint, replace: bool = False) -> None:
        if replace:
            self.replace(constraint.attribute, "hard")
            self.replace(constraint.attribute, "soft")
        key = (constraint.attribute, str(constraint.value).casefold(), constraint.kind)
        existing = {(item.attribute, str(item.value).casefold(), item.kind) for item in self.constraints}
        if key not in existing:
            self.constraints.append(constraint)
