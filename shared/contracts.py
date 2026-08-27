"""Validated, JSON-friendly contracts shared by all five workstreams."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

CONSTRAINT_KEYS = frozenset({
    "category", "brand", "color", "material", "size", "style", "feature", "use_case",
    "budget_min", "budget_max",
})
ALIASES = {"features": "feature", "use_cases": "use_case", "colors": "color",
           "materials": "material", "sizes": "size", "styles": "style"}


def number(value: Any, name: str, minimum: Optional[float] = None,
           maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def strings(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a string or a list of strings")
    return [v.strip() for v in value if v.strip()]


def constraints(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result = {}
    for key, raw in value.items():
        key = ALIASES.get(key, key)
        if key not in CONSTRAINT_KEYS:
            raise ValueError(f"Unsupported {name} key: {key}")
        if raw is None:
            continue
        if key.startswith("budget_"):
            result[key] = number(raw, key, minimum=0)
        else:
            values = strings(raw, key)
            if values:
                result[key] = values
    if result.get("budget_min", 0) > result.get("budget_max", math.inf):
        raise ValueError("budget_min cannot exceed budget_max")
    return result


@dataclass
class SearchState:
    query: str = ""
    intent: str = "buying"
    turn: int = 1
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    negative_constraints: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    asked_attributes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ValueError("query must be a string")
        if self.intent not in {"buying", "browsing"}:
            raise ValueError("intent must be buying or browsing")
        if isinstance(self.turn, bool) or not isinstance(self.turn, int) or not 1 <= self.turn <= 10:
            raise ValueError("turn must be an integer between 1 and 10")
        for name in ("hard_constraints", "soft_preferences", "negative_constraints", "profile"):
            setattr(self, name, constraints(getattr(self, name), name))
        if any(k.startswith("budget_") for k in self.negative_constraints):
            raise ValueError("Use hard budget_min/budget_max, not negative budget constraints")
        self.asked_attributes = strings(self.asked_attributes, "asked_attributes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchState:
        if not isinstance(value, Mapping):
            raise ValueError("state must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown state fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Product:
    parent_asin: str
    title: str
    category: str = ""
    brand: str = ""
    price: Optional[float] = None
    description: str = ""
    color: list[str] = field(default_factory=list)
    material: list[str] = field(default_factory=list)
    size: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("parent_asin", "title", "category", "brand", "description"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not self.parent_asin.strip() or not self.title.strip():
            raise ValueError("Products require a nonempty parent_asin and title")
        if self.price is not None:
            self.price = number(self.price, "price", minimum=0)
        for name in ("color", "material", "size", "style", "features", "use_cases"):
            setattr(self, name, strings(getattr(self, name), name))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Product:
        if not isinstance(value, Mapping):
            raise ValueError("product must be an object")
        for key in ("parent_asin", "title"):
            if key not in value:
                raise ValueError(f"Product is missing {key}")
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        return " ".join([self.title, self.category, self.brand, self.description,
                         *self.color, *self.material, *self.style, *self.features, *self.use_cases])


@dataclass
class Candidate:
    product: Product
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    metadata_score: Optional[float] = None
    profile_score: Optional[float] = None
    query_title_similarity: Optional[float] = None
    query_feature_similarity: Optional[float] = None
    use_case_similarity: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.product, Product):
            raise ValueError("candidate.product must be a Product")
        for name in self.__dataclass_fields__:
            if name == "product" or getattr(self, name) is None:
                continue
            maximum = None if name == "bm25_score" else 1
            setattr(self, name, number(getattr(self, name), name, minimum=0, maximum=maximum))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Candidate:
        if not isinstance(value, Mapping):
            raise ValueError("candidate must be an object")
        product = Product.from_dict(value.get("product", value))
        scores = {k: value[k] for k in cls.__dataclass_fields__ if k != "product" and k in value}
        return cls(product=product, **scores)

    def to_dict(self) -> dict[str, Any]:
        return {**self.product.to_dict(), **{k: v for k, v in asdict(self).items() if k != "product"}}
