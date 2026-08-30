from __future__ import annotations

import re
from dataclasses import dataclass


COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "canvas", "suede", "rubber", "fabric",
)
STYLES = (
    "casual", "formal", "athletic", "sporty", "vintage", "classic", "modern",
    "slim fit", "relaxed fit", "loose fit", "fitted", "long sleeve",
    "short sleeve", "v-neck", "crew neck", "high waisted", "low rise",
)
USE_CASES = (
    "hiking", "running", "walking", "gym", "training", "winter", "summer",
    "outdoor", "work", "travel", "commuting", "wedding", "school", "everyday",
)
FEATURES = (
    "waterproof", "water resistant", "breathable", "comfortable", "comfort",
    "lightweight", "insulated", "durable", "stretch", "pockets", "machine wash",
    "non slip", "arch support", "adjustable", "reversible",
)


def _pattern(vocabulary: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(value.replace("-", " "))
        for value in sorted(vocabulary, key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


COLOR_RE = _pattern(COLORS)
MATERIAL_RE = _pattern(MATERIALS)
STYLE_RE = _pattern(STYLES)
USE_CASE_RE = _pattern(USE_CASES)
FEATURE_RE = _pattern(FEATURES)


@dataclass(frozen=True)
class ProductFacets:
    colors: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()

    def values(self, attribute: str) -> tuple[str, ...]:
        mapping = {
            "color": self.colors,
            "material": self.materials,
            "size": self.sizes,
            "style": self.styles,
            "use_case": self.use_cases,
            "feature": self.features,
            "category": self.categories,
            "brand": self.brands,
        }
        return mapping.get(attribute, ())


def _terms(text: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    lowered = text.casefold().replace("-", " ")
    return tuple(dict.fromkeys(
        "gray" if value == "grey" else value
        for match in pattern.finditer(lowered)
        if (value := match.group(0).casefold())
    ))


def normalize_facets(product: dict, flattened: dict[str, str]) -> ProductFacets:
    searchable = " ".join(flattened.values())
    explicit_sizes = tuple(dict.fromkeys(
        match.group(1).casefold()
        for match in re.finditer(
            r"\b(?:size|width)\s*[:=]?\s*([a-z0-9./-]+(?:\s+(?:wide|narrow|regular))?)\b",
            searchable,
            re.I,
        )
    ))
    categories_value = product.get("categories") or []
    if not isinstance(categories_value, list):
        categories_value = [categories_value]
    categories = tuple(dict.fromkeys(
        re.sub(r"\s+", " ", str(value)).strip().casefold()
        for value in categories_value if str(value).strip()
    ))
    brand = flattened.get("store", "").strip().casefold()
    return ProductFacets(
        colors=_terms(searchable, COLOR_RE),
        materials=_terms(searchable, MATERIAL_RE),
        sizes=explicit_sizes,
        styles=_terms(searchable, STYLE_RE),
        use_cases=_terms(searchable, USE_CASE_RE),
        features=_terms(searchable, FEATURE_RE),
        categories=categories,
        brands=(brand,) if brand else (),
    )
