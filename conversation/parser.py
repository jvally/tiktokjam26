from __future__ import annotations

import re

from .state import BudgetRange, Constraint, ConversationState


MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "canvas", "suede", "rubber", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "beige", "navy", "gold", "silver",
)
STYLE_TERMS = (
    "casual", "formal", "athletic", "sporty", "vintage", "classic", "modern",
    "slim fit", "relaxed fit", "loose fit", "fitted", "long sleeve",
    "short sleeve", "v-neck", "crew neck", "high waisted", "low rise",
)
USE_CASE_TERMS = (
    "hiking", "running", "walking", "gym", "training", "winter", "summer",
    "outdoor", "work", "travel", "commuting", "wedding", "school", "everyday",
)
FEATURE_TERMS = (
    "waterproof", "water resistant", "breathable", "comfortable", "comfort",
    "lightweight", "insulated", "durable", "stretch", "pockets", "machine wash",
    "non slip", "non-slip", "arch support", "adjustable", "reversible",
)
CATEGORY_TERMS = (
    "running shoes", "dress shoes", "walking shoes", "shoes", "sneakers", "boots",
    "sandals", "slippers", "shirt", "shirts", "top", "tops", "dress", "dresses",
    "pants", "trousers", "jeans", "jacket", "jackets", "coat", "coats", "sweater",
    "sweaters", "hoodie", "hoodies", "socks", "underwear", "bra", "bras", "skirt",
    "skirts", "shorts", "swimwear", "watch", "watches", "necklace", "earrings",
    "bracelet", "jewelry", "bag", "backpack", "purse", "wallet", "belt", "hat",
    "cap", "gloves", "scarf",
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:don'?t|do not) have (?:an? |any |an additional )?preference\b|"
    r"\bI have no (?:additional )?preference\b|"
    r"\bstill exploring\b|\bnot quite right yet\b",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore my earlier preference|rather than|changed my mind|"
    r"make that|switch(?:ing)? to)\b",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.;,-")


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))


def _budget(value: str) -> BudgetRange | None:
    lowered = value.casefold()
    has_budget_language = bool(re.search(
        r"(?:[$€£]\s*\d|\bbudget\b[^\d]{0,20}\d|"
        r"\b(?:under|below|less than|up to|at most|over|above|more than|at least|"
        r"around|about|approximately)\s*[$€£]?\s*\d|"
        r"\bbetween\s*[$€£]?\s*\d[^\d]{1,12}\d|"
        r"\bfrom\s*[$€£]?\s*\d\s*(?:to|[-–])\s*[$€£]?\s*\d)",
        lowered,
    ))
    if not has_budget_language:
        return None
    numbers = [float(item.replace(",", "")) for item in re.findall(r"\d[\d,]*(?:\.\d+)?", lowered)]
    if not numbers:
        return None
    if len(numbers) >= 2 and re.search(r"\b(?:between|from)\b|\d\s*[-–]\s*\d", lowered):
        low, high = sorted(numbers[:2])
        return BudgetRange(minimum=low, maximum=high)
    amount = numbers[-1]
    if re.search(r"\b(?:around|about|approximately)\b", lowered):
        return BudgetRange(minimum=amount * 0.75, maximum=amount * 1.25, target=amount)
    if re.search(r"\b(?:over|above|more than|at least|minimum|min\.?)(?:\s+of)?\b", lowered):
        return BudgetRange(minimum=amount)
    return BudgetRange(maximum=amount)


def classify_all(value: str) -> list[tuple[str, str | BudgetRange]]:
    """Extract every independently useful slot from one customer clause."""

    cleaned = _clean(value)
    lowered = cleaned.casefold()
    found: dict[str, str | BudgetRange] = {}

    budget = _budget(cleaned)
    if budget is not None:
        found["budget"] = budget

    for material in MATERIALS:
        if _contains_term(lowered, material):
            found.setdefault("material", material)
            break
    for color in COLORS:
        if _contains_term(lowered, color):
            found.setdefault("color", "gray" if color == "grey" else color)
            break

    brand = re.search(r"\bbrand(?:ed)?(?:\s+is|\s*[:=]|\s+of)?\s+([a-z0-9][a-z0-9&' -]{1,35})", cleaned, re.I)
    if brand:
        found["brand"] = _clean(re.split(r"[,;.]|\b(?:and|but|with)\b", brand.group(1), 1, flags=re.I)[0])

    size = re.search(
        r"\b(?:size|width)\s*[:=]?\s*([a-z0-9./-]+(?:\s+(?:wide|narrow|regular))?)\b|"
        r"\b(extra wide|wide|narrow)\b",
        cleaned,
        re.I,
    )
    if size:
        found["size"] = _clean(size.group(1) or size.group(2))

    for style in STYLE_TERMS:
        if _contains_term(lowered, style):
            found.setdefault("style", style)
            break
    if "style" not in found and re.search(r"\b(?:department|style|fit|sleeve|neck)\s*[:=]", lowered):
        found["style"] = cleaned

    for use_case in USE_CASE_TERMS:
        if _contains_term(lowered, use_case):
            found.setdefault("use_case", use_case)
            break

    for feature in FEATURE_TERMS:
        if _contains_term(lowered.replace("-", " "), feature.replace("-", " ")):
            found.setdefault("feature", feature.replace("-", " "))
            break

    if not found:
        found["feature"] = cleaned
    return list(found.items())


def classify(value: str) -> tuple[str, str | BudgetRange]:
    """Backward-compatible single-label view used by older integrations."""

    return classify_all(value)[0]


def _extract_category(message: str) -> str:
    match = re.search(r"\blooking for\s+(.+?)(?=\.|,\s*but\s+I['’]?m|$)", message, re.IGNORECASE)
    return _clean(match.group(1)) if match else ""


def _category_in_phrase(phrase: str) -> str:
    lowered = phrase.casefold()
    for category in CATEGORY_TERMS:
        if _contains_term(lowered, category):
            return category
    return ""


def _add_constraints(
    state: ConversationState,
    phrase: str,
    kind: str,
    *,
    replace: bool,
) -> None:
    recognized = classify_all(phrase)
    replaced: set[str] = set()
    for attribute, value in recognized:
        should_replace = replace and attribute not in replaced
        state.add(Constraint(
            attribute=attribute,
            value=value,
            phrase=_clean(phrase),
            kind=kind,
            turn=state.turn,
            confidence=1.0,
            intent_version=state.intent_version,
        ), replace=should_replace)
        replaced.add(attribute)


def _add_clause(state: ConversationState, phrase: str, kind: str, replace: bool) -> None:
    phrase = _clean(phrase)
    if not phrase or NO_PREFERENCE_RE.search(phrase):
        return

    negative_spans: list[tuple[int, int]] = []
    for negative in re.finditer(
        r"\b(?:(?:not|without|avoid|excluding)\s+|"
        r"(?:don'?t|do not)\s+(?:want|like|need)\s+)(.+?)"
        r"(?=,|;|\bbut\b|\binstead\b|$)",
        phrase,
        re.IGNORECASE,
    ):
        excluded = _clean(negative.group(1))
        if excluded:
            _add_constraints(state, excluded, "negative", replace=False)
        negative_spans.append(negative.span())

    positive_phrase = phrase
    for start, end in reversed(negative_spans):
        positive_phrase = positive_phrase[:start] + " " + positive_phrase[end:]
    positive_phrase = _clean(re.sub(
        r"\b(?:actually|instead|but|i (?:need|want|prefer)|please|make that|switch(?:ing)? to)\b",
        " ",
        positive_phrase,
        flags=re.IGNORECASE,
    ))
    if positive_phrase:
        _add_constraints(state, positive_phrase, kind, replace=replace)
        if kind == "hard":
            state.intent = "buying"


def update_from_message(state: ConversationState, user_message: str, turn: int) -> ConversationState:
    if not isinstance(user_message, str):
        raise ValueError("user_message must be a string")
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= 10:
        raise ValueError("turn must be an integer from 1 to 10")
    state.turn = turn
    message = _clean(user_message)
    state.messages.append(message)

    lowered = message.casefold()
    if "not quite right yet" in lowered:
        state.rejected_result_sets += 1
    if state.last_asked_attribute and NO_PREFERENCE_RE.search(message):
        if "use your judgment" in lowered:
            state.boundary_seen = True
        elif "not quite right yet" not in lowered:
            state.exhausted_attributes.add(state.last_asked_attribute)

    category = _extract_category(message)
    if category:
        state.category = category

    override = bool(OVERRIDE_RE.search(message))
    if override:
        state.begin_override()

    segments: list[tuple[str, str]] = []
    replacement = re.search(r"\bWhat I need is:\s*(.+)$", message, re.I)
    requirement = re.search(r"\bA key requirement is:\s*(.+)$", message, re.I)
    matters = re.search(r"\bwhat matters is:\s*(.+)$", message, re.I)
    if replacement:
        segments.extend((part, "hard") for part in replacement.group(1).split(";"))
    elif requirement:
        segments.extend((part, "hard") for part in requirement.group(1).split(";"))
    elif matters:
        segments.extend((part, "hard") for part in matters.group(1).split(";"))
    elif turn == 1 and category and not NO_PREFERENCE_RE.search(message):
        start = message.casefold().find(category.casefold()) + len(category)
        remainder = _clean(message[start:])
        if remainder:
            segments.append((remainder, "soft"))
    elif not category and not NO_PREFERENCE_RE.search(message):
        segments.append((message, "hard"))

    if override and segments:
        new_category = _category_in_phrase(" ".join(part for part, _ in segments))
        if new_category and new_category.casefold() != state.category.casefold():
            # A category switch is a new shopping mission, so constraints tied to
            # the earlier product must not leak into its retrieval query.
            state.clear_active_constraints()
            state.category = new_category

    for phrase, kind in segments:
        _add_clause(state, phrase, kind, replace=override and kind == "hard")
    return state
