"""Small deterministic lexical helpers; these are not embeddings."""

import re


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def phrase_match(expected: str, observed: str) -> bool:
    needle, haystack = tokens(expected), tokens(observed)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                                for i in range(len(haystack) - len(needle) + 1))
