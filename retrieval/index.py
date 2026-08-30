from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from conversation import BudgetRange, ConversationState
from .facets import ProductFacets, normalize_facets

if TYPE_CHECKING:
    from .semantic import SemanticRetriever


INDEX_SCHEMA_VERSION = 3
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
QUERY_EXPANSIONS = {
    "sneakers": ("shoes", "trainers"),
    "trainer": ("sneaker", "shoe"),
    "trainers": ("sneakers", "shoes"),
    "waterproof": ("water", "resistant"),
    "breathable": ("ventilated", "mesh"),
    "purse": ("handbag", "bag"),
    "handbag": ("purse", "bag"),
    "trousers": ("pants",),
    "warm": ("thermal", "insulated"),
}
ROUTE_WEIGHTS = {"broad": 0.75, "core": 1.15, "constraints": 1.0, "exact": 1.25}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_tuple(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return ()
    return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    price: float | None
    average_rating: float | None
    rating_number: float | None
    bm25_score: float
    retrieval_rank: int
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fusion_score: float = 0.0
    route_ranks: dict[str, int] = field(default_factory=dict)
    facets: ProductFacets = field(default_factory=ProductFacets)

    @property
    def searchable_text(self) -> str:
        return " ".join((self.title, self.categories, self.features, self.details, self.store, self.description))

    def attribute_text(self, attribute: str) -> str:
        mapping = {
            "category": " ".join((self.title, self.categories)),
            "material": " ".join((self.title, self.features, self.details, self.description)),
            "color": " ".join((self.title, self.features, self.details, self.description)),
            "size": " ".join((self.title, self.features, self.details)),
            "style": " ".join((self.title, self.categories, self.features, self.details)),
            "brand": self.store,
            "feature": " ".join((self.title, self.features, self.details, self.description)),
            "use_case": " ".join((self.title, self.categories, self.features, self.description)),
        }
        return mapping.get(attribute, self.searchable_text)


class CatalogIndex:
    """Frozen-catalog FTS5/facet index with optional multi-route and dense fusion."""

    def __init__(
        self,
        catalog_path: str | Path,
        semantic_retriever: SemanticRetriever | None = None,
        rrf_k: int = 60,
        expand_queries: bool = False,
        multi_route: bool = False,
        use_persistent_cache: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.semantic_retriever = semantic_retriever
        self.rrf_k = rrf_k
        self.expand_queries = expand_queries
        self.multi_route = multi_route
        self.cache_hit = False
        self.cache_path: Path | None = None
        self.last_query_cache_hit = False
        self._lexical_cache: OrderedDict[tuple[object, ...], tuple[Candidate, ...]] = OrderedDict()
        self._lexical_cache_limit = 16
        self.connection = self._open_index(use_persistent_cache)

    def _open_index(self, use_persistent_cache: bool) -> sqlite3.Connection:
        if not use_persistent_cache:
            connection = sqlite3.connect(":memory:")
            self._build(connection)
            return connection
        stat = self.catalog_path.stat()
        cache_directory = self.catalog_path.parent / ".cache"
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_name = (
            f"{self.catalog_path.stem}.{stat.st_size}.{stat.st_mtime_ns}."
            f"fts-v{INDEX_SCHEMA_VERSION}.sqlite3"
        )
        cache_path = cache_directory / cache_name
        self.cache_path = cache_path
        if cache_path.exists():
            self.cache_hit = True
            return self._copy_cache_to_memory(cache_path)

        temporary = cache_directory / f"{cache_name}.{os.getpid()}.{uuid.uuid4().hex}.building"
        connection = sqlite3.connect(temporary)
        try:
            self._build(connection)
        finally:
            connection.close()
        try:
            os.replace(temporary, cache_path)
        except FileExistsError:  # another process completed the immutable cache first
            temporary.unlink(missing_ok=True)
        return self._copy_cache_to_memory(cache_path)

    @staticmethod
    def _copy_cache_to_memory(cache_path: Path) -> sqlite3.Connection:
        """Keep per-turn FTS latency in-memory while avoiding repeated JSONL builds."""

        source = sqlite3.connect(f"file:{cache_path.as_posix()}?mode=ro", uri=True)
        destination = sqlite3.connect(":memory:")
        try:
            source.backup(destination)
        finally:
            source.close()
        return destination

    def _build(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute(
            "CREATE TABLE product_facets("
            "parent_asin TEXT PRIMARY KEY, colors TEXT, materials TEXT, sizes TEXT, styles TEXT, "
            "use_cases TEXT, features TEXT, categories TEXT, brands TEXT)"
        )
        products_batch: list[tuple[object, ...]] = []
        facets_batch: list[tuple[object, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                flattened = {
                    field_name: _text(product.get(field_name))
                    for field_name in ("title", "categories", "features", "details", "store", "description")
                }
                identifier = str(product["parent_asin"])
                products_batch.append((
                    identifier, flattened["title"], flattened["categories"], flattened["features"],
                    flattened["details"], flattened["store"], flattened["description"],
                    product.get("price"), product.get("average_rating"), product.get("rating_number"),
                ))
                facets = normalize_facets(product, flattened)
                facets_batch.append((identifier, *(
                    json.dumps(list(values), separators=(",", ":"))
                    for values in (
                        facets.colors, facets.materials, facets.sizes, facets.styles,
                        facets.use_cases, facets.features, facets.categories, facets.brands,
                    )
                )))
                if len(products_batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", products_batch
                    )
                    connection.executemany(
                        "INSERT INTO product_facets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", facets_batch
                    )
                    products_batch.clear()
                    facets_batch.clear()
        if products_batch:
            connection.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", products_batch
            )
            connection.executemany(
                "INSERT INTO product_facets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", facets_batch
            )
        connection.commit()

    def _tokens(self, text: str) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for token in TOKEN_RE.findall(text.casefold()):
            if len(token) > 1 and token not in STOPWORDS and token not in seen:
                seen.add(token)
                unique.append(token)
                if self.expand_queries:
                    for synonym in QUERY_EXPANSIONS.get(token, ()):
                        if synonym not in seen:
                            seen.add(synonym)
                            unique.append(synonym)
            if len(unique) >= 80:
                break
        return unique

    def _expression(self, text: str) -> str:
        return " OR ".join(f'"{token}"' for token in self._tokens(text))

    def _column_expression(self, columns: tuple[str, ...], text: str) -> str:
        expression = self._expression(text)
        return f"{{{' '.join(columns)}}} : ({expression})" if expression else ""

    def _exact_expression(self, state: ConversationState) -> str:
        phrases: list[str] = []
        for value in [state.category, *(
            str(item.value) for item in state.constraints
            if item.kind != "negative" and not isinstance(item.value, BudgetRange)
        )]:
            tokens = self._tokens(value)
            if len(tokens) >= 2:
                phrases.append('"' + " ".join(tokens[:10]) + '"')
        return " OR ".join(dict.fromkeys(phrases))

    @staticmethod
    def _candidate(row: tuple[object, ...], rank: int, lexical_rank: int | None) -> Candidate:
        return Candidate(
            parent_asin=str(row[0]), title=str(row[1] or ""), categories=str(row[2] or ""),
            features=str(row[3] or ""), details=str(row[4] or ""), store=str(row[5] or ""),
            description=str(row[6] or ""), price=_number(row[7]), average_rating=_number(row[8]),
            rating_number=_number(row[9]), bm25_score=max(0.0, -float(row[10] or 0.0)),
            retrieval_rank=rank, lexical_rank=lexical_rank,
            facets=ProductFacets(
                colors=_json_tuple(row[11]), materials=_json_tuple(row[12]),
                sizes=_json_tuple(row[13]), styles=_json_tuple(row[14]),
                use_cases=_json_tuple(row[15]), features=_json_tuple(row[16]),
                categories=_json_tuple(row[17]), brands=_json_tuple(row[18]),
            ),
        )

    def _search(self, expression: str, limit: int, route: str) -> list[Candidate]:
        if not expression:
            return []
        weights = {
            "broad": (0.0, 6.0, 4.0, 3.0, 3.0, 1.5, 1.0, 0.0, 0.0, 0.0),
            "core": (0.0, 8.0, 6.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0),
            "constraints": (0.0, 2.0, 1.0, 6.0, 5.0, 2.0, 4.0, 0.0, 0.0, 0.0),
            "exact": (0.0, 7.0, 5.0, 5.0, 4.0, 2.0, 3.0, 0.0, 0.0, 0.0),
        }[route]
        placeholders = ", ".join(str(value) for value in weights)
        rows = self.connection.execute(
            "SELECT products.parent_asin, title, products.categories, products.features, details, "
            "store, description, price, average_rating, rating_number, "
            f"bm25(products, {placeholders}) AS rank, "
            "facets.colors, facets.materials, facets.sizes, facets.styles, facets.use_cases, "
            "facets.features, facets.categories, facets.brands "
            "FROM products JOIN product_facets AS facets "
            "ON facets.parent_asin = products.parent_asin "
            "WHERE products MATCH ? ORDER BY rank ASC, products.parent_asin ASC LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [self._candidate(row, rank, rank) for rank, row in enumerate(rows, 1)]

    def _lexical(self, state: ConversationState, limit: int) -> list[Candidate]:
        broad = self._search(self._expression(state.query), limit, "broad")
        if not self.multi_route:
            return broad
        constraint_text = " ".join(
            item.phrase for item in state.constraints if item.kind != "negative"
        )
        routes = {
            "broad": broad,
            "core": self._search(
                self._column_expression(("title", "categories"), state.category), limit, "core"
            ),
            "constraints": self._search(
                self._column_expression(("features", "details", "description"), constraint_text),
                limit,
                "constraints",
            ),
            "exact": self._search(self._exact_expression(state), limit, "exact"),
        }
        combined: dict[str, Candidate] = {}
        route_ranks: dict[str, dict[str, int]] = {}
        route_scores: dict[str, float] = {}
        for route, rows in routes.items():
            for rank, item in enumerate(rows, 1):
                route_ranks.setdefault(item.parent_asin, {})[route] = rank
                route_scores[item.parent_asin] = route_scores.get(item.parent_asin, 0.0) + (
                    ROUTE_WEIGHTS[route] / (self.rrf_k + rank)
                )
                existing = combined.get(item.parent_asin)
                if existing is None or item.bm25_score > existing.bm25_score:
                    combined[item.parent_asin] = item
        fused = [replace(
            item,
            fusion_score=route_scores[identifier],
            route_ranks=route_ranks[identifier],
        ) for identifier, item in combined.items()]
        fused.sort(key=lambda item: (-item.fusion_score, item.route_ranks.get("broad", 10**9),
                                    item.parent_asin))
        return [replace(item, retrieval_rank=rank, lexical_rank=rank)
                for rank, item in enumerate(fused[:limit], 1)]

    def _catalog_candidates(self, identifiers: list[str]) -> dict[str, Candidate]:
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            "SELECT products.parent_asin, title, products.categories, products.features, details, "
            "store, description, price, average_rating, rating_number, 0.0, "
            "facets.colors, facets.materials, facets.sizes, facets.styles, facets.use_cases, "
            "facets.features, facets.categories, facets.brands "
            "FROM products JOIN product_facets AS facets "
            "ON facets.parent_asin = products.parent_asin "
            f"WHERE products.parent_asin IN ({placeholders})",
            identifiers,
        ).fetchall()
        return {str(row[0]): self._candidate(row, 0, None) for row in rows}

    def retrieve(self, state: ConversationState, limit: int = 200) -> list[Candidate]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000")
        # Multi-route queries depend on normalized constraint values as well as the
        # human-readable cumulative query.  Include both so a correction with an
        # unchanged phrase cannot reuse candidates from stale state.
        constraint_signature = tuple(
            (item.attribute, str(item.value), item.phrase, item.kind, item.intent_version)
            for item in state.constraints
        )
        cache_key = (
            state.query,
            constraint_signature,
            limit,
            self.multi_route,
            self.expand_queries,
        )
        cached = self._lexical_cache.get(cache_key)
        self.last_query_cache_hit = cached is not None
        if cached is None:
            lexical = self._lexical(state, limit)
            self._lexical_cache[cache_key] = tuple(lexical)
            self._lexical_cache.move_to_end(cache_key)
            if len(self._lexical_cache) > self._lexical_cache_limit:
                self._lexical_cache.popitem(last=False)
        else:
            self._lexical_cache.move_to_end(cache_key)
            lexical = list(cached)
        if self.semantic_retriever is None or not state.query:
            return lexical
        try:
            semantic = list(self.semantic_retriever.retrieve(state.query, limit))
        except Exception as exc:  # an optional model must never disable the offline agent
            warnings.warn(f"Semantic retrieval failed; continuing with FTS5: {exc}", RuntimeWarning)
            self.semantic_retriever = None
            return lexical

        lexical_by_id = {item.parent_asin: item for item in lexical}
        semantic_rank: dict[str, int] = {}
        semantic_score: dict[str, float] = {}
        for rank, hit in enumerate(semantic, 1):
            identifier = str(hit.parent_asin)
            if identifier and identifier not in semantic_rank:
                semantic_rank[identifier] = rank
                semantic_score[identifier] = float(hit.score)
        missing = [identifier for identifier in semantic_rank if identifier not in lexical_by_id]
        combined = {**self._catalog_candidates(missing), **lexical_by_id}
        fused: list[Candidate] = []
        for identifier, item in combined.items():
            lexical_component = (
                item.fusion_score if item.fusion_score > 0
                else 0.0 if item.lexical_rank is None
                else 1.0 / (self.rrf_k + item.lexical_rank)
            )
            dense_rank = semantic_rank.get(identifier)
            semantic_component = 0.0 if dense_rank is None else 1.0 / (self.rrf_k + dense_rank)
            fused.append(replace(
                item,
                semantic_score=semantic_score.get(identifier),
                semantic_rank=dense_rank,
                fusion_score=lexical_component + semantic_component,
            ))
        fused.sort(key=lambda item: (-item.fusion_score, item.lexical_rank or 10**9,
                                    item.semantic_rank or 10**9, item.parent_asin))
        return [replace(item, retrieval_rank=rank) for rank, item in enumerate(fused[:limit], 1)]

    def close(self) -> None:
        self._lexical_cache.clear()
        self.connection.close()
