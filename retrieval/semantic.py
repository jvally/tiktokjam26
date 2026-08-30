from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SemanticHit:
    parent_asin: str
    score: float


class SemanticRetriever(Protocol):
    """Small boundary that keeps a model dependency out of core retrieval."""

    def retrieve(self, query: str, limit: int) -> Sequence[SemanticHit]: ...


def _optional_packages():
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Semantic search requires requirements-semantic.txt and a locally available model"
        ) from exc
    return np, SentenceTransformer


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _product_text(product: dict) -> str:
    fields = ("title", "categories", "features", "details", "store", "description")
    return " | ".join(part for part in (_flatten(product.get(field)) for field in fields) if part)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_semantic_index(
    catalog_path: str | Path,
    output_directory: str | Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> None:
    """Precompute normalized catalog embeddings for offline final evaluation."""

    np, SentenceTransformer = _optional_packages()
    catalog = Path(catalog_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    identifiers: list[str] = []
    texts: list[str] = []
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            identifiers.append(str(product["parent_asin"]))
            texts.append(_product_text(product))

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(output / "embeddings.npy", embeddings)
    (output / "identifiers.json").write_text(json.dumps(identifiers), encoding="utf-8")
    metadata = {
        "format_version": 1,
        "catalog_sha256": _sha256(catalog),
        "model_name": model_name,
        "count": len(identifiers),
        "dimensions": int(embeddings.shape[1]),
        "normalized": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class SentenceTransformerIndex:
    """Memory-mapped dense index with a Sentence Transformers query encoder."""

    def __init__(self, index_directory: str | Path, model_name: str | None = None) -> None:
        np, SentenceTransformer = _optional_packages()
        directory = Path(index_directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.identifiers = json.loads((directory / "identifiers.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        if len(self.identifiers) != len(self.embeddings) or len(self.identifiers) != metadata.get("count"):
            raise ValueError("Semantic index identifiers and embeddings do not match")
        self._np = np
        self.model_name = model_name or str(metadata["model_name"])
        self.model = SentenceTransformer(self.model_name)

    def retrieve(self, query: str, limit: int) -> list[SemanticHit]:
        if not query.strip() or limit <= 0:
            return []
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self._np.asarray(self.embeddings @ query_embedding)
        limit = min(limit, len(scores))
        if limit == len(scores):
            indices = self._np.arange(len(scores))
        else:
            indices = self._np.argpartition(scores, -limit)[-limit:]
        ordered = sorted(indices.tolist(), key=lambda index: (-float(scores[index]), self.identifiers[index]))
        return [SemanticHit(self.identifiers[index], float(scores[index])) for index in ordered]


def main() -> None:  # pragma: no cover - exercised manually with optional model
    parser = argparse.ArgumentParser(description="Build the optional offline semantic catalog index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/semantic_index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_semantic_index(args.catalog, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
