from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


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
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    model_revision: str | None = None,
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

    revision = (
        DEFAULT_MODEL_REVISION
        if model_name == DEFAULT_MODEL and model_revision is None
        else model_revision
    )
    model_kwargs = {"revision": revision} if revision else {}
    model = SentenceTransformer(model_name, **model_kwargs)
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
        "format_version": 2,
        "catalog_sha256": _sha256(catalog),
        "model_name": model_name,
        "model_revision": revision,
        "model_id": f"{model_name}@{revision}" if revision else model_name,
        "count": len(identifiers),
        "dimensions": int(embeddings.shape[1]),
        "normalized": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _split_model_id(model_id: str) -> tuple[str, str | None]:
    if "@" not in model_id:
        return model_id, None
    model_name, revision = model_id.rsplit("@", 1)
    return model_name, revision or None


class SentenceTransformerIndex:
    """Memory-mapped dense index with a Sentence Transformers query encoder."""

    def __init__(
        self,
        index_directory: str | Path,
        model_name: str | None = None,
        model_revision: str | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        np, SentenceTransformer = _optional_packages()
        directory = Path(index_directory)
        metadata_path = directory / "metadata.json"
        manifest_path = directory / "manifest.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.identifiers = json.loads(
                (directory / "identifiers.json").read_text(encoding="utf-8")
            )
            recorded_model = str(metadata["model_name"])
            recorded_revision = metadata.get("model_revision")
        elif manifest_path.exists():
            # Backward-compatible reader for the first retrieval experiment format.
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.identifiers = metadata.get("parent_asins")
            recorded_model, recorded_revision = _split_model_id(str(metadata["model_id"]))
        else:
            raise ValueError(f"No semantic index metadata found in {directory}")
        self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        expected_count = metadata.get("count", metadata.get("product_count"))
        if (
            not isinstance(self.identifiers, list)
            or len(self.identifiers) != len(self.embeddings)
            or len(self.identifiers) != expected_count
        ):
            raise ValueError("Semantic index identifiers and embeddings do not match")
        dimensions = metadata.get("dimensions", metadata.get("embedding_dimension"))
        if dimensions != int(self.embeddings.shape[1]):
            raise ValueError("Semantic index dimensions do not match metadata")
        if len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("Semantic index identifiers must be unique")
        recorded_catalog_sha256 = metadata.get("catalog_sha256")
        if catalog_path is not None and recorded_catalog_sha256 is not None:
            if _sha256(Path(catalog_path)) != recorded_catalog_sha256:
                raise ValueError("Semantic index was built from a different catalog")
        if model_name is not None and model_name != recorded_model:
            raise ValueError("Embedding model override does not match the semantic index")
        if model_revision is not None and model_revision != recorded_revision:
            raise ValueError("Embedding revision override does not match the semantic index")
        self._np = np
        self.model_name = recorded_model
        self.model_revision = recorded_revision
        model_kwargs = {"revision": self.model_revision} if self.model_revision else {}
        self.model = SentenceTransformer(self.model_name, **model_kwargs)

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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_semantic_index(
        args.catalog,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        model_revision=args.revision,
    )


if __name__ == "__main__":
    main()
