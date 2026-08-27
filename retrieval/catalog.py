"""Load a normalized JSON array or line-delimited JSON catalog."""

import json
from pathlib import Path

from shared import Product


def load_catalog(path: str) -> list[Product]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Catalog must be a JSON array or a .jsonl file")
    products = [Product.from_dict(row) for row in rows]
    ids = [p.parent_asin for p in products]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog contains duplicate parent_asin values")
    return products
