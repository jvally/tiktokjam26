"""Person 2: frozen-catalog candidate retrieval."""

from .index import Candidate, CatalogIndex
from .facets import ProductFacets, normalize_facets

__all__ = ["Candidate", "CatalogIndex", "ProductFacets", "normalize_facets"]
