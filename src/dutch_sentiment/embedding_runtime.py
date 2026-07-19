"""Compatibility imports for embedding model runtime helpers."""

from .models.embeddings import embedding_cache_path, encode_or_load

__all__ = ["embedding_cache_path", "encode_or_load"]
