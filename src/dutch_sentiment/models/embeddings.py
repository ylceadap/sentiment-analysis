"""Revision-aware embedding cache and encoder loading for research experiments."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..data import sha256_file
from ..experiments.common import hash_reviews


def embedding_cache_path(
    cache_dir: Path,
    model_name: str,
    revision: str,
    review_hash: str,
    normalized: bool,
    variant: str = "",
) -> Path:
    """Build a cache path that changes with model, data, and encoding settings."""
    key = hashlib.sha256(
        f"{model_name}|{revision}|{review_hash}|{normalized}|{variant}".encode()
    ).hexdigest()
    return cache_dir / f"{model_name}-{key[:16]}.npz"


def encode_or_load(
    model_spec: dict[str, Any], reviews: list[str], config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load verified cached embeddings or encode and persist a new cache entry."""
    review_hash = hash_reviews(reviews)
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = bool(config["normalize_embeddings"])
    variant = json.dumps(
        {
            "task": model_spec.get("task"),
            "max_sequence_length": model_spec.get("max_sequence_length"),
            "truncate_dimension": model_spec.get("truncate_dimension"),
        },
        sort_keys=True,
    )
    cache = embedding_cache_path(
        cache_dir,
        model_spec["name"],
        model_spec["revision"],
        review_hash,
        normalized,
        variant,
    )
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["review_hash"].item()) != review_hash:
                raise RuntimeError(f"Embedding cache hash mismatch: {cache}")
            embeddings = stored["embeddings"].astype(np.float32, copy=False)
        return embeddings, {
            "cache_hit": True,
            "cache_path": str(cache),
            "cache_sha256": sha256_file(cache),
            "encode_seconds": 0.0,
            "embedding_dimension": embeddings.shape[1],
        }

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Run `make install-embeddings` before this experiment") from exc

    started = time.perf_counter()
    requested_device = str(config.get("device", "auto"))
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    model_kwargs = {"default_task": str(model_spec["task"])} if model_spec.get("task") else None
    model = SentenceTransformer(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_folder=config["huggingface_cache_dir"],
        device=device,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
        truncate_dim=model_spec.get("truncate_dimension"),
        model_kwargs=model_kwargs,
    )
    if model_spec.get("max_sequence_length"):
        model.max_seq_length = int(model_spec["max_sequence_length"])
    if device == "cuda" and bool(config.get("use_fp16_on_cuda", True)):
        model.half()
    shard_size = int(config.get("cache_shard_size", 0))
    if shard_size > 0:
        # Approximate token length with character length so global sorting stays
        # compatible across SentenceTransformer versions without a private API.
        order = np.argsort([-len(review) for review in reviews], kind="stable")
        ordered_reviews = [reviews[index] for index in order]
        parts = cache.with_suffix(".sorted.parts")
        parts.mkdir(parents=True, exist_ok=True)
        encoded_parts: list[np.ndarray] = []
        for start in range(0, len(ordered_reviews), shard_size):
            stop = min(start + shard_size, len(ordered_reviews))
            shard_reviews = ordered_reviews[start:stop]
            shard_hash = hash_reviews(shard_reviews)
            shard_path = parts / f"{start:06d}-{stop:06d}.npz"
            if shard_path.is_file():
                with np.load(shard_path, allow_pickle=False) as stored:
                    if str(stored["review_hash"].item()) != shard_hash:
                        raise RuntimeError(f"Embedding shard hash mismatch: {shard_path}")
                    encoded_parts.append(stored["embeddings"].astype(np.float32, copy=False))
                continue
            shard = model.encode(
                shard_reviews,
                batch_size=int(config["batch_size"]),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalized,
            ).astype(np.float32, copy=False)
            np.savez_compressed(
                shard_path,
                embeddings=shard,
                review_hash=np.asarray(shard_hash),
                start=np.asarray(start),
                stop=np.asarray(stop),
            )
            encoded_parts.append(shard)
        sorted_embeddings = np.concatenate(encoded_parts, axis=0)
        embeddings = sorted_embeddings[np.argsort(order)]
    else:
        embeddings = model.encode(
            reviews,
            batch_size=int(config["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalized,
        ).astype(np.float32, copy=False)
    np.savez_compressed(
        cache,
        embeddings=embeddings,
        review_hash=np.asarray(review_hash),
        model_id=np.asarray(model_spec["model_id"]),
        revision=np.asarray(model_spec["revision"]),
        normalized=np.asarray(normalized),
        variant=np.asarray(variant),
    )
    return embeddings, {
        "cache_hit": False,
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "encode_seconds": time.perf_counter() - started,
        "embedding_dimension": embeddings.shape[1],
        "max_sequence_length": int(model.max_seq_length),
        "encoding_device": device,
        "encoding_dtype": "float16" if device == "cuda" else "float32",
    }
