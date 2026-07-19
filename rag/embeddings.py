"""Proveedores desacoplados de embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    def fingerprint(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        ...


class HashEmbeddingProvider(EmbeddingProvider):
    """Embedding local determinista para índices sin proveedor externo."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("dimensions debe ser al menos 8.")
        self.dimensions = dimensions

    def fingerprint(self) -> str:
        return f"{super().fingerprint()}:{self.dimensions}"

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)
