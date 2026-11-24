"""LLM response cache backed by Arcus."""

import hashlib
from typing import Any, Final

from arcus import Arcus, ArcusError, Metric, VectorOptions
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumpd, load

DEFAULT_CACHE_NAME: Final = "arcus-cache"
DEFAULT_SEMANTIC_CACHE_NAME: Final = "arcus-semantic"
DEFAULT_DISTANCE_THRESHOLD: Final = 0.2


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dump(return_val: RETURN_VAL_TYPE) -> list[dict[str, Any]]:
    return [dumpd(generation) for generation in return_val]


def _load(value: list[dict[str, Any]]) -> RETURN_VAL_TYPE:
    return [load(generation, allowed_objects="core") for generation in value]


def _validate_name(name: str) -> None:
    if not name or ":" in name:
        raise ValueError(f"Name must be non-empty and contain no ':': {name!r}")


class ArcusCache(BaseCache):
    """Cache LLM generations in Arcus."""

    def __init__(self, client: Arcus, *, name: str = DEFAULT_CACHE_NAME) -> None:
        _validate_name(name)
        self._client = client
        self._name = name

    def _key(self, prompt: str, llm_string: str) -> str:
        return f"{self._name}:{_hash(llm_string)}-{_hash(prompt)}"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = self._client.get(self._key(prompt, llm_string))
        if value is None:
            return None
        return _load(value)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        key = self._key(prompt, llm_string)
        if not self._client.set(key, _dump(return_val)):
            raise ArcusError(f"Failed to store the generations: {key!r}")

    def clear(self, **kwargs: Any) -> None:
        self._client.flush(self._name)


class ArcusSemanticCache(BaseCache):
    """Return cached generations for similar prompts.

    One vector index per ``llm_string`` and embedding dimension.
    """

    def __init__(
        self,
        client: Arcus,
        embeddings: Embeddings,
        *,
        name: str = DEFAULT_SEMANTIC_CACHE_NAME,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        metric: Metric = Metric.COSINE,
    ) -> None:
        _validate_name(name)
        self._client = client
        self._embeddings = embeddings
        self._name = name
        self._distance_threshold = distance_threshold
        self._metric = metric

    def _index_key(self, llm_string: str, dim: int) -> str:
        return f"{self._name}:{_hash(llm_string)}-{dim}"

    def _generations_key(self, llm_string: str, vkey: str) -> str:
        return f"{self._name}:{_hash(llm_string)}-{vkey}"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        embedding = self._embeddings.embed_query(prompt)
        index = self._index_key(llm_string, len(embedding))
        results = self._client.vop_search(index, embedding, 1)
        if not results:
            return None

        vkey, distance, _ = results[0]
        if distance > self._distance_threshold:
            return None

        value = self._client.get(self._generations_key(llm_string, vkey))
        if value is None:
            return None
        return _load(value)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        embedding = self._embeddings.embed_query(prompt)
        index = self._index_key(llm_string, len(embedding))
        vkey = _hash(prompt)

        # Value first, so a found vector always has one.
        generations_key = self._generations_key(llm_string, vkey)
        if not self._client.set(generations_key, _dump(return_val)):
            raise ArcusError(f"Failed to store the generations: {generations_key!r}")
        if self._client.vop_upsert(index, vkey, embedding):
            return

        options = VectorOptions(dim=len(embedding), metric=self._metric)
        self._client.vop_create(index, options)
        if not self._client.vop_upsert(index, vkey, embedding):
            raise ArcusError(f"Failed to store the prompt vector in {index!r}")

    def clear(self, **kwargs: Any) -> None:
        self._client.flush(self._name)
