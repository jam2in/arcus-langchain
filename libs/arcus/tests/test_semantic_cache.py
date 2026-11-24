import json
import math
from collections.abc import Sequence
from typing import Any, cast

import pytest
from arcus import Arcus, ArcusError, VectorOptions
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_arcus import ArcusSemanticCache

VECTORS = {
    "capital of France": [1.0, 0.0, 0.0],
    "France's capital city": [0.99, 0.1, 0.0],
    "who wrote Hamlet": [0.0, 1.0, 0.0],
}


class FakeEmbeddings(Embeddings):
    def __init__(self, padding: int = 0) -> None:
        self.padding = padding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return VECTORS[text] + [0.0] * self.padding


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1 - dot / (math.hypot(*a) * math.hypot(*b))


class FakeArcus:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}
        self.indexes: dict[str, dict[str, Sequence[float]]] = {}
        self.options: dict[str, VectorOptions] = {}
        self.max_count = 0
        self.set_fails = False

    def get(self, key: str) -> Any:
        data = self.items.get(key)
        return None if data is None else json.loads(data)

    def set(self, key: str, value: Any, *, exptime: int = 0) -> bool:
        if self.set_fails:
            return False
        self.items[key] = json.dumps(value).encode()
        return True

    def vop_create(self, index: str, options: VectorOptions) -> object | None:
        if index in self.indexes:
            return None
        self.indexes[index] = {}
        self.options[index] = options
        return object()

    def vop_upsert(self, index: str, vkey: str, vector: Sequence[float]) -> bool:
        elements = self.indexes.get(index)
        if elements is None:
            return False
        if self.max_count and vkey not in elements and len(elements) >= self.max_count:
            return False
        elements[vkey] = vector
        return True

    def flush(self, prefix: str | None = None, *, delay: int = 0) -> bool:
        assert prefix is not None
        keys = [key for key in self.items if key.startswith(f"{prefix}:")]
        indexes = [index for index in self.indexes if index.startswith(f"{prefix}:")]
        for key in keys:
            del self.items[key]
        for index in indexes:
            del self.indexes[index]
        return bool(keys or indexes)

    def vop_search(
        self, index: str, query: Sequence[float], count: int
    ) -> list[tuple[str, float, None]] | None:
        if index not in self.indexes:
            return None
        results = [
            (vkey, _cosine_distance(query, vector), None)
            for vkey, vector in self.indexes[index].items()
        ]
        return sorted(results, key=lambda result: result[1])[:count]


@pytest.fixture
def client() -> FakeArcus:
    return FakeArcus()


def make_cache(
    client: FakeArcus, embeddings: Embeddings | None = None, **kwargs: Any
) -> ArcusSemanticCache:
    return ArcusSemanticCache(
        cast(Arcus, client), embeddings or FakeEmbeddings(), **kwargs
    )


def test_lookup_without_index(client: FakeArcus) -> None:
    cache = make_cache(client)
    assert cache.lookup("capital of France", "llm") is None


def test_update_then_lookup_same_prompt(client: FakeArcus) -> None:
    cache = make_cache(client)
    generations = [Generation(text="Paris")]
    cache.update("capital of France", "llm", generations)
    assert cache.lookup("capital of France", "llm") == generations


def test_lookup_similar_prompt(client: FakeArcus) -> None:
    cache = make_cache(client)
    generations = [Generation(text="Paris")]
    cache.update("capital of France", "llm", generations)
    assert cache.lookup("France's capital city", "llm") == generations


def test_lookup_beyond_threshold(client: FakeArcus) -> None:
    cache = make_cache(client)
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert cache.lookup("who wrote Hamlet", "llm") is None


def test_threshold_is_configurable(client: FakeArcus) -> None:
    cache = make_cache(client, distance_threshold=0.001)
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert cache.lookup("France's capital city", "llm") is None


def test_llm_strings_use_separate_indexes(client: FakeArcus) -> None:
    cache = make_cache(client)
    cache.update("capital of France", "llm-a", [Generation(text="Paris")])
    assert cache.lookup("capital of France", "llm-b") is None
    cache.update("capital of France", "llm-b", [Generation(text="Paris, France")])
    assert len(client.indexes) == 2
    assert cache.lookup("capital of France", "llm-a") == [Generation(text="Paris")]


def test_dimensions_use_separate_indexes(client: FakeArcus) -> None:
    make_cache(client).update("capital of France", "llm", [Generation(text="Paris")])
    cache = make_cache(client, FakeEmbeddings(padding=1))
    assert cache.lookup("capital of France", "llm") is None
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert len(client.indexes) == 2


def test_chat_generation_round_trip(client: FakeArcus) -> None:
    cache = make_cache(client)
    generations = [ChatGeneration(message=AIMessage(content="Paris"))]
    cache.update("capital of France", "llm", generations)
    assert cache.lookup("capital of France", "llm") == generations


def test_index_options_and_keys(client: FakeArcus) -> None:
    cache = make_cache(client, name="app")
    cache.update("capital of France", "llm", [Generation(text="Paris")])

    [index] = client.indexes
    assert index.startswith("app:")
    assert client.options[index].dim == 3

    [key] = client.items
    assert key.startswith("app:")
    assert key != index


def test_missing_value_is_a_miss(client: FakeArcus) -> None:
    cache = make_cache(client)
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    client.items.clear()
    assert cache.lookup("capital of France", "llm") is None


def test_update_fails_when_index_is_full(client: FakeArcus) -> None:
    cache = make_cache(client)
    client.max_count = 1
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    with pytest.raises(ArcusError):
        cache.update("who wrote Hamlet", "llm", [Generation(text="Shakespeare")])


def test_update_fails_when_value_is_not_stored(client: FakeArcus) -> None:
    cache = make_cache(client)
    client.set_fails = True
    with pytest.raises(ArcusError):
        cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert not client.indexes


@pytest.mark.parametrize("name", ["", "a:b"])
def test_invalid_name(client: FakeArcus, name: str) -> None:
    with pytest.raises(ValueError):
        make_cache(client, name=name)


def test_clear_flushes_only_its_name(client: FakeArcus) -> None:
    cache = make_cache(client)
    other = make_cache(client, name="other")
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    other.update("capital of France", "llm", [Generation(text="Paris")])

    cache.clear()
    assert cache.lookup("capital of France", "llm") is None
    assert other.lookup("capital of France", "llm") == [Generation(text="Paris")]

    cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert cache.lookup("capital of France", "llm") == [Generation(text="Paris")]
