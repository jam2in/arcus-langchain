import json
from typing import Any, cast

import pytest
from arcus import Arcus, ArcusError
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_arcus import ArcusCache


class FakeArcus:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}
        self.set_fails = False

    def get(self, key: str) -> Any:
        data = self.items.get(key)
        return None if data is None else json.loads(data)

    def set(self, key: str, value: Any, *, exptime: int = 0) -> bool:
        if self.set_fails:
            return False
        self.items[key] = json.dumps(value).encode()
        return True

    def flush(self, prefix: str | None = None, *, delay: int = 0) -> bool:
        assert prefix is not None
        keys = [key for key in self.items if key.startswith(f"{prefix}:")]
        for key in keys:
            del self.items[key]
        return bool(keys)


@pytest.fixture
def client() -> FakeArcus:
    return FakeArcus()


def make_cache(client: FakeArcus, **kwargs: Any) -> ArcusCache:
    return ArcusCache(cast(Arcus, client), **kwargs)


def test_lookup_miss(client: FakeArcus) -> None:
    assert make_cache(client).lookup("capital of France", "llm") is None


def test_update_then_lookup(client: FakeArcus) -> None:
    cache = make_cache(client)
    generations = [Generation(text="Paris")]
    cache.update("capital of France", "llm", generations)
    assert cache.lookup("capital of France", "llm") == generations


def test_only_exact_prompt_hits(client: FakeArcus) -> None:
    cache = make_cache(client)
    cache.update("capital of France", "llm", [Generation(text="Paris")])
    assert cache.lookup("France's capital city", "llm") is None


def test_llm_strings_are_separate(client: FakeArcus) -> None:
    cache = make_cache(client)
    cache.update("capital of France", "llm-a", [Generation(text="Paris")])
    assert cache.lookup("capital of France", "llm-b") is None


def test_chat_generation_round_trip(client: FakeArcus) -> None:
    cache = make_cache(client)
    generations = [ChatGeneration(message=AIMessage(content="Paris"))]
    cache.update("capital of France", "llm", generations)
    assert cache.lookup("capital of France", "llm") == generations


def test_key_uses_name(client: FakeArcus) -> None:
    make_cache(client).update("capital of France", "llm", [Generation(text="Paris")])
    make_cache(client, name="app").update(
        "capital of France", "llm", [Generation(text="Paris")]
    )
    prefixes = sorted(key.split(":")[0] for key in client.items)
    assert prefixes == ["app", "arcus-cache"]


def test_update_fails_when_not_stored(client: FakeArcus) -> None:
    client.set_fails = True
    with pytest.raises(ArcusError):
        make_cache(client).update("capital of France", "llm", [Generation(text="P")])


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
