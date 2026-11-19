"""LLM response cache backed by Arcus."""

from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache


class ArcusCache(BaseCache):
    """Cache LLM generations in Arcus."""

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        raise NotImplementedError

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        raise NotImplementedError

    def clear(self, **kwargs: Any) -> None:
        raise NotImplementedError
