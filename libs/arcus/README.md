# langchain-arcus

[LangChain](https://github.com/langchain-ai/langchain) integration for
[Arcus](https://github.com/naver/arcus) Cache.

> Status: pre-alpha.

## Semantic cache

```python
from arcus import Arcus
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings
from langchain_arcus import ArcusSemanticCache

client = Arcus("127.0.0.1:2181", "test")
set_llm_cache(ArcusSemanticCache(client, OpenAIEmbeddings(), distance_threshold=0.2))
```

`clear()` is not supported yet.

## Development

```sh
uv sync
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
