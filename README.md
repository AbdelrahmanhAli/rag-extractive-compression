# LLM Extractive Compression

Keep only the sentences that answer the question, drop the rest, and cut prompt tokens without losing evidence.

**Part of the [RAG Techniques](https://github.com/AbdelrahmanhAli/rag-techniques-collection) series — 15 self-contained LangChain v1 demos.**

---

## What it demonstrates

Retrieval is coarse. A 700-character chunk may contain one relevant
sentence and six hundred characters of background. Passing all of it to the
generator wastes tokens, costs money, and — per *Lost in the Middle* — pushes
useful evidence into a position the model attends to less.

This app compresses each retrieved chunk before generation:

```
retrieved chunk ──► extract relevant sentences ──► keep or drop
                          │
                          ├── nothing relevant ──► drop whole chunk
                          └── relevant text     ──► compressed context
```

The extractor is instructed to **copy original wording** and to reply
`NO RELEVANT INFORMATION` when nothing applies. That sentinel is what lets the
app drop whole chunks instead of keeping a hallucinated summary.

The script reports estimated tokens before and after, so the saving is visible
rather than assumed.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OpenAI key
```

Then run:

```bash
python compress.py "why does chunk size matter?"
python compress.py "what is reciprocal rank fusion?" -k 8
```

`python <script> --help` lists the available flags. The first run builds a
persistent index in `./.chroma`; later runs reuse it. Delete that folder (or use
`--rebuild` where supported) after changing anything in `data/`.

## Project layout

```
.
├── compress.py
├── core.py
├── conftest.py
├── data/
│   └── sample.txt
├── tests/
│   ├── test_compress.py
│   └── test_core.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── LICENSE
```

## Tests

```bash
pip install pytest
pytest
```

The tests run entirely offline — no API key and no network. They cover the
shared helpers and the technique-specific logic, and use a deterministic fake
embedding model for the vector-store path.

## Notes

- One LLM call per retrieved chunk — latency grows with `-k`.
- Extraction can drop nuance that the final answer needed; compare against no compression before adopting it.
- Token counts are estimates (`len(text) // 4`), not tokenizer-exact.
- `rag-contextual-compression` does the same job with LangChain's built-in compressor retriever.

## References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Contextual compression in LangChain RAG from scratch](https://github.com/langchain-ai/rag-from-scratch)
- [LangChain v1 overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain integrations reference](https://reference.langchain.com/python/integrations/overview)
- [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)

## License

MIT — see [LICENSE](LICENSE).
