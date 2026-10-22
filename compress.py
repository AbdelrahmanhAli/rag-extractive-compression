"""Extractive compression of retrieved context.

Retrieval is coarse. A chunk can be 30% relevant and still get pasted into the
prompt in full, wasting tokens and diluting the answer.

This demo asks the model to lift *only* the source text that answers the
question — adding nothing, inventing nothing — and drops passages that carry no
relevant material at all. What survives is a much smaller, cleaner context.
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import core

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Extract only the source text that is directly relevant to the "
                "question. Do not paraphrase, summarise, or add facts. Copy the "
                "original wording. If nothing is relevant, reply exactly: "
                "NO RELEVANT INFORMATION"
            ),
        ),
        ("human", "Question: {question}\n\nSource:\n{text}"),
    ]
)

NOISE = "no relevant information"


def is_relevant(extracted: str) -> bool:
    """True when the extractor returned usable source text."""
    return bool(extracted.strip()) and NOISE not in extracted.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    parser.add_argument("-k", type=int, default=6, help="chunks to compress")
    parser.add_argument(
        "--max-per-doc", type=int, default=600, help="truncated context per passage"
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() or core.question_from_args(["prog"])
    model = core.chat_model()

    retrieved = core.get_vectorstore().similarity_search(question, k=args.k)
    if not retrieved:
        print("No context retrieved.")
        return

    chain = EXTRACT_PROMPT | model | StrOutputParser()
    compressed: list[Document] = []
    for doc in retrieved:
        kept = chain.invoke(
            {"question": question, "text": doc.page_content[: args.max_per_doc]}
        ).strip()
        if not is_relevant(kept):
            continue
        compressed.append(Document(page_content=kept, metadata=dict(doc.metadata)))

    before = sum(core.estimate_tokens(d.page_content) for d in retrieved)
    after = sum(core.estimate_tokens(d.page_content) for d in compressed)

    print(f"Retrieved   : {len(retrieved)} chunk(s), ~{before} tokens")
    print(f"Compressed  : {len(compressed)} passage(s), ~{after} tokens")
    if before:
        print(f"Reduction   : {(1 - after / before) * 100:.0f}%")

    print("\n" + "=" * 60 + "\n")
    print(core.answer(question, compressed, model=model, show_sources=True))


if __name__ == "__main__":
    main()
