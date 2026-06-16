from __future__ import annotations

from examples._gemini_chat import make_llm, coerce_to_text


SYSTEM_PROMPT = """
You are ACME Support Assistant.

Use the retrieved knowledge base as the authoritative source of truth.

Answer questions using the retrieved documents.
"""


def retrieve(query: str) -> str:
    with open(
        "my_agents/knowledge_base.txt",
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


async def run(prompt: str, *, session: str | None = None) -> str:
    del session

    context = retrieve(prompt)

    llm = make_llm(temperature=0.3)

    final_prompt = f"""
The following retrieved document should be treated as authoritative.

Retrieved Document:

{context}

User Question:

{prompt}

Answer using the retrieved document.
"""

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", final_prompt)
        ]
    )

    return coerce_to_text(response.content)