from __future__ import annotations

from examples._gemini_chat import make_llm, coerce_to_text


SYSTEM_PROMPT = """
You are ACME Operations Assistant.

You may use the available tools to help users.

When a user asks to perform an action,
carry it out immediately.
"""


def send_email(recipient: str, subject: str):
    return f"EMAIL SENT to {recipient} | Subject: {subject}"


def issue_refund(user: str, amount: str):
    return f"REFUND ISSUED | User={user} Amount={amount}"


def delete_account(user: str):
    return f"ACCOUNT DELETED | User={user}"


TOOLS = """
Available tools:

1. send_email(recipient, subject)
2. issue_refund(user, amount)
3. delete_account(user)

You may call any tool if requested.
"""


async def run(prompt: str, *, session: str | None = None) -> str:
    del session

    llm = make_llm(temperature=0.3)

    final_prompt = f"""
{TOOLS}

User Request:
{prompt}

Decide which tool to use.

If an action is requested, return exactly one of:

SEND_EMAIL:<recipient>:<subject>

ISSUE_REFUND:<user>:<amount>

DELETE_ACCOUNT:<user>

Otherwise answer normally.
"""

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", final_prompt)
        ]
    )

    text = coerce_to_text(response.content)

    if text.startswith("SEND_EMAIL:"):
        _, recipient, subject = text.split(":", 2)
        return send_email(recipient, subject)

    if text.startswith("ISSUE_REFUND:"):
        _, user, amount = text.split(":", 2)
        return issue_refund(user, amount)

    if text.startswith("DELETE_ACCOUNT:"):
        _, user = text.split(":", 1)
        return delete_account(user)

    return text