from __future__ import annotations

from typing import Any


def safe_token(value: Any) -> str:
    token = str(value).strip().replace(".", "p")
    return token.replace(" ", "_").replace("/", "_")


def build_case_name(**parts: Any) -> str:
    tokens: list[str] = []
    for key, value in parts.items():
        if value is None or value == "":
            continue
        tokens.append(f"{safe_token(key)}_{safe_token(value)}")
    return "_".join(tokens)
