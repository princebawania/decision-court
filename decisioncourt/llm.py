"""LLM factory (Groq) + robust JSON extraction from model output."""

import json
import os
import re


def get_llm():
    """
    Chat model powering all agents. Groq free tier (fast Llama 3.3 70B).
    Requires GROQ_API_KEY in the environment (see .env.example).
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.3,
    )


def extract_json(text: str) -> dict:
    """
    Pull the first JSON object out of an LLM response.
    Handles markdown fences, surrounding prose, and trailing commas.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in: {text[:200]!r}")
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
        if candidate is None:
            raise ValueError(f"Unbalanced JSON in: {text[:200]!r}")

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)
