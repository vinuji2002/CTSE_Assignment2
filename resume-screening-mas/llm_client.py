"""
llm_client.py — Ollama local LLM client.
Runs entirely on your machine. No API keys needed.
"""
from __future__ import annotations
import requests


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen2.5:7b",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Send a prompt to a locally running Ollama model.

    Args:
        system_prompt: The agent system instructions.
        user_prompt:   The task and data to process.
        model:         Ollama model name (must be pulled first).
        max_tokens:    Maximum tokens to generate.
        temperature:   Sampling temperature.

    Returns:
        Generated text response as a string.

    Raises:
        RuntimeError: If Ollama is not running or the call fails.
    """
    api_url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out.")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama API error: {exc}")
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Ollama response: {exc}")


def clean_json_response(raw: str) -> str:
    """
    Remove markdown fences from LLM output if model added them.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        Clean JSON string with no markdown fences.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()