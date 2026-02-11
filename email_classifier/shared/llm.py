from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from litellm import completion, embedding
from pydantic import BaseModel

logger = logging.getLogger("email_classifier.llm")


def call_llm(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "timeout": timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = completion(**kwargs)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = retry_backoff**attempt
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s. Retrying in %.2fs",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
            else:
                logger.error("LLM call failed after %s attempts: %s", max_retries + 1, e)
    raise last_err  # type: ignore[misc]


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
    return t


def _fix_common_json_issues(s: str) -> str:
    # Remove trailing commas before object/array close
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_json_object(text: str) -> dict[str, Any]:
    t = _strip_code_fences(text)
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("No JSON object found in LLM output.")
    candidate = t[s : e + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = _fix_common_json_issues(candidate)
        return json.loads(candidate)


def schema_str(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2)


def embed_text(model: str, text: str) -> list[float]:
    resp = embedding(model=model, input=[text])
    return resp["data"][0]["embedding"]


def call_llm_json(
    model: str,
    system: str,
    user: str,
    schema: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
    max_attempts: int = 3,
) -> dict[str, Any]:
    last_err: Exception | None = None
    prompt = user
    for attempt in range(max_attempts):
        raw = call_llm(
            model=model,
            system=system,
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            return parse_json_object(raw)
        except Exception as e:
            last_err = e
            repair_schema = schema or ""
            raw_trim = raw.strip()
            if len(raw_trim) > 4000:
                raw_trim = raw_trim[:4000] + "\n...[truncated]"
            prompt = (
                user
                + "\n\nYour previous response was not valid JSON."
                + "\nReturn ONLY valid JSON that matches the schema."
                + ("\nSchema:\n" + repair_schema if repair_schema else "")
                + "\nFix this output into valid JSON only:\n"
                + raw_trim
            )
            logger.warning(
                "Invalid JSON from LLM (attempt %s/%s): %s", attempt + 1, max_attempts, e
            )
    raise last_err  # type: ignore[misc]


def call_llm_json_model(
    model: str,
    system: str,
    user: str,
    model_cls: type[BaseModel],
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
    max_attempts: int = 3,
) -> BaseModel:
    schema = schema_str(model_cls)
    try:
        raw = call_llm_json(
            model=model,
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_attempts=max_attempts,
        )
        return model_cls.model_validate(raw)
    except Exception as e:
        logger.warning("Model validation failed: %s", e)
        # Best-effort fallback: mark as needs_human_review if supported
        data = {}
        if "needs_human_review" in model_cls.model_fields:
            data["needs_human_review"] = True
        return model_cls.model_validate(data)
