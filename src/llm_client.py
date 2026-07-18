"""OpenRouter LLM client for natural-language query understanding."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests

from src.config import get_openrouter_api_key, get_openrouter_model, get_openrouter_model_fallbacks
from src.intent_parser import AnalysisIntent

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ALLOWED_INTENTS = {
    "summary",
    "ranking",
    "trend",
    "compare",
    "correlation",
    "distribution",
    "aggregation",
    "forecast",
    "rfm",
    "abc",
    "cohort",
    "anomaly",
}

SYSTEM_PROMPT = """You are a data analysis assistant. Given a dataset schema and user question,
return ONLY valid JSON (no markdown) with these fields:
{
  "intent_type": "summary|ranking|trend|compare|correlation|distribution|aggregation|forecast|rfm|abc|cohort|anomaly",
  "description": "short description",
  "metric_col": "column name or null",
  "group_col": "column name or null",
  "compare_values": ["value1", "value2"],
  "top_n": 10,
  "time_grain": "month|year|week|day|null",
  "relevant_columns": ["col1", "col2"],
  "confidence": 0.0-1.0
}
Rules:
- Use ONLY column names from the provided schema.
- Never invent columns.
- For ranking use intent_type ranking.
- For time trends use intent_type trend and set group_col to the date column.
- For predict/forecast use intent_type forecast.
- For RFM customer segmentation use intent_type rfm.
- For ABC analysis use intent_type abc.
- For cohort/retention use intent_type cohort.
- For outliers/anomalies use intent_type anomaly.
- compare_values only for compare intent.
"""


def _build_user_prompt(
    question: str,
    profile: dict[str, Any],
    df: pd.DataFrame,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    sample = df.head(3).to_dict(orient="records")
    payload = {
        "question": question,
        "columns": profile.get("column_names", []),
        "roles": profile.get("roles", {}),
        "dtypes": profile.get("dtypes", {}),
        "rows": profile.get("rows", 0),
        "sample_rows": sample,
    }
    if chat_history:
        payload["prior_questions"] = [m["content"] for m in chat_history if m.get("role") == "user"][-5:]
    return json.dumps(payload, default=str)


def _call_openrouter(messages: list[dict[str, str]], model: str) -> str:
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 600,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Data Analysis Assistant",
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if not response.ok:
        detail = response.text[:200]
        raise ValueError(f"{response.status_code} for model '{model}': {detail}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError(f"Empty response from model '{model}'")
    return content


def _call_openrouter_with_fallbacks(messages: list[dict[str, str]]) -> tuple[str, str]:
    errors: list[str] = []
    for model in get_openrouter_model_fallbacks():
        try:
            return _call_openrouter(messages, model), model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model}: {exc}")
    raise ValueError("All OpenRouter models failed. " + " | ".join(errors))


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _validate_columns(data: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    col_set = set(columns)

    def pick(name: str | None) -> str | None:
        if name and name in col_set:
            return name
        return None

    data["metric_col"] = pick(data.get("metric_col"))
    data["group_col"] = pick(data.get("group_col"))
    relevant = [c for c in data.get("relevant_columns", []) if c in col_set]
    data["relevant_columns"] = relevant or [c for c in [data["metric_col"], data["group_col"]] if c]
    intent = data.get("intent_type", "summary")
    if intent not in ALLOWED_INTENTS:
        data["intent_type"] = "summary"
    return data


def _to_intent(data: dict[str, Any]) -> AnalysisIntent:
    return AnalysisIntent(
        intent_type=data.get("intent_type", "summary"),
        description=data.get("description", "LLM-parsed analysis"),
        metric_col=data.get("metric_col"),
        group_col=data.get("group_col"),
        compare_values=data.get("compare_values") or [],
        top_n=int(data.get("top_n") or 10),
        time_grain=data.get("time_grain"),
        confidence=float(data.get("confidence") or 0.8),
        relevant_columns=data.get("relevant_columns") or [],
    )


def parse_question_with_llm(
    question: str,
    profile: dict[str, Any],
    df: pd.DataFrame,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[AnalysisIntent | None, str | None]:
    if not get_openrouter_api_key():
        return None, "OPENROUTER_API_KEY not set in .env"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, profile, df, chat_history)},
    ]

    last_error: str | None = None
    used_model: str | None = None
    for attempt in range(2):
        try:
            content, used_model = _call_openrouter_with_fallbacks(messages)
            parsed = _extract_json(content)
            parsed = _validate_columns(parsed, profile.get("column_names", []))
            intent = _to_intent(parsed)
            if used_model and used_model != get_openrouter_model():
                intent.confidence = max(0.5, intent.confidence - 0.05)
            return intent, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "Return ONLY raw JSON. No markdown, no explanation.",
                    }
                )

    return None, last_error


def is_llm_configured() -> bool:
    return bool(get_openrouter_api_key())
