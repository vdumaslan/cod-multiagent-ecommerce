from __future__ import annotations

import os
from typing import Any

import requests


class LLMRuntime:
    def __init__(
        self,
        model_id: str,
        fallback_model_id: str | None = None,
        timeout_seconds: int = 45,
    ) -> None:
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.timeout_seconds = timeout_seconds
        self.token = os.getenv("HF_TOKEN")
        self.last_model_used: str | None = None
        self.last_error: str | None = None

    def _extract_text(self, payload: Any) -> str | None:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("text") or first.get("summary_text")
                return str(text).strip() if text else None
        if isinstance(payload, dict):
            text = payload.get("generated_text") or payload.get("text") or payload.get("summary_text")
            if text:
                return str(text).strip()
        return None

    def _request(self, model_id: str, prompt: str, max_new_tokens: int, temperature: float) -> str | None:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        body = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True, "use_cache": True},
        }
        response = requests.post(url, json=body, headers=headers, timeout=self.timeout_seconds)
        if response.status_code != 200:
            self.last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            return None
        text = self._extract_text(response.json())
        if not text:
            self.last_error = "Model returned empty text."
            return None
        self.last_error = None
        self.last_model_used = model_id
        return text

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.1,
    ) -> str | None:
        composed_prompt = (
            f"System:\n{system_prompt.strip()}\n\n"
            f"User:\n{user_prompt.strip()}\n\n"
            "Assistant:\n"
        )
        for model in (self.model_id, self.fallback_model_id):
            if not model:
                continue
            try:
                text = self._request(model, composed_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                if text:
                    return text
            except Exception as exc:
                self.last_error = str(exc)
        return None
