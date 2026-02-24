from __future__ import annotations

import os

from huggingface_hub import InferenceClient


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

    def _request(self, model_id: str, prompt: str, max_new_tokens: int, temperature: float) -> str | None:
        if not self.token:
            self.last_error = "HF_TOKEN is not set."
            return None
        client = InferenceClient(model=model_id, token=self.token, timeout=self.timeout_seconds)

        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            text = completion.choices[0].message.content if completion.choices else None
            if text:
                self.last_error = None
                self.last_model_used = model_id
                return str(text).strip()
        except Exception:
            pass

        try:
            text = client.text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            if text:
                self.last_error = None
                self.last_model_used = model_id
                return str(text).strip()
        except Exception as exc:
            self.last_error = str(exc)
        return None

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
