from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaChatResult:
    content: str
    raw: dict[str, Any]
    elapsed_s: float


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        retries: int = 1,
        backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_predict: int = 512,
        seed: int = 42,
        stop: list[str] | None = None,
    ) -> OllamaChatResult:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": str(model),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(num_predict),
                "seed": int(seed),
            },
        }
        if stop:
            payload["options"]["stop"] = list(stop)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                    raw_txt = resp.read().decode("utf-8", errors="replace")
                j = json.loads(raw_txt)
                content = str((j.get("message") or {}).get("content") or "")
                return OllamaChatResult(content=content, raw=j, elapsed_s=float(time.time() - t0))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
                    continue
                raise

