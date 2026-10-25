from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .llm_audit import record_completion


class LLMClient(Protocol):
    def complete(
        self, *, system: str, user: str, json_mode: bool | None = None
    ) -> str:
        """Return one text completion."""


@dataclass
class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat endpoints."""

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_seconds: int = 180
    max_tokens: int = 16_000
    max_retries: int = 3
    json_mode: bool = True
    reasoning_effort: str | None = None
    seed: int | None = None
    last_response_metadata: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model.lower() == "gemini-3.5-flash":
            self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if self.base_url.rstrip("/") == "https://api.openai.com/v1":
                self.base_url = os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
        elif self.model.lower().startswith("qwen3.8-") or self.model.lower() == "deepseek-v4.1-flash":
            # Keep DashScope credentials isolated even when a GPT key is also set.
            self.api_key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
            if self.base_url.rstrip("/") == "https://api.openai.com/v1":
                self.base_url = (os.environ.get("DASHSCOPE_BASE_URL")
                                 or os.environ.get("QWEN_BASE_URL")
                                 or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        elif self.model.lower() == "mimo-v2.6-flash":
            self.api_key = self.api_key or os.environ.get("XIAOMI_API_KEY")
            if self.base_url.rstrip("/") == "https://api.openai.com/v1":
                self.base_url = "https://api.xiaomimimo.com/v1"
        elif self.model.lower() == "glm-5.3-flash":
            self.api_key = self.api_key or os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY")
            if self.base_url.rstrip("/") == "https://api.openai.com/v1":
                self.base_url = os.environ.get("GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
        elif self.model.lower() == "deepseek-flash":
            self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            if self.base_url.rstrip("/") == "https://api.openai.com/v1":
                self.base_url = "https://api.deepseek.com"
        else:
            self.api_key = (
                self.api_key
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
            )
        self.base_url = self.base_url.rstrip("/")
        if not self.api_key:
            raise ValueError("an API key is required (argument or provider-specific environment variable)")
        if self.reasoning_effort not in {None, "none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"unsupported reasoning_effort: {self.reasoning_effort}")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**31):
            raise ValueError("seed must be an integer in [0, 2**31 - 1]")

    def _build_request_payload(
        self, *, system: str, user: str, json_mode: bool | None = None
    ) -> dict:
        is_openai_reasoning = (
            "api.openai.com" in self.base_url
            and self.model.startswith(("gpt-5.6", "gpt-6-luna"))
        )
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if not is_openai_reasoning or self.reasoning_effort == "none":
            request_payload["temperature"] = self.temperature
        if is_openai_reasoning or self.model.lower().startswith("qwen3.8-") or self.model.lower() == "mimo-v2.6-flash":
            request_payload["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort is not None and self.model.lower() != "mimo-v2.6-flash":
                request_payload["reasoning_effort"] = self.reasoning_effort
        else:
            request_payload["max_tokens"] = self.max_tokens
        if self.model.lower() == "gemini-3.5-flash" and self.reasoning_effort is not None:
            request_payload["reasoning_effort"] = self.reasoning_effort
        if self.model.lower().startswith("qwen3.8-") and self.reasoning_effort is not None:
            request_payload["enable_thinking"] = self.reasoning_effort != "none"
        if self.model.lower().startswith("qwen3.8-"):
            # Best-effort reproducibility, not a cloud determinism guarantee.
            request_payload["seed"] = 0 if self.seed is None else self.seed
            if self.temperature == 0 and self.reasoning_effort == "none":
                request_payload["top_k"] = 1
        if (self.model.lower() == "deepseek-v4.1-flash"
                and "aliyuncs.com" in self.base_url.lower()
                and self.seed is not None):
            request_payload["seed"] = self.seed
        if (self.model.lower() == "deepseek-flash"
                and "api.deepseek.com" in self.base_url.lower()
                and self.seed is not None):
            request_payload["seed"] = self.seed
        if self.model.lower() == "deepseek-v4.1-flash" and self.reasoning_effort == "none":
            # DashScope DeepSeek defaults to thinking; disable it explicitly so
            # temperature=0 is effective in the non-thinking comparison.
            request_payload["enable_thinking"] = False
        if ("api.deepseek.com" in self.base_url.lower()
                and self.model.lower() == "deepseek-flash"
                and self.reasoning_effort == "none"):
            # The official endpoint defaults to thinking even when temperature
            # is supplied. Disable it explicitly for a genuine temp=0 run.
            request_payload["thinking"] = {"type": "disabled"}
        if self.model.lower() == "mimo-v2.6-flash":
            if self.reasoning_effort != "none":
                raise ValueError("MiMo benchmark requires reasoning_effort='none'")
            request_payload["thinking"] = {"type": "disabled"}
        if self.model.lower() == "glm-5.3-flash":
            if self.reasoning_effort not in {"low", "high", "max"}:
                raise ValueError("GLM-5.3-Flash requires reasoning_effort low, high, or max")
            request_payload["thinking"] = {"type": "enabled"}
            request_payload["reasoning_effort"] = self.reasoning_effort
        use_json_mode = self.json_mode if json_mode is None else json_mode
        if use_json_mode:
            request_payload["response_format"] = {"type": "json_object"}
        return request_payload

    def complete(
        self, *, system: str, user: str, json_mode: bool | None = None
    ) -> str:
        request_payload = self._build_request_payload(
            system=system, user=user, json_mode=json_mode
        )
        body = json.dumps(request_payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            retry_delay = float(2**attempt)
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.last_response_metadata = record_completion(request_payload, payload)
                content = payload["choices"][0]["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM endpoint returned an empty message.content")
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
                if exc.code == 429:
                    suggested = None
                    for header in ("retry-after-ms", "Retry-After"):
                        raw = exc.headers.get(header)
                        if raw:
                            try:
                                suggested = float(raw) / (1000 if header.endswith("-ms") else 1)
                                break
                            except ValueError:
                                pass
                    if suggested is None:
                        match = re.search(
                            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)",
                            detail,
                            flags=re.IGNORECASE,
                        )
                        if match:
                            suggested = float(match.group(1))
                            if match.group(2).lower() == "ms":
                                suggested /= 1000
                    if suggested is not None:
                        retry_delay = max(retry_delay, suggested + 0.5)
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
            if attempt + 1 < self.max_retries:
                # Small jitter prevents concurrent branches from retrying in lockstep.
                time.sleep(min(retry_delay + random.uniform(0.1, 0.75), 60.0))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")
