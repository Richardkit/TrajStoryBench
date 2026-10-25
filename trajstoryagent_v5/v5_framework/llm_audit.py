"""Opt-in, metadata-only request audit: no credentials, prompts, or reasoning text."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def record_completion(request: dict, response: dict) -> dict:
    parameters = dict(request)
    parameters.update(request.get("extra_body", {}))
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or ""
    usage = response.get("usage") or {}
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "request": {key: parameters[key] for key in (
            "model", "temperature", "reasoning_effort", "enable_thinking", "thinking",
            "max_tokens", "max_completion_tokens", "response_format", "stream",
            "seed", "top_k", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
        ) if key in parameters},
        "messages_sha256": (hashlib.sha256(json.dumps(
            parameters["messages"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest() if "messages" in parameters else None),
        "content_sha256": hashlib.sha256((message.get("content") or "").encode("utf-8")).hexdigest(),
        "response_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_present": bool(reasoning),
        "reasoning_characters": len(reasoning),
        "content_characters": len(message.get("content") or ""),
        "usage": {key: usage[key] for key in (
            "prompt_tokens", "completion_tokens", "total_tokens", "completion_tokens_details",
        ) if key in usage},
    }
    path = os.environ.get("TRAJSTORY_LLM_AUDIT_PATH")
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
