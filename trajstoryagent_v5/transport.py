"""Bounded retry for connection drops omitted by the legacy HTTP client."""
import http.client
import logging
import time
from trajstoryagent_v5.llm import OpenAICompatibleClient

class ResilientClient(OpenAICompatibleClient):
    def complete(self, **kwargs):
        for attempt in range(4):
            try:
                return super().complete(**kwargs)
            except (ConnectionError, http.client.IncompleteRead) as exc:
                if attempt == 3:
                    raise
                logging.getLogger(__name__).warning('Transport retry %s: %s', attempt+1, type(exc).__name__)
                time.sleep(2 ** attempt)
