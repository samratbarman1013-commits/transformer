"""Web search tool.

Set BRAVE_API_KEY (or SEARCH_API_KEY) in the environment to enable live
search; without a key the tool returns a graceful error that the agent can
relay to the user instead of crashing the run.

The interface is fixed now so the model side can be built against it; only the
HTTP call inside needs wiring when a provider is chosen.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .registry import ToolSpec

SPEC = ToolSpec(
    name="web_search",
    description="Search the web and return the top results (title, url, snippet). Use for questions about current events or facts you are unsure of.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "count": {"type": "integer", "description": "number of results (default 5)"},
        },
        "required": ["query"],
    },
)

_API_URL = "https://api.search.brave.com/res/v1/web/search"


def web_search(query: str, count: int = 5) -> dict:
    key = os.environ.get("BRAVE_API_KEY") or os.environ.get("SEARCH_API_KEY")
    if not key:
        return {
            "error": "search is not configured on this server (no API key). Tell the user you cannot search right now."
        }

    url = f"{_API_URL}?q={urllib.parse.quote(query)}&count={max(1, min(count, 10))}"
    request = urllib.request.Request(url, headers={"X-Subscription-Token": key, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())

    results = []
    for item in payload.get("web", {}).get("results", [])[:count]:
        results.append({"title": item.get("title"), "url": item.get("url"), "snippet": item.get("description")})
    return {"query": query, "results": results}
