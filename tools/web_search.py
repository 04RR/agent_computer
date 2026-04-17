"""DuckDuckGo HTML search tool.

Uses the html.duckduckgo.com endpoint which returns scrape-friendly static HTML
(no JS required). Rate-limited but reliable for typical agent use.
"""

from __future__ import annotations
import json
import logging
from typing import Any
from urllib.parse import urlparse, parse_qs, unquote

from tool_registry import Tool, ToolParam, ToolRegistry

logger = logging.getLogger("agent_computer.tools.web_search")


async def _web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (1-10, default 5).

    Returns:
        JSON string with query, results list, and count.
    """
    from scrapling.fetchers import AsyncFetcher

    max_results = max(1, min(10, max_results))
    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"

    try:
        page = await AsyncFetcher.get(
            url,
            stealthy_headers=True,
            follow_redirects=True,
            timeout=15,
        )
    except Exception as e:
        logger.warning(f"web_search failed: {e}")
        return json.dumps({"query": query, "results": [], "count": 0, "error": str(e)})

    if page.status != 200:
        return json.dumps({
            "query": query,
            "results": [],
            "count": 0,
            "error": f"HTTP {page.status}",
        })

    # DuckDuckGo HTML layout: each result is a div.result with nested
    # a.result__a (title+url) and a.result__snippet (snippet text)
    results = []
    for result_el in page.css("div.result")[:max_results]:
        title_els = result_el.css("a.result__a")
        snippet_els = result_el.css("a.result__snippet")

        if not title_els:
            continue

        title = title_els[0].text.strip()
        raw_url = title_els[0].attrib.get("href", "")
        real_url = _unwrap_ddg_redirect(raw_url)
        snippet = snippet_els[0].text.strip() if snippet_els else ""

        results.append({
            "title": title,
            "url": real_url,
            "snippet": snippet,
        })

    return json.dumps({
        "query": query,
        "results": results,
        "count": len(results),
    })


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps outbound URLs like //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com. Extract the real URL."""
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href  # Already a direct URL

    try:
        # Handle protocol-relative URLs (//duckduckgo.com/l/?uddg=...)
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    except Exception:
        pass
    return href


def register_web_search_tool(registry: ToolRegistry, allowed: list[str] | None = None) -> None:
    """Register the web_search tool."""
    if allowed is not None and "web_search" not in allowed:
        return

    registry.register(Tool(
        name="web_search",
        description=(
            "Search the web via DuckDuckGo. Returns up to 5 search results with title, URL, and snippet. "
            "USE THIS for any lookup task where you don't already know the target URL. "
            "Much more reliable than fetching google.com/search, which is blocked in this environment. "
            "After searching, use web_fetch or web_fetch_js on the specific URLs you want to read in detail."
        ),
        params=[
            ToolParam("query", "string", "Search query"),
            ToolParam("max_results", "integer", "Max results to return (1-10, default 5)", required=False),
        ],
        handler=_web_search,
    ))
