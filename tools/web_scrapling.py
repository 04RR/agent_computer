"""Scrapling-based web fetching tools.

Provides three tiers of web fetching:
- web_fetch: Fast HTTP with browser TLS impersonation (AsyncFetcher)
- web_fetch_js: Full browser JS rendering (DynamicFetcher)
- web_fetch_stealth: Stealth anti-bot bypass (StealthyFetcher)
"""

from __future__ import annotations
import json

from tool_registry import Tool, ToolParam, ToolRegistry


def _extract_response(response, css_selector: str | None, max_chars: int) -> str:
    """Extract text from a Scrapling Response, optionally filtered by CSS selector."""
    if css_selector:
        elements = response.css(css_selector)
        text = "\n".join(str(el.get_all_text(separator=" ", strip=True)) for el in elements)
    else:
        text = str(response.get_all_text(separator="\n", strip=True))
    truncated = len(text) > max_chars
    return json.dumps({
        "status_code": response.status,
        "url": response.url,
        "body": text[:max_chars],
        "truncated": truncated,
    })


def register_scrapling_tools(registry: ToolRegistry, workspace: str, allowed: list[str] | None = None) -> None:
    """Register all Scrapling-based web fetching tools. If allowed is given, only register tools in the list."""

    # ─── web_fetch (AsyncFetcher) ───

    async def web_fetch(url: str, css_selector: str | None = None, max_chars: int = 10000) -> str:
        """Fetch a URL via HTTP with browser TLS impersonation."""
        try:
            from scrapling.fetchers import AsyncFetcher
            response = await AsyncFetcher.get(
                url,
                stealthy_headers=True,
                follow_redirects=True,
                timeout=15,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch" in allowed:
        registry.register(Tool(
            name="web_fetch",
            description="Fast HTTP fetch with browser TLS impersonation. Returns raw HTML only — does NOT execute JavaScript.\n\nUSE for: static HTML pages, APIs, text-based sites (wttr.in, arxiv.org, plain documentation), GitHub search pages (github.com/search), raw README files.\n\nDO NOT USE for sites where data is JavaScript-rendered. These include: GitHub repo pages (star counts, issue counts), Reddit, Twitter/X, LinkedIn, Claude.com pricing, OpenAI pricing, AccuWeather, weather.com, Google Search. Those require web_fetch_js.\n\nAlso DO NOT USE google.com/search — Google blocks non-browser clients from this environment. Use the web_search tool instead.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch,
        ))

    # ─── web_fetch_js (DynamicFetcher) ───

    async def web_fetch_js(url: str, css_selector: str | None = None, wait_seconds: int = 3, max_chars: int = 10000) -> str:
        """Fetch a URL with a full browser for JS rendering."""
        try:
            from scrapling.fetchers import DynamicFetcher
            response = await DynamicFetcher.async_fetch(
                url,
                headless=True,
                network_idle=True,
                wait=wait_seconds * 1000,
                timeout=30000,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch_js" in allowed:
        registry.register(Tool(
            name="web_fetch_js",
            description="Full browser rendering with JavaScript execution. Slower (5-10s) but retrieves dynamic content.\n\nUSE for: GitHub repo pages (to get star counts, commit dates), Reddit threads, Twitter/X, LinkedIn, Claude/OpenAI/Anthropic pricing pages, AccuWeather, weather.com, or any site where key data is rendered by JavaScript rather than present in raw HTML.\n\nDO NOT USE for google.com/search — Google has its own bot detection that blocks even full browsers from this network. Use the web_search tool instead.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("wait_seconds", "integer", "Seconds to wait for JS to render (default 3)", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch_js,
        ))

    # ─── web_fetch_stealth (StealthyFetcher) ───

    async def web_fetch_stealth(url: str, css_selector: str | None = None, max_chars: int = 10000) -> str:
        """Fetch a URL with a stealth browser that bypasses bot detection."""
        try:
            from scrapling.fetchers import StealthyFetcher
            response = await StealthyFetcher.async_fetch(
                url,
                headless=True,
                network_idle=True,
                solve_cloudflare=True,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch_stealth" in allowed:
        registry.register(Tool(
            name="web_fetch_stealth",
            description="Stealth browser with Cloudflare challenge bypass. Very slow (30-120s). Last resort.\n\nUSE ONLY when web_fetch_js fails with a Cloudflare challenge page. Not effective against Google's bot detection — do not use for google.com.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch_stealth,
        ))
