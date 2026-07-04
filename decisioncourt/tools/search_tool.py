"""
Web search tool (DuckDuckGo via the free `ddgs` package — no API key).

FAILURE HANDLING (per the orchestration checklist): search engines
rate-limit and hang. Every call is wrapped: on ANY failure it returns []
instead of raising, and downstream agents are designed to degrade
gracefully — an advocate argues from general knowledge, and the
fact-checker marks unverifiable claims as "unverified" rather than
crashing the run. A short sleep between calls avoids rate-limit bursts.
"""

import time

_LAST_CALL = [0.0]
_MIN_GAP_S = 1.0  # be polite to the search engine


def web_search(query: str, max_results: int = 4) -> list:
    """Return [{"title", "url", "snippet"}, ...]; [] on any failure."""
    gap = time.time() - _LAST_CALL[0]
    if gap < _MIN_GAP_S:
        time.sleep(_MIN_GAP_S - gap)
    _LAST_CALL[0] = time.time()

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"    [search] FAILED for {query!r}: {e} — degrading gracefully")
        return []

    results = []
    for r in raw:
        results.append({
            "title": str(r.get("title", ""))[:120],
            "url": str(r.get("href") or r.get("url") or ""),
            "snippet": str(r.get("body") or r.get("snippet") or "")[:300],
        })
    return results


def render_sources(sources: list) -> str:
    """Format search results for inclusion in a prompt."""
    if not sources:
        return "(no web sources available — reason from general knowledge, and say so)"
    return "\n".join(f"[{i}] {s['title']}\n    {s['snippet']}\n    ({s['url']})"
                     for i, s in enumerate(sources, 1))
