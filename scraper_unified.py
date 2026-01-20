"""
Unified Scraper for ProPublica Nonprofit Explorer.

This file combines:
- Search functionality from scraper.py (used by main.py GUI)
- XML sanitization from scraper_900s.py (fixes broken XML with naked '&')
- Cash field fallback extraction from scraper_900s.py

Usage:
  - Import `scrape_search` for search results
  - Import `sanitize_xml_bytes` to fix XML before parsing
  - Import `extract_cash_fallback_from_xml_bytes` and `backfill_cash_fields` for cash extraction
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

from curl_cffi.requests import AsyncSession
from parsel import Selector
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path

from lxml import etree

BASE = "https://projects.propublica.org"

# Default rate limit settings
DEFAULT_RATE_LIMIT_SLEEP_SECONDS = 61
DEFAULT_RATE_LIMIT_MAX_RETRIES = 3

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/nonprofits/",
}


# ============================================================
# Rate-Limit Gate (global pause on 429)
# ============================================================


class RateLimitGate:
    """
    Shared gate across ALL tasks.
    - Each request calls `await gate.wait()` before sending.
    - On HTTP 429, we call `await gate.trigger(seconds)` to pause everyone.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._event.set()
        self._lock = asyncio.Lock()
        self._resume_at: float = 0.0
        self._sleep_task: Optional[asyncio.Task[None]] = None

    async def wait(self) -> None:
        await self._event.wait()

    async def trigger(self, seconds: int) -> None:
        async with self._lock:
            now = time.monotonic()
            new_resume_at = now + float(seconds)

            if new_resume_at <= self._resume_at:
                return

            self._resume_at = new_resume_at
            self._event.clear()

            if self._sleep_task and not self._sleep_task.done():
                self._sleep_task.cancel()

            self._sleep_task = asyncio.create_task(self._sleep_until_resume())

    async def _sleep_until_resume(self) -> None:
        try:
            while True:
                now = time.monotonic()
                remaining = self._resume_at - now
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
        finally:
            self._event.set()


# ============================================================
# Search Result Model (from scraper.py)
# ============================================================


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    org_name: str = Field(..., min_length=1)
    org_url: str = Field(..., min_length=1)
    org_id: Optional[str] = None

    city: Optional[str] = None
    state: Optional[str] = None
    category_text: Optional[str] = None

    revenue_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    revenue_value: Optional[str] = None  # keep raw like "$7,810,137,842"


# ============================================================
# Text Cleaning Helpers
# ============================================================


def _clean_text(s: str | None) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _parse_city_state_and_category(text_sub: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    From:
      "Seattle, WA • Philanthropy, Voluntarism and Grantmaking Foundations - Private Independent Foundations"
    """
    text_sub = _clean_text(text_sub) or ""
    parts = [p.strip() for p in text_sub.split("•")]

    city = state = category = None
    if parts:
        m = re.match(r"^(.*?),\s*([A-Z]{2})$", parts[0])
        if m:
            city = _clean_text(m.group(1))
            state = _clean_text(m.group(2))
        else:
            city = _clean_text(parts[0])

    if len(parts) >= 2:
        category = _clean_text(parts[1])

    return city, state, category


def _parse_revenue(row_sel: Selector) -> tuple[Optional[int], Optional[str]]:
    label = _clean_text(row_sel.css(".metrics-wrapper .text-sub.nowrap::text").get())
    year = None
    if label:
        ym = re.search(r"\((\d{4})\)", label)
        if ym:
            year = int(ym.group(1))

    value = _clean_text(row_sel.css(".metrics-wrapper .font-weight-500::text").get())
    return year, value


# ============================================================
# HTML Parsing (from scraper.py)
# ============================================================


def parse_results(html: str) -> tuple[list[SearchResult], Optional[str]]:
    """
    Parse ProPublica search results HTML page.
    Returns list of SearchResult and optional next page URL.
    """
    sel = Selector(text=html)

    out: list[SearchResult] = []
    for row in sel.css("div.result-row.result-row-org"):
        a = row.css(".result-item__hed a")
        org_name = _clean_text(a.css("::text").get()) or ""
        href = a.attrib.get("href", "")
        org_url = urljoin(BASE, href) if href else ""

        org_id = None
        m = re.search(r"/nonprofits/organizations/(\d+)", href)
        if m:
            org_id = m.group(1)

        # Collect all text nodes inside .text-sub (often broken by newlines / bullets)
        sub_texts = [t for t in row.css(".text-sub::text").getall() if _clean_text(t)]
        text_sub = _clean_text(" ".join(sub_texts)) if sub_texts else None
        city, state, category = _parse_city_state_and_category(text_sub or "")

        revenue_year, revenue_value = _parse_revenue(row)

        if org_name and org_url:
            out.append(
                SearchResult(
                    org_name=org_name,
                    org_url=org_url,
                    org_id=org_id,
                    city=city,
                    state=state,
                    category_text=category,
                    revenue_year=revenue_year,
                    revenue_value=revenue_value,
                )
            )

    # Pagination: try common patterns
    next_href = (
        sel.css('a[rel="next"]::attr(href)').get()
        or sel.css('a:contains("Next")::attr(href)').get()
        or sel.css('a.pagination__next::attr(href)').get()
    )
    next_url = urljoin(BASE, next_href) if next_href else None

    return out, next_url


# ============================================================
# HTTP Helpers with Rate Limit Retry
# ============================================================


async def _get_with_429_retry(
    session: AsyncSession,
    *,
    url: str,
    headers: Dict[str, str],
    gate: Optional[RateLimitGate],
    sleep_seconds: int = DEFAULT_RATE_LIMIT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_RATE_LIMIT_MAX_RETRIES,
):
    """
    GET request with automatic 429 retry logic.
    If a RateLimitGate is provided, it coordinates waiting across all requests.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        if gate:
            await gate.wait()
        try:
            r = await session.get(url, headers=headers)
        except Exception as e:
            last_exc = e
            raise

        if getattr(r, "status_code", None) == 429:
            if gate:
                await gate.trigger(sleep_seconds)
            else:
                await asyncio.sleep(sleep_seconds)
            if attempt == max_retries:
                raise RuntimeError(f"HTTP 429 after {max_retries} attempts: {url}")
            continue

        r.raise_for_status()
        return r

    raise RuntimeError(f"Request failed after {max_retries} attempts: {url}") from last_exc


async def fetch(
    session: AsyncSession,
    url: str,
    *,
    gate: Optional[RateLimitGate] = None,
    sleep_seconds: int = DEFAULT_RATE_LIMIT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_RATE_LIMIT_MAX_RETRIES,
) -> str:
    """Fetch HTML text with 429 retry support."""
    r = await _get_with_429_retry(
        session,
        url=url,
        headers={"Accept": "text/html,application/xhtml+xml"},
        gate=gate,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )
    return r.text


async def fetch_bytes(
    session: AsyncSession,
    url: str,
    *,
    gate: Optional[RateLimitGate] = None,
    sleep_seconds: int = DEFAULT_RATE_LIMIT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_RATE_LIMIT_MAX_RETRIES,
) -> bytes:
    """Fetch bytes (e.g., XML) with 429 retry support."""
    r = await _get_with_429_retry(
        session,
        url=url,
        headers={"Accept": "*/*"},
        gate=gate,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )
    return r.content


async def scrape_search(
    q: str,
    max_pages: int,
    delay_s: float = 0.0,
    *,
    session: Optional[AsyncSession] = None,
    gate: Optional[RateLimitGate] = None,
) -> list[SearchResult]:
    """
    Scrape ProPublica nonprofit search for a query.
    Returns list of SearchResult objects.
    
    If session is provided, uses it (for session sharing).
    If gate is provided, coordinates rate limiting across all requests.
    """
    start_url = f"{BASE}/nonprofits/search?{urlencode({'q': q})}"
    results: list[SearchResult] = []

    # Use provided session or create a new one
    if session is not None:
        url: Optional[str] = start_url
        page = 0
        while url and page < max_pages:
            page += 1
            html = await fetch(session, url, gate=gate)
            page_results, next_url = parse_results(html)
            results.extend(page_results)
            if delay_s:
                await asyncio.sleep(delay_s)
            url = next_url
    else:
        # Create new session (legacy behavior)
        async with AsyncSession(
            impersonate="chrome",
            timeout=30,
            headers=DEFAULT_HEADERS,
        ) as new_session:
            url: Optional[str] = start_url
            page = 0
            while url and page < max_pages:
                page += 1
                html = await fetch(new_session, url, gate=gate)
                page_results, next_url = parse_results(html)
                results.extend(page_results)
                if delay_s:
                    await asyncio.sleep(delay_s)
                url = next_url

    return results


# ============================================================
# XML Sanitization (from scraper_900s.py)
# Fixes broken XML with naked '&' characters like "... A & B ..."
# ============================================================

_AMP_FIX_RE = re.compile(rb"&(?!(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)")


def sanitize_xml_bytes(b: bytes) -> bytes:
    """
    Replace naked '&' with '&amp;', while preserving existing entities.
    Prevents XMLSyntaxError: xmlParseEntityRef: no name.
    
    Use this BEFORE writing XML to disk or parsing it with lxml.
    """
    return _AMP_FIX_RE.sub(b"&amp;", b)


# ============================================================
# Cash Field Extraction Fallback (from scraper_900s.py)
# Extracts cash fields even when the main parser misses them
# ============================================================

CASH_COLUMNS = (
    "Cash Non Interest Bearing Group - EOY Amount",
    "Cash Non Interest Bearing Group - BOY Amount",
    "Cash Eoyfmv Amt",
    "Cash Boy Amt",
)


def _is_blankish_cash(v: Any) -> bool:
    """
    Check if a cash value is "missing" (None, empty, or '-').
    Note: '0' is NOT missing.
    """
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s == "-"


def has_any_cash_column(row: Dict[str, Any]) -> bool:
    """Check if any cash column key exists in the row."""
    return any(col in row for col in CASH_COLUMNS)


def has_any_cash_value(row: Dict[str, Any]) -> bool:
    """Check if any cash column has a real value (not '' and not '-')."""
    for col in CASH_COLUMNS:
        if col in row and not _is_blankish_cash(row.get(col)):
            return True
    return False


def _first_text(root: etree._Element, xpath: str) -> Optional[str]:
    """
    Return first matched text for xpath, stripped. None if not found/empty.
    XPath should use local-name() to ignore namespaces.
    """
    try:
        vals = root.xpath(xpath)
    except Exception:
        return None
    if not vals:
        return None

    v0 = vals[0]
    if isinstance(v0, str):
        s = v0.strip()
        return s or None
    if hasattr(v0, "text"):
        s = (v0.text or "").strip()
        return s or None
    return None


def extract_cash_fallback_from_xml_bytes(xml_bytes: bytes) -> Dict[str, str]:
    """
    Fallback extractor that finds cash fields even when your schema misses them.
    It ignores namespaces and searches by tag local-name().
    
    Call this on sanitized XML bytes, pass result to backfill_cash_fields().
    """
    out: Dict[str, str] = {}
    parser = etree.XMLParser(recover=True, huge_tree=True)

    try:
        root = etree.fromstring(xml_bytes, parser=parser)
    except Exception:
        return out

    # 990: CashNonInterestBearingGrp/BOYAmt + EOYAmt
    boy = _first_text(
        root,
        "//*[local-name()='CashNonInterestBearingGrp']/*[local-name()='BOYAmt']/text()",
    )
    eoy = _first_text(
        root,
        "//*[local-name()='CashNonInterestBearingGrp']/*[local-name()='EOYAmt']/text()",
    )

    if boy is not None:
        out["Cash Non Interest Bearing Group - BOY Amount"] = boy
    if eoy is not None:
        out["Cash Non Interest Bearing Group - EOY Amount"] = eoy

    # 990-PF best-effort (varies by schema)
    pf_eoy = (
        _first_text(root, "//*[local-name()='CashEoyfmvAmt']/text()")
        or _first_text(root, "//*[local-name()='CashEOYFMVAmt']/text()")
        or _first_text(root, "//*[local-name()='CashEOYAmt']/text()")
    )
    pf_boy = (
        _first_text(root, "//*[local-name()='CashBoyAmt']/text()")
        or _first_text(root, "//*[local-name()='CashBOYAmt']/text()")
        or _first_text(root, "//*[local-name()='CashBOYFMVAmt']/text()")
    )

    if pf_eoy is not None:
        out["Cash Eoyfmv Amt"] = pf_eoy
    if pf_boy is not None:
        out["Cash Boy Amt"] = pf_boy

    return out


def backfill_cash_fields(row: Dict[str, Any], *, fallback: Dict[str, str]) -> None:
    """
    Backfill cash fields into an existing row:
      - if key missing -> set it
      - if key present but blankish (None/""/"-") -> replace it
    """
    for k, v in fallback.items():
        if v is None:
            continue
        if k not in row or _is_blankish_cash(row.get(k)):
            row[k] = v


# ============================================================
# Convenience Functions
# ============================================================


async def fetch_html_once(url: str, out_html: str = "page.html") -> str:
    """
    Fetch a single URL and save the raw HTML to `out_html`.
    Returns the HTML string so you can also pass it to parse_results().
    """
    async with AsyncSession(
        impersonate="chrome",
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
            "Referer": BASE + "/nonprofits/",
        },
    ) as session:
        r = await session.get(url)
        r.raise_for_status()
        html = r.text

    Path(out_html).write_text(html, encoding="utf-8")
    return html


def dump_search_html(q: str, out_html: str = "search.html") -> str:
    """
    Convenience wrapper: builds the ProPublica search URL for q,
    fetches once, writes HTML to disk, and returns the saved path.
    """
    url = f"{BASE}/nonprofits/search?{urlencode({'q': q})}"
    asyncio.run(fetch_html_once(url, out_html=out_html))
    return out_html


# ============================================================
# CLI
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help='Search query (e.g. "Gates")')
    ap.add_argument("--pages", type=int, default=1, help="Max pages to scrape (default: 1)")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between pages in seconds")
    ap.add_argument("--out", default="", help="Output path (.jsonl). If empty, prints JSON to stdout.")
    args = ap.parse_args()

    items = asyncio.run(scrape_search(args.q, args.pages, args.delay))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.model_dump_json() + "\n")
        print(f"Wrote {len(items)} results -> {args.out}")
    else:
        print(json.dumps([x.model_dump() for x in items], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
