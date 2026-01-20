from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Optional
from urllib.parse import urlencode, urljoin

from curl_cffi.requests import AsyncSession
from parsel import Selector
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path

from lxml import html as lhtml

BASE = "https://projects.propublica.org"


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


def parse_results(html: str) -> tuple[list[SearchResult], Optional[str]]:
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


async def fetch(session: AsyncSession, url: str) -> str:
    r = await session.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
    r.raise_for_status()
    return r.text


async def scrape_search(q: str, max_pages: int, delay_s: float = 0.0) -> list[SearchResult]:
    start_url = f"{BASE}/nonprofits/search?{urlencode({'q': q})}"

    results: list[SearchResult] = []

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
            "Referer": BASE + "/nonprofits/",
        },
    ) as session:
        url: Optional[str] = start_url
        page = 0
        while url and page < max_pages:
            page += 1
            html = await fetch(session, url)
            page_results, next_url = parse_results(html)
            results.extend(page_results)

            if delay_s:
                await asyncio.sleep(delay_s)

            url = next_url

    return results


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




if __name__ == "__main__":
    main()
    # path = dump_search_html("Gates", "gates_search.html")
    # print("Saved:", path)

    # # 2) Or fetch + parse in one go
    # html = asyncio.run(fetch_html_once(f"{BASE}/nonprofits/search?{urlencode({'q':'Gates'})}", "tmp.html"))
    # items, next_url = parse_results(html)
    # print(len(items), next_url)