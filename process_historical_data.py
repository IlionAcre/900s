from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import html as lhtml


# ============================================================
# CONFIG
# ============================================================
DEFAULT_YEARS = list(range(2012, 2026))  # 2012..2023 inclusive


# ============================================================
# HELPERS
# ============================================================
def clean(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = re.sub(r"[^\d\-]", "", str(x))
    if not s or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_jsonld_dataset(doc) -> Dict[str, str]:
    """
    ProPublica org pages often include a JSON-LD dataset with csvw:columns.
    Extract name->value.
    """
    for script_text in doc.xpath("//script/text()"):
        s = (script_text or "").strip()
        if s.startswith("{") and '"csvw:tableSchema"' in s and '"csvw:columns"' in s:
            try:
                data = json.loads(re.sub(r";\s*$", "", s))
                cols = data["mainEntity"]["csvw:tableSchema"]["csvw:columns"]
                out: Dict[str, str] = {}
                for c in cols:
                    name = c.get("csvw:name")
                    cells = c.get("csvw:cells") or []
                    val = None
                    if isinstance(cells, list) and cells:
                        val = cells[0].get("csvw:value")
                    if name and val is not None:
                        out[name] = str(val)
                return out
            except Exception:
                return {}
    return {}


def parse_summary_charts(doc) -> Dict[str, Dict[int, int]]:
    """
    Extract yearly values from the summary charts.
    Bars usually have:
      - data-range="YYYY"
      - data-count="123456"
    Returns: { "Revenue": {2012:..., ...}, ... }
    """
    charts: Dict[str, Dict[int, int]] = defaultdict(dict)

    for node in doc.xpath('//div[contains(@class,"org-summary-chart")]//*[@data-range and @data-count]'):
        year_s = node.attrib.get("data-range")
        val_s = node.attrib.get("data-count")
        if not year_s or not val_s:
            continue

        try:
            year = int(year_s)
            val = int(val_s)
        except ValueError:
            continue

        anc = node.xpath('ancestor::div[contains(@class,"org-summary-chart")][1]')
        if not anc:
            continue

        h4_nodes = anc[0].xpath(".//h4")
        label = clean(" ".join(h4_nodes[0].xpath(".//text()")) if h4_nodes else None)
        if not label:
            continue

        label = re.sub(r"\s+\(.*?\)\s*$", "", label)
        label = label.split("$")[0].strip()

        charts[label][year] = val

    return charts


# ============================================================
# SCHEMA
# ============================================================
@dataclass(frozen=True)
class SchemaRow:
    column_name: str
    source: str  # a human-readable "path" that explains where it comes from


def build_historical_schema(years: List[int]) -> List[Dict[str, str]]:
    """
    Build schema in the same spirit as your XML schema builder.
    Each column has a source string indicating how it is derived.
    """
    schema: List[Dict[str, str]] = []
    for y in years:
        schema.extend(
            [
                {"column_name": f"Assets{y}", "source": 'result.html chart: label="Total Assets", year=data-range'},
                {"column_name": f"Revenues{y}", "source": 'result.html chart: label="Revenue", year=data-range'},
                {"column_name": f"Expenses{y}", "source": 'result.html chart: label="Expenses", year=data-range'},
                {"column_name": f"Liabilities{y}", "source": 'result.html chart: label="Total Liabilities", year=data-range'},
                {"column_name": f"Income{y}", "source": "derived: RevenuesYYYY - ExpensesYYYY"},
                {"column_name": f"ContributionsReceived{y}", "source": 'result.html JSON-LD: "Contributions ($)" mapped to latest year from "End of fiscal year for most recent extracted data"'},
            ]
        )
    return schema


# ============================================================
# PURE FUNCTION
# ============================================================
def parse_historical_data_from_html_text(
    html_text: str,
    *,
    years: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """
    PURE FUNCTION.

    Given HTML text (ProPublica org page saved as result.html), returns:
      - row: {column_name: value}
      - meta: {
          "column_map": {column_name: source_string},
          "columns": tuple[str, ...],
          "schema": list[dict[str, str]],
        }

    No file writing. No printing.
    """
    years = years or DEFAULT_YEARS
    schema = build_historical_schema(years)
    columns = tuple(r["column_name"] for r in schema)

    doc = lhtml.fromstring(html_text)
    jsonld = parse_jsonld_dataset(doc)
    charts = parse_summary_charts(doc)

    row: Dict[str, Any] = {c: None for c in columns}

    label_to_prefix = {
        "Total Assets": "Assets",
        "Revenue": "Revenues",
        "Expenses": "Expenses",
        "Total Liabilities": "Liabilities",
    }

    # Fill series from charts
    for label, prefix in label_to_prefix.items():
        series = charts.get(label, {})
        for y in years:
            if y in series:
                row[f"{prefix}{y}"] = series[y]

    # Derived IncomeYYYY
    for y in years:
        r = row.get(f"Revenues{y}")
        e = row.get(f"Expenses{y}")
        row[f"Income{y}"] = (r - e) if (r is not None and e is not None) else None

    # ContributionsReceivedYYYY from JSON-LD (typically only one number for the latest year)
    contrib = to_int(jsonld.get("Contributions ($)"))
    if contrib is not None:
        tax_period = clean(jsonld.get("End of fiscal year for most recent extracted data"))
        m = re.search(r"(\d{4})", tax_period or "")
        latest_year = int(m.group(1)) if m else None
        if latest_year in years:
            row[f"ContributionsReceived{latest_year}"] = contrib

    meta: Dict[str, object] = {
        "column_map": {r["column_name"]: r["source"] for r in schema},
        "columns": columns,
        "schema": schema,
    }
    return row, meta


def parse_historical_data(
    html_path: Path,
    *,
    years: Optional[List[int]] = None,
    encoding: str = "utf-8",
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """
    Convenience wrapper (still pure: no writes/prints).
    Reads the HTML file and returns (row, meta).
    """
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return parse_historical_data_from_html_text(html_text, years=years)


def is_990pf_from_html_text(html_text: str) -> bool:
    """
    Return True if the FIRST document-links section is a 990-PF filing.
    Return False if it is not 990-PF.

    Raises:
        ValueError: if no document-links section is found.
    """
    doc = lhtml.fromstring(html_text)

    sections = doc.xpath('//section[contains(@class, "document-links")]')
    if not sections:
        raise ValueError("No document-links section found in HTML")

    h5_text = sections[0].xpath(".//h5/text()")
    if not h5_text:
        return False

    value = (h5_text[0] or "").strip().upper()
    return value in {"990-PF", "990PF"}


def is_990pf(
    html_path: Path,
    *,
    encoding: str = "utf-8",
) -> bool:
    """
    Convenience wrapper (still pure: no writes/prints).
    Reads HTML file and returns whether it is a 990-PF filing.
    """
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return is_990pf_from_html_text(html_text)


def extract_xml_link_from_html_text(html_text: str) -> str:
    """
    Extract the XML download link from the FIRST document-links section.

    Returns:
        The href value of the XML link (string).

    Raises:
        ValueError: if the document-links section or XML link is not found.
    """
    doc = lhtml.fromstring(html_text)

    sections = doc.xpath('//section[contains(@class, "document-links")]')
    if not sections:
        raise ValueError("No document-links section found in HTML")

    xml_links = sections[0].xpath('.//a[contains(@href, "download-xml")]/@href')
    if not xml_links:
        raise ValueError("No XML download link found in first document-links section")

    return xml_links[0]


def extract_xml_link(
    html_path: Path,
    *,
    encoding: str = "utf-8",
) -> str:
    """
    Convenience wrapper (still pure: no writes/prints).
    Reads HTML file and returns the XML download link.
    """
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return extract_xml_link_from_html_text(html_text)