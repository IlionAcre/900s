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

# You asked for 2012..2025 inclusive.
DEFAULT_YEARS: List[int] = list(range(2012, 2026))

BASE_URL = "https://projects.propublica.org"


# ============================================================
# HELPERS
# ============================================================

_WS_RE = re.compile(r"\s+")
_TRAIL_SEMI_RE = re.compile(r";\s*$")
_MONEY_CLEAN_RE = re.compile(r"[,\s$]")


def clean(s: Optional[str]) -> Optional[str]:
    """Normalize whitespace and trim. Returns None if empty."""
    if not s:
        return None
    s2 = _WS_RE.sub(" ", str(s)).strip()
    return s2 or None


def to_int(x: Any) -> Optional[int]:
    """
    Best-effort numeric parsing for values like "$1,234", "(123)", "1 234", etc.
    Returns None if parsing fails.
    """
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)

    s = str(x).replace("\u2212", "-")  # unicode minus
    s = _MONEY_CLEAN_RE.sub("", s)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = re.sub(r"[^\d\-]", "", s)
    if not s or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def abs_url(href: str) -> str:
    """Join relative links to BASE_URL."""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE_URL + href


# ============================================================
# JSON-LD (CSVW) DATASET PARSING
# ============================================================

def _loads_json_obj(script_text: str) -> Optional[dict]:
    s = (script_text or "").strip()
    if not s.startswith("{"):
        return None
    s = _TRAIL_SEMI_RE.sub("", s)
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_csvw_dataset_from_script_text(script_text: str) -> Dict[str, Dict[str, Any]]:
    """
    ProPublica pages embed a JSON-LD dataset with csvw:columns.
    We return: {column_name: {"value": ..., "notes": ...}}
    """
    data = _loads_json_obj(script_text)
    if not data:
        return {}

    main = data.get("mainEntity") or {}
    schema = (main.get("csvw:tableSchema") or {})
    cols = schema.get("csvw:columns") or []
    out: Dict[str, Dict[str, Any]] = {}

    for c in cols:
        name = c.get("csvw:name")
        cells = c.get("csvw:cells") or []
        val = None
        notes = None
        if isinstance(cells, list) and cells:
            cell0 = cells[0] or {}
            val = cell0.get("csvw:value")
            notes = cell0.get("csvw:notes")
        if name and val is not None:
            out[str(name)] = {"value": val, "notes": notes}

    return out


def parse_first_csvw_dataset(doc) -> Dict[str, Dict[str, Any]]:
    """Find the first JSON-LD CSVW dataset anywhere in the doc."""
    for script_text in doc.xpath("//script[@type='application/ld+json']/text()"):
        s = (script_text or "").strip()
        if '"csvw:tableSchema"' in s and '"csvw:columns"' in s:
            parsed = parse_csvw_dataset_from_script_text(s)
            if parsed:
                return parsed
    return {}


def parse_csvw_dataset_in_node(node) -> Dict[str, Dict[str, Any]]:
    """Find the first CSVW JSON-LD dataset within a specific node."""
    for script_text in node.xpath(".//script[@type='application/ld+json']/text()"):
        s = (script_text or "").strip()
        if '"csvw:tableSchema"' in s and '"csvw:columns"' in s:
            parsed = parse_csvw_dataset_from_script_text(s)
            if parsed:
                return parsed
    return {}


def ds_value(ds: Dict[str, Dict[str, Any]], key: str) -> Optional[Any]:
    d = ds.get(key)
    if not d:
        return None
    return d.get("value")


def ds_notes(ds: Dict[str, Dict[str, Any]], key: str) -> Optional[Any]:
    d = ds.get(key)
    if not d:
        return None
    return d.get("notes")


# ============================================================
# SUMMARY CHARTS (YEARLY SERIES)
# ============================================================

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

        # "Revenue (2012-2023)" -> "Revenue"
        label = re.sub(r"\s+\(.*?\)\s*$", "", label)
        label = label.split("$")[0].strip()

        charts[label][year] = val

    return charts


# ============================================================
# FILING PERIODS (DETAILED YEAR BLOCKS)
# ============================================================

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_table_kv(table) -> Dict[str, Optional[int]]:
    """
    Convert tables like 'revenue', 'expenses', 'assets-debt' into label->int(value).
    Uses the first two <td> cells.
    """
    out: Dict[str, Optional[int]] = {}
    for tr in table.xpath(".//tbody/tr"):
        tds = tr.xpath("./td")
        if len(tds) < 2:
            continue
        label = clean(" ".join(tds[0].xpath(".//text()")))
        val_text = clean(" ".join(tds[1].xpath(".//text()")))
        if not label:
            continue
        out[label] = to_int(val_text)
    return out


def _parse_accounting_end_month(section) -> Optional[int]:
    """Parse 'Fiscal Year Ending Dec.' -> 12 (if present)."""
    txt = clean(" ".join(section.xpath('.//span[contains(@class,"small-label")]/text()')))
    if not txt:
        return None
    m = re.search(r"Ending\s+([A-Za-z]{3})\.?", txt)
    if not m:
        return None
    return _MONTHS.get(m.group(1)[:3])


def _parse_first_officer(section) -> Tuple[Optional[str], Optional[str]]:
    """
    From the 'Key Employees and Officers' table, take the first row:
      'Josephine Hopkins (Settlor & Co Trustee)' -> (name, title)
    """
    for t in section.xpath('.//table[contains(@class,"compensation")]'):
        head = clean(" ".join(t.xpath(".//thead//th[1]//text()")))
        if head and "Key Employees" in head:
            txt = clean(" ".join(t.xpath(".//tbody/tr[1]/td[1]//text()")))
            if not txt:
                return None, None
            m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", txt)
            if m:
                return clean(m.group(1)), clean(m.group(2))
            return txt, None
    return None, None


def parse_filing_periods(doc) -> Dict[int, Dict[str, Any]]:
    """
    Parse each filing-period year block under div.filing-periods.

    Returns:
      {year: {
         "_dataset": { ... csvw dataset ... },
         "_revenue_table": {...},
         "_expenses_table": {...},
         "_assets_table": {...},
         "_AccountingEndMonth": 12,
         "_OfficerName": "...",
         "_OfficerTitle": "...",
      }}
    """
    out: Dict[int, Dict[str, Any]] = {}

    for sec in doc.xpath('//div[contains(@class,"filing-periods")]//section[contains(@class,"single-filing-period")]'):
        year_txt = clean(" ".join(sec.xpath('.//div[contains(@class,"year-label")]/text()')))
        if not year_txt or not year_txt.isdigit():
            continue
        year = int(year_txt)

        item: Dict[str, Any] = {}
        item["_dataset"] = parse_csvw_dataset_in_node(sec)

        rev_agg: Dict[str, Optional[int]] = {}
        exp_agg: Dict[str, Optional[int]] = {}
        assets_agg: Dict[str, Optional[int]] = {}

        for t in sec.xpath('.//table[contains(@class,"revenue") and contains(@class,"table--small")]'):
            rev_agg.update(_parse_table_kv(t))
        for t in sec.xpath('.//table[contains(@class,"expenses") and contains(@class,"table--small")]'):
            exp_agg.update(_parse_table_kv(t))
        for t in sec.xpath('.//table[contains(@class,"assets-debt") and contains(@class,"table--small")]'):
            assets_agg.update(_parse_table_kv(t))

        if rev_agg:
            item["_revenue_table"] = rev_agg
        if exp_agg:
            item["_expenses_table"] = exp_agg
        if assets_agg:
            item["_assets_table"] = assets_agg

        end_m = _parse_accounting_end_month(sec)
        if end_m:
            item["_AccountingEndMonth"] = end_m

        off_name, off_title = _parse_first_officer(sec)
        if off_name:
            item["_OfficerName"] = off_name
        if off_title:
            item["_OfficerTitle"] = off_title

        out[year] = item

    return out


# ============================================================
# DOCUMENT LINKS + 990-PF CHECKS
# ============================================================

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


def is_990pf(html_path: Path, *, encoding: str = "utf-8") -> bool:
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return is_990pf_from_html_text(html_text)


def extract_xml_link_from_html_text(html_text: str) -> str:
    """
    Extract the XML download link from the FIRST document-links section.

    Returns:
        The href value of the XML link (string), usually a relative path.
    """
    doc = lhtml.fromstring(html_text)

    sections = doc.xpath('//section[contains(@class, "document-links")]')
    if not sections:
        raise ValueError("No document-links section found in HTML")

    # Prefer the explicit "XML" link label if present
    xml_links = sections[0].xpath('.//a[normalize-space(text())="XML"]/@href')
    if not xml_links:
        # Fallback to older patterns
        xml_links = sections[0].xpath('.//a[contains(@href, "download-xml")]/@href')

    if not xml_links:
        raise ValueError("No XML download link found in first document-links section")

    return xml_links[0]


def extract_xml_link(html_path: Path, *, encoding: str = "utf-8") -> str:
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return extract_xml_link_from_html_text(html_text)


def extract_pdf_link_from_html_text(html_text: str) -> str:
    """
    Extract the PDF link from the FIRST document-links section.
    (This is what you want for Form990PDFFilesLocation.)
    """
    doc = lhtml.fromstring(html_text)
    sections = doc.xpath('//section[contains(@class, "document-links")]')
    if not sections:
        raise ValueError("No document-links section found in HTML")

    pdf_links = sections[0].xpath('.//a[normalize-space(text())="PDF"]/@href')
    if not pdf_links:
        pdf_links = sections[0].xpath('.//a[contains(translate(normalize-space(text()), "PDF", "pdf"), "pdf")]/@href')
    if not pdf_links:
        raise ValueError("No PDF link found in first document-links section")

    return abs_url(pdf_links[0])


def extract_pdf_link(html_path: Path, *, encoding: str = "utf-8") -> str:
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return extract_pdf_link_from_html_text(html_text)


def parse_historical_data_from_html_text(
    html_text: str,
    *,
    years: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """
    Backwards-compatible alias.
    Historically this project called the org-page parser
    'parse_historical_data_from_html_text'.
    """
    return parse_org_page_from_html_text(html_text, years=years)
# ============================================================
# OUTPUT SCHEMA (ONLY COLUMNS WE CAN ACTUALLY FILL FROM HTML)
# ============================================================

@dataclass(frozen=True)
class SchemaRow:
    column_name: str
    source: str


def build_output_schema(years: List[int]) -> List[SchemaRow]:
    """
    Build a schema for columns that we can populate from ProPublica org HTML.

    Per your instruction, we do NOT include columns that will always be empty
    from these HTML pages (e.g., County, phone numbers, etc.).
    """
    schema: List[SchemaRow] = [
        SchemaRow("OrganizationName", 'header: h1.text-hed-900'),
        SchemaRow("SecondaryName", 'header: div.org-sort-name OR JSON-LD "Organization secondary name"'),
        SchemaRow("EmployerIdentificationNumber", 'header list: "EIN: XX-XXXXXXX"'),
        SchemaRow("Form990PDFFilesLocation", 'first document-links section: link text "PDF"'),
        SchemaRow("TaxExemptSince", 'header list: "Tax-exempt since MMM. YYYY"'),
        SchemaRow("AssetsLatest", "derived: Assets{latest_year}"),
        SchemaRow("RevenuesLatest", "derived: Revenues{latest_year}"),
        SchemaRow("IncomeLatest", "derived: Income{latest_year}"),
        SchemaRow("Address", 'JSON-LD: "Organization address"'),
        SchemaRow("City", 'header list OR JSON-LD: "Organization city"'),
        SchemaRow("State", 'header list OR JSON-LD: "Organization state"'),
        SchemaRow("ZipCode", 'JSON-LD: "Organization zip code"'),
        SchemaRow("OfficerName", 'latest filing period: first row in "Key Employees and Officers" table (if present)'),
        SchemaRow("OfficerTitle", 'latest filing period: parsed from parentheses in officer cell (if present)'),
        SchemaRow("TaxExemptCategory", 'header "501(c)(3)" OR JSON-LD: "Tax code designation"'),
        SchemaRow("RulingDate", 'JSON-LD: "Ruling date of organization\'s tax exempt status"'),
        SchemaRow("Deductibility", 'JSON-LD: "Tax deductibility of donations to organization"'),
        SchemaRow("TaxPeriod", 'JSON-LD: "End of fiscal year for most recent extracted data"'),
        SchemaRow("AccountingEndMonth", 'latest filing period header: "Fiscal Year Ending XXX" -> month number'),
        SchemaRow("TaxonomyCategory", 'JSON-LD: "Organization NTEE Code" notes (preferred) or value'),
        SchemaRow("ExecutiveCompensation", 'latest filing period: notable expenses table OR JSON-LD "Executive compensation ($)"'),
        SchemaRow("InvestmentIncome", 'latest filing period: notable revenue table OR JSON-LD "Investment income ($)"'),
        SchemaRow("GrossRentalIncome", 'latest filing period: notable revenue table OR JSON-LD "Gross rental income ($)"'),
    ]

    for y in years:
        schema.extend(
            [
                SchemaRow(f"Assets{y}", 'summary chart "Total Assets" OR filing period "Assets/Debt" table'),
                SchemaRow(f"Revenues{y}", 'summary chart "Revenue" OR filing period summary/table'),
                SchemaRow(f"Expenses{y}", 'summary chart "Expenses" OR filing period summary/table'),
                SchemaRow(f"Liabilities{y}", 'summary chart "Total Liabilities" OR filing period "Assets/Debt" table'),
                SchemaRow(f"Income{y}", 'JSON-LD "Net income ($)" OR derived: RevenuesYYYY - ExpensesYYYY'),
                SchemaRow(f"ContributionsReceived{y}", 'filing period JSON-LD "Contributions ($)" OR notable revenue row "Contributions"'),
            ]
        )
    return schema


# ============================================================
# PURE FUNCTION: PARSE ORG PAGE HTML -> ROW
# ============================================================

def parse_org_page_from_html_text(
    html_text: str,
    *,
    years: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """
    PURE FUNCTION.

    Given HTML text (ProPublica org page saved as result.html), returns:
      - row: {column_name: value}  (omits keys we could not find)
      - meta: {
          "column_map": {column_name: source_string},
          "columns": tuple[str, ...],
          "schema": list[dict[str, str]],
        }

    No file writing. No printing.
    """
    years = years or DEFAULT_YEARS
    doc = lhtml.fromstring(html_text)

    schema_rows = build_output_schema(years)
    schema = [{"column_name": r.column_name, "source": r.source} for r in schema_rows]

    # ----------------------------
    # Header metadata
    # ----------------------------
    row: Dict[str, Any] = {}

    org_name = clean(" ".join(doc.xpath('//h1[contains(@class,"text-hed-900")]/text()')))
    if org_name:
        row["OrganizationName"] = org_name

    secondary = clean(" ".join(doc.xpath('//div[contains(@class,"org-sort-name")]/text()')))
    if secondary:
        row["SecondaryName"] = secondary

    # City, ST are usually in the first item of the basic metadata list.
    loc = clean(" ".join(doc.xpath('//ul[contains(@class,"basic-org-metadata")]//li[1]//text()')))
    if loc and "," in loc:
        city, st = [p.strip() for p in loc.split(",", 1)]
        if city:
            row["City"] = city
        if st:
            row["State"] = st

    ein_line = clean(" ".join(doc.xpath('//ul[contains(@class,"basic-org-metadata")]//li[contains(., "EIN")]/text()')))
    m = re.search(r"EIN:\s*([0-9]{2}-?[0-9]{7})", ein_line or "")
    if m:
        row["EmployerIdentificationNumber"] = m.group(1)

    tax_exempt_since = clean(" ".join(doc.xpath('//ul[contains(@class,"basic-org-metadata")]//li[contains(., "Tax-exempt since")]/text()')))
    if tax_exempt_since:
        row["TaxExemptSince"] = tax_exempt_since.replace("Tax-exempt since", "").strip()

    # 501(c)(3), etc.
    cat = clean(" ".join(doc.xpath('//section[contains(@class,"org-top")]//label[contains(@class,"code-definitions-trigger")]/text()')))
    if cat:
        row["TaxExemptCategory"] = cat

    # ----------------------------
    # Document links (first filing)
    # ----------------------------
    sections = doc.xpath('//section[contains(@class, "document-links")]')
    if sections:
        first_links = sections[0]
        pdf = first_links.xpath('.//a[normalize-space(text())="PDF"]/@href')
        if pdf:
            row["Form990PDFFilesLocation"] = abs_url(pdf[0])

    # ----------------------------
    # Yearly series (charts) + filing period details
    # ----------------------------
    charts = parse_summary_charts(doc)
    filing = parse_filing_periods(doc)

    # Fill from summary charts first (best coverage across years)
    label_to_prefix = {
        "Total Assets": "Assets",
        "Revenue": "Revenues",
        "Expenses": "Expenses",
        "Total Liabilities": "Liabilities",
    }
    for label, prefix in label_to_prefix.items():
        series = charts.get(label, {})
        for y in years:
            if y in series:
                row[f"{prefix}{y}"] = series[y]

    # Fill from filing periods (can add contributions, override missing chart data)
    for y, item in filing.items():
        if y not in years:
            continue

        ds = item.get("_dataset") or {}
        rev_table = item.get("_revenue_table") or {}
        exp_table = item.get("_expenses_table") or {}
        assets_table = item.get("_assets_table") or {}

        def pick_int(*keys: str) -> Optional[int]:
            for k in keys:
                v = ds_value(ds, k)
                vi = to_int(v) if v is not None else None
                if vi is not None:
                    return vi
            return None

        # Revenue / Expenses / Assets / Liabilities
        rev = pick_int("Total revenue ($)", "Revenue ($)") or rev_table.get("Revenue") or rev_table.get("Total Revenue")
        exp = pick_int("Total expenses ($)", "Expenses ($)") or exp_table.get("Expenses") or exp_table.get("Total Expenses")
        assets = pick_int("Total assets ($)") or assets_table.get("Total Assets")
        liab = pick_int("Total liabilities ($)") or assets_table.get("Total Liabilities")

        if rev is not None:
            row[f"Revenues{y}"] = rev
        if exp is not None:
            row[f"Expenses{y}"] = exp
        if assets is not None:
            row[f"Assets{y}"] = assets
        if liab is not None:
            row[f"Liabilities{y}"] = liab

        contrib = pick_int("Contributions ($)", "Contributions received ($)") or rev_table.get("Contributions") or rev_table.get("Contributions Received")
        if contrib is not None:
            row[f"ContributionsReceived{y}"] = contrib

        income = pick_int("Net income ($)")
        if income is None and rev is not None and exp is not None:
            income = rev - exp
        if income is not None:
            row[f"Income{y}"] = income

    # Derive IncomeYYYY where missing but revenue+expenses exist
    for y in years:
        ky = f"Income{y}"
        if ky in row:
            continue
        r = row.get(f"Revenues{y}")
        e = row.get(f"Expenses{y}")
        if r is not None and e is not None:
            row[ky] = r - e

    # Latest year = max year that has at least one of Assets/Revenues/Income present
    years_with_data: List[int] = []
    for y in years:
        if any(row.get(f"{p}{y}") is not None for p in ("Assets", "Revenues", "Income")):
            years_with_data.append(y)
    latest_year = max(years_with_data) if years_with_data else None

    if latest_year is not None:
        if f"Assets{latest_year}" in row:
            row["AssetsLatest"] = row[f"Assets{latest_year}"]
        if f"Revenues{latest_year}" in row:
            row["RevenuesLatest"] = row[f"Revenues{latest_year}"]
        if f"Income{latest_year}" in row:
            row["IncomeLatest"] = row[f"Income{latest_year}"]

    # ----------------------------
    # Latest-year metadata from JSON-LD (address, ruling date, etc.)
    # ----------------------------
    ds_latest: Dict[str, Dict[str, Any]] = {}
    if latest_year is not None and latest_year in filing:
        ds_latest = (filing[latest_year].get("_dataset") or {})
    if not ds_latest:
        ds_latest = parse_first_csvw_dataset(doc)

    if ds_latest:
        sec = clean(str(ds_value(ds_latest, "Organization secondary name") or ""))
        if sec and "SecondaryName" not in row:
            row["SecondaryName"] = sec

        addr = clean(str(ds_value(ds_latest, "Organization address") or ""))
        if addr:
            row["Address"] = addr

        city = clean(str(ds_value(ds_latest, "Organization city") or ""))
        if city:
            row["City"] = city

        st = clean(str(ds_value(ds_latest, "Organization state") or ""))
        if st:
            row["State"] = st

        zipc = clean(str(ds_value(ds_latest, "Organization zip code") or ""))
        if zipc:
            row["ZipCode"] = zipc

        ruling = clean(str(ds_value(ds_latest, "Ruling date of organization's tax exempt status") or ""))
        if ruling:
            row["RulingDate"] = ruling

        ded = clean(str(ds_value(ds_latest, "Tax deductibility of donations to organization") or ""))
        if ded:
            row["Deductibility"] = ded

        cat2 = clean(str(ds_value(ds_latest, "Tax code designation") or ""))
        if cat2 and "TaxExemptCategory" not in row:
            row["TaxExemptCategory"] = cat2

        tax_period = clean(str(ds_value(ds_latest, "End of fiscal year for most recent extracted data") or ""))
        if tax_period:
            row["TaxPeriod"] = tax_period

        ntee_note = ds_notes(ds_latest, "Organization NTEE Code")
        ntee_val = ds_value(ds_latest, "Organization NTEE Code")
        if ntee_note:
            row["TaxonomyCategory"] = clean(str(ntee_note))
        elif ntee_val:
            row["TaxonomyCategory"] = clean(str(ntee_val))

    # ----------------------------
    # Latest-year "one-off" numeric columns (ExecutiveCompensation, etc.)
    # ----------------------------
    if latest_year is not None and latest_year in filing:
        item = filing[latest_year]
        ds = item.get("_dataset") or {}
        rev_table = item.get("_revenue_table") or {}
        exp_table = item.get("_expenses_table") or {}

        inv_income = to_int(ds_value(ds, "Investment income ($)")) or rev_table.get("Investment Income")
        if inv_income is not None:
            row["InvestmentIncome"] = inv_income

        exec_comp = to_int(ds_value(ds, "Executive compensation ($)")) or exp_table.get("Executive Compensation")
        if exec_comp is not None:
            row["ExecutiveCompensation"] = exec_comp

        gross_rent = to_int(ds_value(ds, "Gross rental income ($)")) or rev_table.get("Rental Property Income")
        if gross_rent is not None:
            row["GrossRentalIncome"] = gross_rent

        if item.get("_OfficerName"):
            row["OfficerName"] = item["_OfficerName"]
        if item.get("_OfficerTitle"):
            row["OfficerTitle"] = item["_OfficerTitle"]

        if item.get("_AccountingEndMonth"):
            row["AccountingEndMonth"] = item["_AccountingEndMonth"]

    # Per your instruction: omit keys we couldn't find (so they don't become always-empty columns)
    row = {k: v for k, v in row.items() if v is not None and v != ""}

    columns = tuple(sr.column_name for sr in schema_rows if sr.column_name in row)
    meta: Dict[str, object] = {
        "column_map": {sr.column_name: sr.source for sr in schema_rows},
        "columns": columns,
        "schema": schema,
    }
    return row, meta


def parse_org_page(
    html_path: Path,
    *,
    years: Optional[List[int]] = None,
    encoding: str = "utf-8",
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """Convenience wrapper (still pure: no writes/prints)."""
    html_text = html_path.read_text(encoding=encoding, errors="ignore")
    return parse_org_page_from_html_text(html_text, years=years)



