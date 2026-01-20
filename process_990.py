from __future__ import annotations

import csv
import json
import re
from io import BytesIO
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from lxml import etree


# ============================================================
# CONFIG
# ============================================================
FORMAT_XML_PATH = Path("format.xml")

OUTPUT_CSV_PATH = Path("flattened_format.csv")
OUTPUT_COLUMN_MAP_JSON = Path("column_map.json")
OUTPUT_COLUMN_MAP_REVERSE_JSON = Path("column_map_reverse.json")


# ============================================================
# HELPERS
# ============================================================
def clean_text(s: Optional[str]) -> Optional[str]:
    """Normalize whitespace and return None for empty strings."""
    if s is None:
        return None
    s2 = " ".join(s.split()).strip()
    return s2 or None


def local_name(tag: str) -> str:
    """Strip XML namespace from tag name if present."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


@dataclass(frozen=True)
class FlattenOptions:
    include_attributes: bool = True
    include_empty: bool = False
    keep_mixed_content_text: bool = False


# ============================================================
# CORE: FLATTEN XML INTO A SINGLE ROW (FULL PATH KEYS)
# ============================================================
def flatten_xml_to_row(
    root: etree._Element,
    *,
    opts: FlattenOptions = FlattenOptions(),
) -> Dict[str, str]:
    """
    Flatten an XML document into a single dict (CSV-ready).
    Keys are full XML paths.
    """
    out: Dict[str, str] = {}

    def add_value(key: str, value: Optional[str]) -> None:
        v = clean_text(value)
        if v is None and not opts.include_empty:
            return
        out[key] = v if v is not None else ""

    def walk(elem: etree._Element, path: str) -> None:
        # Attributes
        if opts.include_attributes and elem.attrib:
            for attr_name, attr_val in elem.attrib.items():
                add_value(f"{path}/@{local_name(attr_name)}", attr_val)

        children = [c for c in elem if isinstance(c.tag, str)]
        text = clean_text(elem.text)

        if not children:
            add_value(path, text)
            return

        if opts.keep_mixed_content_text and text is not None:
            add_value(path, text)

        # Repeated siblings get indexed
        counts: defaultdict[str, int] = defaultdict(int)
        for child in children:
            counts[local_name(child.tag)] += 1

        seen: defaultdict[str, int] = defaultdict(int)
        for child in children:
            tag = local_name(child.tag)
            seen[tag] += 1
            child_path = (
                f"{path}/{tag}[{seen[tag]}]"
                if counts[tag] > 1
                else f"{path}/{tag}"
            )
            walk(child, child_path)

    root_path = f"/{local_name(root.tag)}"
    walk(root, root_path)
    return out


def parse_xml_file_to_row(
    xml_path: Path,
    *,
    opts: FlattenOptions = FlattenOptions(),
) -> Dict[str, str]:
    """Parse XML file and flatten it into one row."""
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=True)
    root = etree.parse(str(xml_path), parser=parser).getroot()
    return flatten_xml_to_row(root, opts=opts)


# ============================================================
# NAMING: SMART LABELS + ABBREVIATION POLISH + CONTEXT RULES
# ============================================================
# Exact acronyms to preserve as-is
_GLOSSARY_EXACT = {
    "EIN": "EIN",
    "PTIN": "PTIN",
    "ZIP": "ZIP",
    "BOY": "BOY",
    "EOY": "EOY",
    "IRS": "IRS",
    "W2G": "W2G",
    "UBI": "UBI",
    "FASB": "FASB",
    "CEO": "CEO",
}

# Suffix expansions when token endswith these
_GLOSSARY_SUFFIX = {
    "Amt": "Amount",
    "Ind": "Indicator",
    "Cnt": "Count",
    "Cd": "Code",
    "Nm": "Name",
    "Txt": "Text",
    "Dt": "Date",
    "Yr": "Year",
    "Pct": "Percent",
    "Num": "Number",
    "Rt": "Rate",
    "Desc": "Description",
}

# Standalone token expansions (fixes cases like "... Ind" / "... Amt")
_GLOSSARY_STANDALONE = {
    "Amt": "Amount",
    "Ind": "Indicator",
    "Cnt": "Count",
    "Cd": "Code",
    "Nm": "Name",
    "Txt": "Text",
    "Dt": "Date",
    "Yr": "Year",
    "Pct": "Percent",
    "Num": "Number",
    "Rt": "Rate",
    "Desc": "Description",
    "Grp": "Group",
}

# Abbreviation word map (domain-ish cleanup)
_ABBREV_WORD_MAP = {
    "Prtshp": "Partnership",
    "Wthld": "Withholding",
    "Rln": "Relationship",
    "Ctrl": "Control",
    "Ent": "Entity",
    "Fincl": "Financial",
    "Stmt": "Statement",
    "Gvrn": "Governing",
    "Acty": "Activity",
    "Expnss": "Expenses",
    "Expns": "Expenses",
    "Rltd": "Related",
    "Srvc": "Service",
    "Accom": "Accomplishment",
    "Invst": "Investment",
    "Txbl": "Taxable",
    "Incm": "Income",
    "Excs": "Excise",
    "Chrtbl": "Charitable",
    "Nonchrtbl": "Non Charitable",
    "Trnsfr": "Transfer",
    "Exmpt": "Exempt",
    "Rmnrtn": "Remuneration",
    "Prcht": "Purchase",
    "Pymt": "Payment",
    "Ofcr": "Officer",
    "Bnft": "Benefit",
    "Prof": "Professional",
    "Fndrsng": "Fundraising",
    "Oth": "Other",
    "Defrd": "Deferred",
    "Rcvd": "Received",
    "Indiv": "Individual",
    "Cntrct": "Contract",
    "Bldg": "Building",
    "Bss": "Basis",
    "Tot": "Total",
    "Liab": "Liability",
    "Compr": "Compensation",
    "Orgn": "Organization",
    "Org": "Organization",
    "Chg": "Change",
    "Mgm": "Management",
    "Mgmt": "Management",
    "Tx": "Tax",
}

# Split rules: camelCase boundaries + digit boundaries + underscores/dashes/spaces
_TOKEN_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z])        |
    (?<=[A-Z])(?=[A-Z][a-z])   |
    (?<=[A-Za-z])(?=\d)        |
    (?<=\d)(?=[A-Za-z])        |
    [_\-\s]+
    """,
    re.VERBOSE,
)


def _strip_index(seg: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*)\[(\d+)\]$", seg)
    if not m:
        return seg, None
    return m.group(1), m.group(2)


def _clean_segment(seg: str) -> str:
    # drop leading '@' for naming; keep ':' as a separator
    seg2 = seg.lstrip("@").replace(":", " ")
    return seg2


def _tokenize(seg: str) -> List[str]:
    seg = _clean_segment(seg)
    if not seg:
        return []
    parts = [p for p in _TOKEN_SPLIT_RE.split(seg) if p and not p.isspace()]
    out: List[str] = []
    for p in parts:
        out.extend([x for x in p.split() if x])
    return out


def _titleish(token: str) -> str:
    """Title-case a token without lowercasing acronyms."""
    if token.isupper() and len(token) <= 6:
        return token
    return token[:1].upper() + token[1:]


def _apply_word_abbrev_map(token: str) -> List[str]:
    """
    Convert known abbreviations.
    If mapping includes spaces (e.g., 'Non Charitable') it returns multiple tokens.
    """
    mapped = _ABBREV_WORD_MAP.get(token, _ABBREV_WORD_MAP.get(token.capitalize()))
    if not mapped:
        return [token]
    return mapped.split()


def _apply_glossary(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        raw = t.strip()
        if not raw:
            continue

        # Preserve exact acronyms
        upper = raw.upper()
        if upper in _GLOSSARY_EXACT and raw.isalpha():
            out.append(_GLOSSARY_EXACT[upper])
            continue

        # Expand standalone glossary items (Ind/Amt/Cnt/Cd/Nm/Txt/Dt/Yr/etc.)
        if raw in _GLOSSARY_STANDALONE:
            out.append(_GLOSSARY_STANDALONE[raw])
            continue

        # Apply suffix expansions: CityNm -> City + Name, EOYAmt -> EOY + Amount
        expanded = False
        for suf, repl in _GLOSSARY_SUFFIX.items():
            if raw.endswith(suf) and len(raw) > len(suf):
                prefix = raw[: -len(suf)]
                prefix_tokens = _tokenize(prefix)
                prefix_tokens = _apply_glossary(prefix_tokens) if prefix_tokens else []
                out.extend(prefix_tokens)
                out.append(repl)
                expanded = True
                break
        if expanded:
            continue

        # Handle known mixed like FIN48 -> FIN 48
        if upper.startswith("FIN") and any(ch.isdigit() for ch in raw):
            m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
            if m:
                head = m.group(1).upper()
                tail = m.group(2)
                out.append(_GLOSSARY_EXACT.get(head, head))
                out.append(tail)
                continue

        # Apply word-level abbreviation mapping (Prtshp, Wthld, Rltd, ...)
        mapped_words = _apply_word_abbrev_map(raw)
        for mw in mapped_words:
            # preserve acronym again after mapping
            mw_upper = mw.upper()
            if mw_upper in _GLOSSARY_EXACT and mw.isalpha():
                out.append(_GLOSSARY_EXACT[mw_upper])
            else:
                out.append(_titleish(mw))

    return out


def humanize_segment(seg: str) -> str:
    base, idx = _strip_index(seg)
    tokens = _tokenize(base)
    tokens = _apply_glossary(tokens)

    # Special casing: keep "US" as US if it shows up
    tokens2: List[str] = []
    for tok in tokens:
        if tok.lower() == "us":
            tokens2.append("US")
        else:
            tokens2.append(tok)
    tokens = tokens2

    label = " ".join(tokens).strip() or base.lstrip("@")
    if idx is not None:
        label = f"{label} (Index {idx})"
    return label


def split_full_path(full_key: str) -> List[str]:
    return [p for p in full_key.split("/") if p]


# ============================================================
# CONTEXTUAL PREFIX RULES (Contributor / Fundraising, etc.)
# ============================================================
def _context_prefix_for_path(segments: List[str]) -> Optional[str]:
    """
    Return a prefix string (already humanized) if the path indicates a special context.
    We keep this minimal and targeted.
    """
    # Fundraising event info: IRS990ScheduleG/FundraisingEventInformationGrp/...
    if "FundraisingEventInformationGrp" in segments:
        return "Fundraising Event"

    # Contributor info: IRS990ScheduleB/ContributorInformationGrp/...
    # More specific first:
    if "ContributorUSAddress" in segments:
        return "Contributor US Address"
    if "ContributorInformationGrp" in segments:
        return "Contributor"

    return None


def build_candidate_name(path_segments: List[str], take_last_n: int) -> str:
    segs = path_segments[-take_last_n:]
    labels = [humanize_segment(s) for s in segs]
    return " - ".join([x for x in labels if x]).strip()


def build_column_name_map(full_row: Dict[str, str]) -> Dict[str, str]:
    """
    Map full XML keys -> final human-readable column names.

    Strategy:
      1) Build base name using last segment; if collision, expand to last 2, last 3...
      2) Apply contextual prefixes for contributor/fundraising when detected
      3) If still collides, suffix '__N'
    """
    used: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    mapping: Dict[str, str] = {}

    for full_key in sorted(full_row.keys()):
        segs = split_full_path(full_key)

        # Build a name that tries to remain short unless it collides
        final: Optional[str] = None
        for n in range(1, len(segs) + 1):
            cand = build_candidate_name(segs, take_last_n=n)
            if not cand:
                continue

            # Apply contextual prefix where appropriate
            prefix = _context_prefix_for_path(segs)
            if prefix:
                # Avoid doubling if cand already starts with the same prefix
                if not cand.startswith(prefix):
                    cand = f"{prefix} - {cand}"

            if cand not in used:
                final = cand
                break

        if final is None:
            final = humanize_segment(segs[-1]) if segs else full_key
            prefix = _context_prefix_for_path(segs)
            if prefix and not final.startswith(prefix):
                final = f"{prefix} - {final}"

        # Collision fallback suffix
        if final in used:
            counts[final] += 1
            suffixed = f"{final}__{counts[final]}"
            while suffixed in used:
                counts[final] += 1
                suffixed = f"{final}__{counts[final]}"
            final = suffixed

        used.add(final)
        mapping[full_key] = final

    return mapping


def apply_column_name_map(full_row: Dict[str, str], mapping: Dict[str, str]) -> Dict[str, str]:
    return {mapping[k]: v for k, v in full_row.items()}


def invert_map(col_map: Dict[str, str]) -> Dict[str, List[str]]:
    inv: Dict[str, List[str]] = defaultdict(list)
    for full_path, final_name in col_map.items():
        inv[final_name].append(full_path)
    return {k: sorted(v) for k, v in sorted(inv.items(), key=lambda kv: kv[0])}


# ============================================================
# COLUMN MANAGEMENT
# ============================================================
def normalize_row_to_columns(row: Dict[str, str], columns: Iterable[str]) -> Dict[str, str]:
    return {c: row.get(c, "") for c in columns}


def parse_990(
    xml_path: Path,
    *,
    opts: FlattenOptions = FlattenOptions(),
):
    """
    PURE FUNCTION.

    Returns:
      - row: {final_column_name: value}
      - meta: {
          "column_map": {full_xml_path: final_column_name},
          "columns": tuple[str, ...],
        }

    No file writing. No printing.
    """
    # 1) Full flatten (full XML paths)
    full_row = parse_xml_file_to_row(xml_path, opts=opts)

    # 2) Build human-readable column names
    col_map = build_column_name_map(full_row)

    # 3) Apply names
    renamed_row = apply_column_name_map(full_row, col_map)

    # 4) Stable column ordering
    columns = tuple(sorted(renamed_row.keys()))

    # 5) Normalize row to those columns
    normalized = normalize_row_to_columns(renamed_row, columns)

    meta = {
        "column_map": col_map,
        "columns": columns,
    }
    return normalized, meta


# Backwards-compat / typo-friendly alias (some scripts might refer to 900 by mistake)
def parse_900_main(xml_path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    return parse_990_main(xml_path)




# ============================================================
# MAIN-ONLY (REFINED) 990 PARSER
#   - Keeps ReturnHeader + ReturnData/IRS990
#   - Drops all IRS990Schedule* content
#   - Drops anything under repeating *Grp / *Group nodes that repeat
#   - No attributes
# ============================================================

_GROUP_TAG_SUFFIXES: Tuple[str, ...] = ("Grp", "Group")


def _strip_preamble_bytes(raw: bytes) -> bytes:
    """Remove any junk bytes before the first <?xml ...?> declaration."""
    i = raw.find(b"<?xml")
    return raw[i:] if i > 0 else raw


def _is_group_tag(tag: str) -> bool:
    return any(tag.endswith(suf) for suf in _GROUP_TAG_SUFFIXES)


def _is_main_990_path(path: str) -> bool:
    """
    Decide whether an XML *leaf path* belongs to the "main" 990 content.

    We keep:
      - /Return/ReturnHeader/...
      - /Return/ReturnData/IRS990/...

    We drop:
      - Anything containing an IRS990Schedule* segment
    """
    if path.startswith("/Return/ReturnHeader/"):
        return True

    if path.startswith("/Return/ReturnData/IRS990/"):
        # Exclude schedules (sometimes embedded or referenced)
        segs = split_full_path(path)
        if any(seg.startswith("IRS990Schedule") for seg in segs):
            return False
        return True

    return False


def build_main_schema_990(xml_path: Path) -> List[Dict[str, str]]:
    """
    Build a compact schema of "main" leaf paths for Form 990.

    Output rows:
      - {"column_name": <human label>, "xml_path": </Return/.../>}

    This mimics the strategy used in process_990PF:
      - discover leaf paths
      - remove anything under repeating group nodes (*Grp / *Group that repeat)
      - exclude schedules
    """
    raw = _strip_preamble_bytes(xml_path.read_bytes())
    bio = BytesIO(raw)

    stack: List[str] = []
    leaf_paths: Set[str] = set()
    repeat_counter: Dict[Tuple[str, str], int] = {}

    for event, elem in etree.iterparse(
        bio,
        events=("start", "end"),
        recover=True,
        huge_tree=True,
        remove_blank_text=True,
    ):
        if not isinstance(elem.tag, str):
            continue

        if event == "start":
            tag = local_name(elem.tag)
            parent = "/" + "/".join(stack) if stack else ""
            repeat_counter[(parent, tag)] = repeat_counter.get((parent, tag), 0) + 1
            stack.append(tag)

        else:
            # leaf node = no element children
            if not any(isinstance(c.tag, str) for c in elem):
                p = "/" + "/".join(stack)
                if _is_main_990_path(p):
                    leaf_paths.add(p)

            stack.pop()
            elem.clear()

    # Identify repeating group containers
    group_paths = {
        f"{p}/{t}" if p else f"/{t}"
        for (p, t), c in repeat_counter.items()
        if c > 1 and _is_group_tag(t)
    }

    def under_group(p: str) -> bool:
        return any(p.startswith(g + "/") for g in group_paths)

    main_paths = sorted(p for p in leaf_paths if not under_group(p))

    # Create stable, collision-safe labels using existing naming logic
    dummy = {p: "" for p in main_paths}
    xml_to_col = build_column_name_map(dummy)

    return [{"column_name": xml_to_col[p], "xml_path": p} for p in main_paths]


def extract_main_row_990(xml_path: Path, schema_rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Extract only the values described by the main schema."""
    raw = _strip_preamble_bytes(xml_path.read_bytes())
    root = etree.fromstring(raw, etree.XMLParser(recover=True, huge_tree=True))

    path_to_col = {r["xml_path"]: r["column_name"] for r in schema_rows}
    values: defaultdict[str, List[str]] = defaultdict(list)

    def walk(elem: etree._Element, path: List[str]) -> None:
        if not isinstance(elem.tag, str):
            return

        tag = local_name(elem.tag)
        cur_path = "/" + "/".join(path + [tag])

        if cur_path in path_to_col:
            txt = (elem.text or "").strip()
            if txt:
                values[path_to_col[cur_path]].append(txt)

        for child in elem:
            if isinstance(child.tag, str):
                walk(child, path + [tag])

    walk(root, [])

    return {k: " | ".join(v) for k, v in values.items()}


def parse_990_main(xml_path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    PURE FUNCTION.

    Compact Form 990 parser:
      - Only "main" (ReturnHeader + IRS990) leaf fields
      - No schedules
      - No repeating-group explosions

    Returns:
      - row: {column_name: value}
      - meta: {
          "column_map": {column_name: xml_path},
          "xml_to_column": {xml_path: column_name},
          "columns": tuple[str, ...],
          "schema": list[dict[str, str]],
        }

    No file writing. No printing.
    """
    schema = build_main_schema_990(xml_path)
    row = extract_main_row_990(xml_path, schema)

    columns = tuple(r["column_name"] for r in schema)
    normalized = normalize_row_to_columns(row, columns)

    meta: Dict[str, Any] = {
        "column_map": {r["column_name"]: r["xml_path"] for r in schema},
        "xml_to_column": {r["xml_path"]: r["column_name"] for r in schema},
        "columns": columns,
        "schema": schema,
    }
    return normalized, meta


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    full_row = parse_xml_file_to_row(FORMAT_XML_PATH)

    col_map = build_column_name_map(full_row)

    with OUTPUT_COLUMN_MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(col_map, f, ensure_ascii=False, indent=2, sort_keys=True)

    reverse_map = invert_map(col_map)
    with OUTPUT_COLUMN_MAP_REVERSE_JSON.open("w", encoding="utf-8") as f:
        json.dump(reverse_map, f, ensure_ascii=False, indent=2, sort_keys=True)

    renamed_row = apply_column_name_map(full_row, col_map)

    columns = tuple(sorted(renamed_row.keys()))
    normalized = normalize_row_to_columns(renamed_row, columns)

    # Single-row CSV export for inspection
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(normalized)

    print(f"✅ CSV written to: {OUTPUT_CSV_PATH}")
    print(f"✅ Column map written to: {OUTPUT_COLUMN_MAP_JSON}")
    print(f"✅ Reverse map written to: {OUTPUT_COLUMN_MAP_REVERSE_JSON}")
    print(f"Total columns: {len(columns)}")
