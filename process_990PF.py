from __future__ import annotations

import argparse
import csv
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from lxml import etree


# ============================================================
# CONFIG
# ============================================================
DEFAULT_XML_PATH = Path("format2.xml")
DEFAULT_OUT_DIR = Path("out_format2")

SKIP_TAGS_FOR_NAMING: Set[str] = {
    "Return",
    "ReturnData",
}

GROUP_TAG_SUFFIXES: Tuple[str, ...] = ("Grp", "Group")

REMOVE_PREFIXES = (
    "IRS990PF",
    "IRS990EZ",
    "IRS990",
)

IDX_RE = re.compile(r"\[\d+\]")
CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


# ============================================================
# HELPERS
# ============================================================
def strip_preamble(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="replace")
    i = text.find("<")
    if i == -1:
        raise ValueError("Invalid XML file")
    return text[i:].encode("utf-8")


def local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def camel_to_words(s: str) -> str:
    s = CAMEL_2.sub(r"\1 \2", s)
    s = CAMEL_1.sub(r"\1 \2", s)
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def is_group_tag(tag: str) -> bool:
    return any(tag.endswith(s) for s in GROUP_TAG_SUFFIXES)


def naming_parts_from_path(path: str) -> List[str]:
    parts = [p for p in IDX_RE.sub("", path).split("/") if p]
    return [p for p in parts if p not in SKIP_TAGS_FOR_NAMING]


def make_column_name(parts: List[str]) -> str:
    """
    Human-readable column name:
    - Use leaf tag only
    - Remove IRS form prefixes
    - CamelCase → words
    - Title Case
    """
    if not parts:
        return "Value"

    name = parts[-1]

    for p in REMOVE_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]

    name = CAMEL_2.sub(r"\1 \2", name)
    name = CAMEL_1.sub(r"\1 \2", name)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    return name.title()


# ============================================================
# PHASE 1 — BUILD MAIN SCHEMA
# ============================================================
def build_main_schema(xml_path: Path) -> List[Dict[str, str]]:
    raw = strip_preamble(xml_path.read_bytes())
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
            if not any(isinstance(c.tag, str) for c in elem):
                leaf_paths.add("/" + "/".join(stack))
            stack.pop()
            elem.clear()

    group_paths = {
        f"{p}/{t}" if p else f"/{t}"
        for (p, t), c in repeat_counter.items()
        if c > 1 and is_group_tag(t)
    }

    def under_group(p: str) -> bool:
        return any(p.startswith(g + "/") for g in group_paths)

    main_paths = sorted(p for p in leaf_paths if not under_group(p))

    rows: List[Dict[str, str]] = []
    used_cols: Set[str] = set()

    for p in main_paths:
        parts = naming_parts_from_path(p)
        col = make_column_name(parts)

        base = col
        i = 2
        while col in used_cols:
            col = f"{base} {i}"
            i += 1

        used_cols.add(col)

        rows.append(
            {
                "column_name": col,
                "xml_path": p,
            }
        )

    return rows


# ============================================================
# PHASE 2 — EXTRACT VALUES
# ============================================================
def extract_main_row(xml_path: Path, schema_rows: List[Dict[str, str]]) -> Dict[str, str]:
    raw = strip_preamble(xml_path.read_bytes())
    root = etree.fromstring(raw, etree.XMLParser(recover=True, huge_tree=True))

    path_to_col = {r["xml_path"]: r["column_name"] for r in schema_rows}
    values: Dict[str, List[str]] = {c: [] for c in path_to_col.values()}

    def walk(elem: etree._Element, path: List[str]):
        tag = local_name(elem.tag)
        cur_path = "/" + "/".join(path + [tag])

        if cur_path in path_to_col:
            text = (elem.text or "").strip()
            if text:
                values[path_to_col[cur_path]].append(text)

        for child in elem:
            if isinstance(child.tag, str):
                walk(child, path + [tag])

    walk(root, [])

    return {k: " | ".join(v) for k, v in values.items()}


# ============================================================
# OUTPUT
# ============================================================
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")



def parse_990PF(xml_path: Path) -> Tuple[Dict[str, str], Dict[str, object]]:
    """
    PURE FUNCTION.

    Given an XML file, returns:
      - row: {column_name: value}
      - meta: {
          "column_map": {column_name: xml_path},
          "columns": tuple[str, ...],
          "schema": list[dict[str, str]],
        }

    No file writing. No printing.
    """
    schema = build_main_schema(xml_path)
    row = extract_main_row(xml_path, schema)

    columns = tuple(r["column_name"] for r in schema)
    meta: Dict[str, object] = {
        "column_map": {r["column_name"]: r["xml_path"] for r in schema},
        "columns": columns,
        "schema": schema,
    }
    return row, meta



# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Extract main XML fields into a single-row CSV")
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    schema = build_main_schema(args.xml)
    row = extract_main_row(args.xml, schema)

    write_csv(out / "main_columns.csv", schema)
    write_json(out / "main_columns.json", schema)

    write_csv(out / "main_data.csv", [row])
    write_json(out / "main_data.json", row)

    print(f"[done] columns: {len(schema)}")
    print("[done] data extracted successfully")


if __name__ == "__main__":
    main()
