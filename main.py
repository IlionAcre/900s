from __future__ import annotations
import asyncio
import csv
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, NamedTuple, Optional
from urllib.parse import urljoin

import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, X

from curl_cffi.requests import AsyncSession


from scraper_unified import (
    BASE,
    scrape_search,
    sanitize_xml_bytes,
    RateLimitGate,
    fetch,
    fetch_bytes,
    DEFAULT_HEADERS,
)
from column_selector import select_columns, get_columns_for_type
from process_990 import parse_990
from process_990PF import parse_990PF
from process_historical_data import (
    extract_xml_link_from_html_text,
    is_990pf_from_html_text,
    parse_historical_data_from_html_text,
)


"""
Modes:
1) Single query:
   - User types a query and clicks Run
   - File name is ALWAYS derived automatically:
       <query>_990.csv or <query>_990pf.csv (depending on detected filing type)
   - If file already exists, the program APPENDS a new row to it.

2) Multi query:
   - Reads queries from ./queries.txt (project root)
   - Outputs TWO CSVs:
       <base>_990.csv (only 990 rows)
       <base>_990pf.csv (only 990-PF rows)
   - User can edit <base> in UI
   - If output files exist, the program APPENDS new rows to them.
   - Wait time between queries is configurable (helps avoid rate limiting).

Per query pipeline:
- Query -> scrape_search -> pick first result
- Fetch org HTML
- Parse historical data from HTML
- Extract XML link from first filing section
- Download XML
- Detect 990 vs 990-PF
- Parse with correct parser
- Merge rows
"""

# ============================================================
# Helpers
# ============================================================


def _safe_merge_rows(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dict rows. If a key collides, b's value gets a suffix."""
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        i = 2
        nk = f"{k}__{i}"
        while nk in out:
            i += 1
            nk = f"{k}__{i}"
        out[nk] = v
    return out


def _sanitize_filename_stem(s: str) -> str:
    """Make a readable, filesystem-safe filename stem."""
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:80] or "query"


def _read_queries_file(path: Path) -> List[str]:
    """
    Read queries from a text file or CSV file:
    - For CSV: reads first column as query
    - For TXT: one query per line
    - ignore blank lines
    - ignore lines starting with '#'
    - handles all line ending styles (\n, \r\n, \r) for Windows/Mac/Linux
    """
    if not path.exists():
        raise RuntimeError(f"Queries file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    # Remove BOM if present (common in Windows-edited files)
    content = content.lstrip('\ufeff')
    # Normalize line endings: \r\n -> \n, then \r -> \n
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    out: List[str] = []
    
    if path.suffix.lower() == ".csv":
        # CSV format: first column is query
        lines = content.split('\n')
        reader = csv.reader(lines)
        for row in reader:
            if row and row[0].strip() and not row[0].strip().startswith('#'):
                out.append(row[0].strip())
    else:
        # TXT format: one query per line
        lines = content.split('\n')
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    
    if not out:
        raise RuntimeError("Queries file is empty (no valid queries found).")
    return out



def build_preferred_column_order(
    *,
    historical_cols: List[str],
    rows: List[Dict[str, Any]],
) -> List[str]:
    """
    Build a deterministic column order that guarantees historical columns
    appear first (preserving the given order), followed by any remaining
    columns in first-seen order across the provided rows.

    This is used to keep CSV headers stable and human-friendly.
    """
    ordered: List[str] = []
    seen: set[str] = set()

    for c in historical_cols or []:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                ordered.append(k)

    return ordered


def _append_rows_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    preferred_columns: List[str] | None = None,
) -> None:
    """
    Append rows to CSV.
    - If file doesn't exist or is empty: write header first.
    - If file exists: append rows using the existing header.
    - Missing/blank cells become '-'.

    NOTE:
      If new columns appear in later runs, we do NOT rewrite the existing header
      (that would require rewriting the whole file). We append using the existing header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    file_exists = path.exists()
    existing_header: List[str] = []

    if file_exists:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
        except Exception:
            existing_header = []

    incoming_cols = {k for r in rows for k in r.keys()}

    # If a preferred order is provided, honor it (historical columns first),
    # then append any remaining columns deterministically.
    if preferred_columns:
        all_cols: List[str] = [c for c in preferred_columns if c in incoming_cols or c in existing_header]
        for c in list(existing_header) + sorted(incoming_cols):
            if c not in all_cols:
                all_cols.append(c)
    else:
        all_cols = list(existing_header)
        for c in sorted(incoming_cols):
            if c not in all_cols:
                all_cols.append(c)


    write_header = (not file_exists) or (path.stat().st_size == 0) or (not existing_header)

    with path.open("a", newline="", encoding="utf-8") as f:
        if write_header:
            writer_cols = all_cols
            w = csv.DictWriter(f, fieldnames=writer_cols)
            w.writeheader()
        else:
            writer_cols = existing_header
            w = csv.DictWriter(f, fieldnames=writer_cols)

        for r in rows:
            w.writerow({k: ("-" if r.get(k) in (None, "") else r.get(k)) for k in writer_cols})


async def run_one_query_to_row(
    *,
    query: str,
    output_dir: Path,
    session: Optional[AsyncSession] = None,
    gate: Optional[RateLimitGate] = None,
) -> Tuple[Dict[str, Any], str, List[str]] | None:
    """
    Run pipeline for ONE query and return:
      - combined_row
      - filing_type: "990" or "990-PF"
      - hist_columns: ordered list of historical-data columns (for stable CSV headers)
    Returns None if no results found (not an error).
    
    If session and gate are provided, they are reused (for session sharing).
    """
    # Use provided session or create a new one
    if session is not None:
        results = await scrape_search(query, max_pages=1, delay_s=0.0, session=session, gate=gate)
        if not results:
            return None  # No results found, continue to next query

        org = results[0]
        org_url = org.org_url

        html_text = await fetch(session, org_url, gate=gate)

        hist_row, _hist_meta = parse_historical_data_from_html_text(html_text)
        hist_columns = list(_hist_meta.get("columns", tuple(hist_row.keys())))

        xml_href = extract_xml_link_from_html_text(html_text)
        xml_url = urljoin(BASE, xml_href)

        is_pf = is_990pf_from_html_text(html_text)
        filing_type = "990-PF" if is_pf else "990"

        xml_bytes = await fetch_bytes(session, xml_url, gate=gate)
    else:
        # Legacy behavior: create new session
        results = await scrape_search(query, max_pages=1, delay_s=0.0)
        if not results:
            return None  # No results found, continue to next query

        org = results[0]
        org_url = org.org_url

        async with AsyncSession(
            impersonate="chrome",
            timeout=60,
            headers=DEFAULT_HEADERS,
        ) as new_session:
            html_text = await fetch(new_session, org_url)

            hist_row, _hist_meta = parse_historical_data_from_html_text(html_text)
            hist_columns = list(_hist_meta.get("columns", tuple(hist_row.keys())))

            xml_href = extract_xml_link_from_html_text(html_text)
            xml_url = urljoin(BASE, xml_href)

            is_pf = is_990pf_from_html_text(html_text)
            filing_type = "990-PF" if is_pf else "990"

            xml_bytes = await fetch_bytes(new_session, xml_url)

    # Fix broken XMLs (naked '&', etc.)
    xml_bytes = sanitize_xml_bytes(xml_bytes)

    tmp_xml_path = output_dir / f"__tmp_download_{_sanitize_filename_stem(query)}.xml"
    tmp_xml_path.write_bytes(xml_bytes)

    try:
        if filing_type == "990-PF":
            xml_row, _xml_meta = parse_990PF(tmp_xml_path)
        else:
            xml_row, _xml_meta = parse_990(tmp_xml_path)

        combined = _safe_merge_rows(hist_row, xml_row)

        combined = _safe_merge_rows(
            {
                "input_query": query,
                "org_name": org.org_name,
                "org_url": org.org_url,
                "org_id": org.org_id or "",
                "city": org.city or "",
                "state": org.state or "",
                "category_text": org.category_text or "",
                "revenue_year": org.revenue_year or "",
                "revenue_value": org.revenue_value or "",
                "filing_type": filing_type,
                "xml_url": xml_url,
            },
            combined,
        )
        return combined, filing_type, hist_columns
    finally:
        try:
            tmp_xml_path.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# UI
# ============================================================


class App(tb.Frame):
    def __init__(self, master: tb.Window) -> None:
        super().__init__(master, padding=18)
        self.master = master

        self.project_root = Path(__file__).resolve().parent
        self.default_data_dir = self.project_root / "data"
        self.default_queries_file = self.project_root / "queries.csv"

        # mode
        self.var_mode = tk.StringVar(value="single")  # "single" or "multi"

        # single
        self.var_query = tk.StringVar(value="")
        self.var_single_preview_990 = tk.StringVar(value="(type a query)")
        self.var_single_preview_990pf = tk.StringVar(value="(type a query)")

        # output dir (shared)
        self.var_output_dir = tk.StringVar(value=str(self.default_data_dir))

        # multi
        self.var_queries_file = tk.StringVar(value=str(self.default_queries_file))
        self.var_multi_base_name = tk.StringVar(value="results")  # stem only
        self.var_wait_seconds = tk.StringVar(value="0")  # wait between queries (default: 0)
        self.var_batch_size = tk.StringVar(value="50")  # queries per batch
        self.var_batch_wait_minutes = tk.StringVar(value="1.02")  # 61 seconds = ~1.02 min

        # export options (shared)
        self.var_export_full = tk.BooleanVar(value=False)  # Default: cleaned only
        self.var_report_name = tk.StringVar(value="report")  # Report file base name

        self.var_status = tk.StringVar(value="Ready.")
        self._run_btn: tb.Button | None = None
        self._stop_btn: tb.Button | None = None
        self._stop_requested: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.master.title("Nonprofit Filing Exporter")
        self.master.minsize(920, 590)
        self.pack(fill=BOTH, expand=True)

        header = tb.Frame(self)
        header.pack(fill=X, pady=(0, 12))

        tb.Label(header, text="Search & Export", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tb.Label(
            header,
            text="Single exports one CSV (auto-named, appends if exists). Multi reads ./queries.txt and exports two CSVs (appends).",
            font=("Segoe UI", 10),
            bootstyle="secondary",
        ).pack(anchor="w", pady=(4, 0))

        # Mode
        mode_card = tb.Labelframe(self, text="Mode", padding=14)
        mode_card.pack(fill=X, pady=(0, 12))

        row_mode = tb.Frame(mode_card)
        row_mode.pack(fill=X)

        tb.Radiobutton(
            row_mode,
            text="Single query",
            value="single",
            variable=self.var_mode,
            bootstyle="success-outline-toolbutton",
            command=self._refresh_mode,
        ).pack(side=LEFT, padx=(0, 10))

        tb.Radiobutton(
            row_mode,
            text="Multi query (from file)",
            value="multi",
            variable=self.var_mode,
            bootstyle="success-outline-toolbutton",
            command=self._refresh_mode,
        ).pack(side=LEFT)

        # Content container
        self.content = tb.Frame(self)
        self.content.pack(fill=BOTH, expand=True)

        # ---- Single frame
        self.single_frame = tb.Frame(self.content)
        self.single_frame.pack(fill=BOTH, expand=True)

        card_search = tb.Labelframe(self.single_frame, text="Single query", padding=14)
        card_search.pack(fill=X, pady=(0, 12))

        row_query = tb.Frame(card_search)
        row_query.pack(fill=X)

        tb.Label(row_query, text="Query", width=10).pack(side=LEFT, padx=(0, 10))
        ent_query = tb.Entry(row_query, textvariable=self.var_query)
        ent_query.pack(side=LEFT, fill=X, expand=True)
        ent_query.bind("<KeyRelease>", lambda _e: self._update_single_previews())
        ent_query.focus_set()

        # output folder (shared)
        card_out_single = tb.Labelframe(self.single_frame, text="Output", padding=14)
        card_out_single.pack(fill=X)

        row_folder = tb.Frame(card_out_single)
        row_folder.pack(fill=X, pady=(0, 10))

        tb.Label(row_folder, text="Folder", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_folder, textvariable=self.var_output_dir).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_folder, text="Choose…", bootstyle="secondary", command=self.on_choose_folder).pack(
            side=LEFT, padx=(10, 0)
        )

        # preview names (read-only labels)
        preview_box = tb.Frame(card_out_single)
        preview_box.pack(fill=X)

        tb.Label(preview_box, text="Will save as:", width=10).pack(side=LEFT, padx=(0, 10))
        prev_text = tb.Frame(preview_box)
        prev_text.pack(side=LEFT, fill=X, expand=True)

        tb.Label(prev_text, textvariable=self.var_single_preview_990, bootstyle="secondary").pack(anchor="w")
        tb.Label(prev_text, textvariable=self.var_single_preview_990pf, bootstyle="secondary").pack(anchor="w")

        tb.Label(
            card_out_single,
            text="The program chooses 990 vs 990-PF automatically and APPENDS to the file if it already exists.",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(10, 0))

        # Export full file checkbox (single mode)
        tb.Checkbutton(
            card_out_single,
            text="Also export full (uncleaned) file",
            variable=self.var_export_full,
            bootstyle="round-toggle",
        ).pack(anchor="w", pady=(8, 0))

        # ---- Multi frame
        self.multi_frame = tb.Frame(self.content)

        card_multi = tb.Labelframe(self.multi_frame, text="Multi query", padding=14)
        card_multi.pack(fill=X, pady=(0, 12))

        row_qfile = tb.Frame(card_multi)
        row_qfile.pack(fill=X, pady=(0, 10))

        tb.Label(row_qfile, text="Queries file", width=12).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_qfile, textvariable=self.var_queries_file).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_qfile, text="Choose…", bootstyle="secondary", command=self.on_choose_queries_file).pack(
            side=LEFT, padx=(10, 0)
        )

        tb.Label(
            card_multi,
            text="CSV format: first column is query. TXT also supported. Blank lines and # lines are ignored.",
            bootstyle="secondary",
        ).pack(anchor="w")

        card_out_multi = tb.Labelframe(self.multi_frame, text="Output", padding=14)
        card_out_multi.pack(fill=X)

        row_folder2 = tb.Frame(card_out_multi)
        row_folder2.pack(fill=X, pady=(0, 10))

        tb.Label(row_folder2, text="Folder", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_folder2, textvariable=self.var_output_dir).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_folder2, text="Choose…", bootstyle="secondary", command=self.on_choose_folder).pack(
            side=LEFT, padx=(10, 0)
        )

        row_base = tb.Frame(card_out_multi)
        row_base.pack(fill=X, pady=(0, 8))
        tb.Label(row_base, text="Base name", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_base, textvariable=self.var_multi_base_name).pack(side=LEFT, fill=X, expand=True)

        row_wait = tb.Frame(card_out_multi)
        row_wait.pack(fill=X, pady=(0, 8))
        tb.Label(row_wait, text="Wait", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_wait, textvariable=self.var_wait_seconds, width=4).pack(side=LEFT)
        tb.Label(row_wait, text="sec between queries", bootstyle="secondary").pack(side=LEFT, padx=(4, 0))

        row_batch = tb.Frame(card_out_multi)
        row_batch.pack(fill=X, pady=(0, 8))
        tb.Label(row_batch, text="Batch", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_batch, textvariable=self.var_batch_size, width=4).pack(side=LEFT)
        tb.Label(row_batch, text="queries, then wait", bootstyle="secondary").pack(side=LEFT, padx=(4, 0))
        tb.Entry(row_batch, textvariable=self.var_batch_wait_minutes, width=4).pack(side=LEFT, padx=(4, 0))
        tb.Label(row_batch, text="min", bootstyle="secondary").pack(side=LEFT, padx=(4, 0))

        tb.Label(
            card_out_multi,
            text="This produces TWO files: <base>_990.csv and <base>_990pf.csv. It APPENDS if files already exist.",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(10, 0))

        # Export full file checkbox (multi mode)
        tb.Checkbutton(
            card_out_multi,
            text="Also export full (uncleaned) files",
            variable=self.var_export_full,
            bootstyle="round-toggle",
        ).pack(anchor="w", pady=(8, 0))

        # Report name
        row_report = tb.Frame(card_out_multi)
        row_report.pack(fill=X, pady=(8, 0))
        tb.Label(row_report, text="Report", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_report, textvariable=self.var_report_name).pack(side=LEFT, fill=X, expand=True)
        tb.Label(row_report, text="(summary file with date)", bootstyle="secondary").pack(side=LEFT, padx=(8, 0))

        # Bottom bar
        bottom = tb.Frame(self)
        bottom.pack(fill=X, pady=(16, 0))

        tb.Label(bottom, textvariable=self.var_status, bootstyle="secondary").pack(side=LEFT)

        self._run_btn = tb.Button(bottom, text="Run", bootstyle="primary", width=12, command=self.on_run)
        self._run_btn.pack(side=RIGHT)

        self._stop_btn = tb.Button(bottom, text="Stop", bootstyle="danger", width=12, command=self.on_stop, state="disabled")
        self._stop_btn.pack(side=RIGHT, padx=(0, 8))

        self.master.bind("<Return>", lambda _e: self.on_run())

        # init
        self._refresh_mode()
        self._update_single_previews()

    # ============================================================
    # Mode helpers
    # ============================================================
    def _refresh_mode(self) -> None:
        if self.var_mode.get() == "single":
            self.multi_frame.pack_forget()
            self.single_frame.pack(fill=BOTH, expand=True)
            self.var_status.set("Ready (Single query).")
        else:
            self.single_frame.pack_forget()
            self.multi_frame.pack(fill=BOTH, expand=True)
            self.var_status.set("Ready (Multi query).")

    def _update_single_previews(self) -> None:
        stem = _sanitize_filename_stem(self.var_query.get())
        self.var_single_preview_990.set(f"{stem}_990.csv")
        self.var_single_preview_990pf.set(f"{stem}_990pf.csv")

    # ============================================================
    # UI actions
    # ============================================================
    def on_choose_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose output folder")
        if chosen:
            self.var_output_dir.set(chosen)
            self.var_status.set("Output folder updated.")

    def on_choose_queries_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose queries file",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if chosen:
            self.var_queries_file.set(chosen)
            self.var_status.set("Queries file updated.")

    def on_stop(self) -> None:
        """Request stop of the current scraping operation."""
        self._stop_requested = True
        self.var_status.set("Stopping... (saving progress)")
        if self._stop_btn is not None:
            self._stop_btn.configure(state="disabled")

    def on_run(self) -> None:
        settings = self._collect_settings()

        if settings["mode"] == "single":
            if not settings["query"].strip():
                messagebox.showwarning("Missing query", "Please type an organization name.")
                return
        else:
            qfile = Path(settings["queries_file"])
            if not qfile.exists():
                messagebox.showwarning("Missing queries file", f"Queries file not found:\n\n{qfile}")
                return

        # Reset stop flag and toggle buttons
        self._stop_requested = False
        if self._run_btn is not None:
            self._run_btn.configure(state="disabled")
        if self._stop_btn is not None:
            self._stop_btn.configure(state="normal")
        self.var_status.set("Working...")

        threading.Thread(target=self._run_in_thread, args=(settings,), daemon=True).start()

    def _run_in_thread(self, settings: dict[str, str]) -> None:
        try:
            output_dir = Path(settings["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            if settings["mode"] == "single":
                csv_path, filing_type = asyncio.run(
                    self._run_single(
                        query=settings["query"],
                        output_dir=output_dir,
                        export_full=self.var_export_full.get(),
                    )
                )
                self.master.after(0, lambda p=csv_path, t=filing_type: self._on_success_single(p, t))
            else:
                wait_s = self._parse_wait_seconds(settings["wait_seconds"])
                batch_size = max(1, int(self.var_batch_size.get() or "50"))
                batch_wait_s = self._parse_wait_seconds(self.var_batch_wait_minutes.get()) * 60  # minutes to seconds
                result = asyncio.run(
                    self._run_multi(
                        queries_file=Path(settings["queries_file"]),
                        output_dir=output_dir,
                        base_name=settings["multi_base_name"],
                        wait_seconds=wait_s,
                        batch_size=batch_size,
                        batch_wait_seconds=batch_wait_s,
                        export_full=self.var_export_full.get(),
                        report_name=self.var_report_name.get().strip() or "report",
                    )
                )
                self.master.after(0, lambda r=result: self._on_success_multi(r))

        except Exception as err:
            # capture err safely for later execution
            self.master.after(0, lambda e=err: self._on_error(e))

    def _parse_wait_seconds(self, s: str) -> float:
        try:
            v = float(s)
            return max(0.0, v)
        except Exception:
            return 0.0

    async def _run_single(self, *, query: str, output_dir: Path, export_full: bool) -> Tuple[Path, str]:
        row, filing_type, hist_columns = await run_one_query_to_row(query=query, output_dir=output_dir)

        stem = _sanitize_filename_stem(query)
        # Full file (uncleaned)
        full_filename = f"{stem}_990pf_full.csv" if filing_type == "990-PF" else f"{stem}_990_full.csv"
        full_csv_path = output_dir / full_filename

        preferred_cols = build_preferred_column_order(historical_cols=hist_columns, rows=[row])
        _append_rows_csv(full_csv_path, [row], preferred_columns=preferred_cols)

        # Cleaned file (selected columns only)
        cleaned_filename = f"{stem}_990pf.csv" if filing_type == "990-PF" else f"{stem}_990.csv"
        cleaned_csv_path = output_dir / cleaned_filename
        file_type = "990pf" if filing_type == "990-PF" else "990"
        select_columns(full_csv_path, cleaned_csv_path, get_columns_for_type(file_type))

        # If export_full is False, remove the full file
        if not export_full:
            try:
                full_csv_path.unlink(missing_ok=True)
            except Exception:
                pass

        return cleaned_csv_path, filing_type

    async def _interruptible_sleep(
        self,
        total_seconds: float,
        status_format: str,
        check_interval: float = 1.0,
    ) -> bool:
        """
        Sleep for total_seconds, updating status with countdown every check_interval.
        Returns True if completed, False if interrupted by stop request.
        
        status_format should contain {remaining} placeholder, e.g.:
            "({i}/{n}) Waiting {remaining}s..."
        """
        remaining = total_seconds
        while remaining > 0:
            if self._stop_requested:
                return False
            
            # Update status with remaining time
            if remaining >= 60:
                time_str = f"{remaining/60:.1f} min"
            else:
                time_str = f"{int(remaining)}s"
            
            self.master.after(0, lambda t=time_str, fmt=status_format: self.var_status.set(
                fmt.format(remaining=t)
            ))
            
            # Sleep for check_interval or remaining time, whichever is smaller
            sleep_time = min(check_interval, remaining)
            await asyncio.sleep(sleep_time)
            remaining -= sleep_time
        
        return True

    async def _run_multi(
        self,
        *,
        queries_file: Path,
        output_dir: Path,
        base_name: str,
        wait_seconds: float,
        batch_size: int,
        batch_wait_seconds: float,
        export_full: bool,
        report_name: str,
    ) -> Dict[str, Any]:
        """Run multi-query and return summary dict with counts and paths."""
        queries = _read_queries_file(queries_file)

        rows_990: List[Dict[str, Any]] = []
        rows_990pf: List[Dict[str, Any]] = []
        hist_columns_first: List[str] = []

        # Track results for report
        results_log: List[Tuple[str, str, str]] = []  # (query, status, error)
        count_success = 0
        count_no_results = 0
        count_errors = 0

        consecutive_429s = 0  # Track consecutive rate-limit failures
        ip_rate_limited = False  # Flag to break out and save
        stopped = False  # Track if user stopped

        # Process queries in batches, each batch gets a fresh session
        total_queries = len(queries)
        processed_count = 0

        # Calculate batch boundaries
        batch_starts = list(range(0, total_queries, batch_size))

        for batch_idx, batch_start in enumerate(batch_starts):
            if ip_rate_limited or stopped or self._stop_requested:
                if self._stop_requested:
                    stopped = True
                break

            batch_end = min(batch_start + batch_size, total_queries)
            batch_queries = queries[batch_start:batch_end]
            batch_num = batch_idx + 1
            total_batches = len(batch_starts)

            # Show batch start message
            self.master.after(0, lambda b=batch_num, t=total_batches: self.var_status.set(
                f"Starting batch {b}/{t}... (new session)"
            ))

            # Create a fresh session and gate for this batch
            gate = RateLimitGate()

            async with AsyncSession(
                impersonate="chrome",
                timeout=60,
                headers=DEFAULT_HEADERS,
            ) as session:
                for j, q in enumerate(batch_queries):
                    # Check stop flag at start of each query
                    if self._stop_requested:
                        stopped = True
                        break
                    if ip_rate_limited:
                        break

                    i = batch_start + j + 1  # Overall query index (1-based)
                    self.master.after(0, lambda q=q, i=i, n=total_queries: self.var_status.set(
                        f"Working... ({i}/{n}) {q}"
                    ))

                    # Query execution with shared session - retry is handled at request level
                    result = None
                    last_error = None
                    got_429 = False

                    try:
                        result = await run_one_query_to_row(
                            query=q,
                            output_dir=output_dir,
                            session=session,
                            gate=gate,
                        )
                        got_429 = False
                        consecutive_429s = 0  # Reset on success
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                            got_429 = True
                            last_error = e
                        else:
                            # Non-429 error
                            last_error = e
                            got_429 = False

                    # Process result
                    if got_429 and last_error:
                        consecutive_429s += 1
                        results_log.append((q, "rate_limited", str(last_error)))
                        count_errors += 1

                        if consecutive_429s >= 2:
                            # Two consecutive 429s = IP rate limited, save what we have and abort
                            ip_rate_limited = True
                    elif last_error:
                        # Non-429 error
                        consecutive_429s = 0
                        results_log.append((q, "error", str(last_error)))
                        count_errors += 1
                    elif result is None:
                        # No results found - not an error
                        consecutive_429s = 0
                        results_log.append((q, "no_results", ""))
                        count_no_results += 1
                    else:
                        # Success
                        row, filing_type, hist_columns = result
                        if not hist_columns_first and hist_columns:
                            hist_columns_first = hist_columns

                        if filing_type == "990-PF":
                            rows_990pf.append(row)
                        else:
                            rows_990.append(row)

                        results_log.append((q, "success", filing_type))
                        count_success += 1

                    processed_count += 1

                    if ip_rate_limited or stopped:
                        break

                    # Wait between queries (within batch) - interruptible with countdown
                    if j < len(batch_queries) - 1 and wait_seconds > 0:
                        wait_completed = await self._interruptible_sleep(
                            wait_seconds,
                            f"({i}/{total_queries}) Waiting {{remaining}}...",
                            check_interval=1.0,
                        )
                        if not wait_completed:
                            stopped = True
                            break

            # Session is now closed for this batch

            # Batch wait before next batch (if not the last batch and not stopped) - interruptible
            if batch_idx < len(batch_starts) - 1 and batch_wait_seconds > 0:
                if not ip_rate_limited and not stopped and not self._stop_requested:
                    wait_completed = await self._interruptible_sleep(
                        batch_wait_seconds,
                        f"Batch {batch_num} complete. {{remaining}} until next batch...",
                        check_interval=1.0,
                    )
                    if not wait_completed:
                        stopped = True

        stem = _sanitize_filename_stem(base_name) or "results"

        # Only write CSVs if we have rows
        cleaned_csv_990 = output_dir / f"{stem}_990.csv"
        cleaned_csv_990pf = output_dir / f"{stem}_990pf.csv"

        if rows_990 or rows_990pf:
            # Full files (uncleaned)
            full_csv_990 = output_dir / f"{stem}_990_full.csv"
            full_csv_990pf = output_dir / f"{stem}_990pf_full.csv"

            preferred_cols = build_preferred_column_order(
                historical_cols=hist_columns_first,
                rows=rows_990 + rows_990pf,
            )

            if rows_990:
                _append_rows_csv(full_csv_990, rows_990, preferred_columns=preferred_cols)
                select_columns(full_csv_990, cleaned_csv_990, get_columns_for_type("990"))
                if not export_full:
                    try:
                        full_csv_990.unlink(missing_ok=True)
                    except Exception:
                        pass

            if rows_990pf:
                _append_rows_csv(full_csv_990pf, rows_990pf, preferred_columns=preferred_cols)
                select_columns(full_csv_990pf, cleaned_csv_990pf, get_columns_for_type("990pf"))
                if not export_full:
                    try:
                        full_csv_990pf.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Write report file
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = output_dir / f"{_sanitize_filename_stem(report_name)}_{date_str}.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Status", "Details"])
            for query, status, details in results_log:
                writer.writerow([query, status, details])

        return {
            "total": len(queries),
            "processed": processed_count,
            "success": count_success,
            "no_results": count_no_results,
            "errors": count_errors,
            "csv_990": cleaned_csv_990 if rows_990 else None,
            "csv_990pf": cleaned_csv_990pf if rows_990pf else None,
            "report": report_path,
            "ip_rate_limited": ip_rate_limited,
            "stopped": stopped,
        }

    def _on_success_single(self, csv_path: Path, filing_type: str) -> None:
        self.var_status.set(f"Done. Exported {filing_type} → {csv_path.name}")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        if self._stop_btn is not None:
            self._stop_btn.configure(state="disabled")
        messagebox.showinfo("Success", f"Export complete.\n\nSaved/appended: {csv_path}")

    def _on_success_multi(self, result: Dict[str, Any]) -> None:
        self.var_status.set("Done. Exported multi-query CSVs.")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        if self._stop_btn is not None:
            self._stop_btn.configure(state="disabled")

        # Build summary message
        msg_lines = []
        if result.get("stopped"):
            msg_lines.append("⏹️ Stopped by user.")
            msg_lines.append("Partial results saved.\n")
        elif result.get("ip_rate_limited"):
            msg_lines.append("⚠️ Your IP has been rate limited.")
            msg_lines.append("Partial results saved. Please try again later.\n")
        else:
            msg_lines.append("Multi export complete.\n")

        msg_lines.append(f"Total queries: {result['total']}")
        msg_lines.append(f"Processed: {result.get('processed', result['success'] + result['no_results'] + result['errors'])}")
        msg_lines.append(f"Success: {result['success']}")
        msg_lines.append(f"No results: {result['no_results']}")
        msg_lines.append(f"Errors: {result['errors']}")
        msg_lines.append("")

        if result.get("csv_990"):
            msg_lines.append(f"990: {result['csv_990'].name}")
        if result.get("csv_990pf"):
            msg_lines.append(f"990-PF: {result['csv_990pf'].name}")

        msg_lines.append(f"\nReport: {result['report'].name}")

        title = "Stopped" if result.get("stopped") else ("Rate Limited" if result.get("ip_rate_limited") else "Summary")
        messagebox.showinfo(title, "\n".join(msg_lines))

    def _on_error(self, err: Exception) -> None:
        self.var_status.set("Error.")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        if self._stop_btn is not None:
            self._stop_btn.configure(state="disabled")
        messagebox.showerror("Error", str(err))

    def _collect_settings(self) -> dict[str, str]:
        return {
            "mode": self.var_mode.get(),
            "query": self.var_query.get().strip(),
            "output_dir": self.var_output_dir.get().strip(),
            "queries_file": self.var_queries_file.get().strip(),
            "multi_base_name": (self.var_multi_base_name.get().strip() or "results"),
            "wait_seconds": self.var_wait_seconds.get().strip(),
        }


def main() -> None:
    app = tb.Window(themename="flatly")
    App(app)
    app.mainloop()


if __name__ == "__main__":
    main()
